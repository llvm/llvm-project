//===-- OpenMP.cpp -- Open MP directive lowering --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenMP.h"

#include "ClauseProcessor.h"
#include "Clauses.h"
#include "DataSharingProcessor.h"
#include "Decomposer.h"
#include "ReductionProcessor.h"
#include "Utils.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/DirectivesCommon.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/tools.h"
#include "flang/Support/OpenMP-utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace Fortran::lower::omp;
using namespace Fortran::common::openmp;

//===----------------------------------------------------------------------===//
// Code generation helper functions
//===----------------------------------------------------------------------===//

static void genOMPDispatch(lower::AbstractConverter &converter,
                           lower::SymMap &symTable,
                           semantics::SemanticsContext &semaCtx,
                           lower::pft::Evaluation &eval, mlir::Location loc,
                           const ConstructQueue &queue,
                           ConstructQueue::const_iterator item);

static void processHostEvalClauses(lower::AbstractConverter &converter,
                                   semantics::SemanticsContext &semaCtx,
                                   lower::StatementContext &stmtCtx,
                                   lower::pft::Evaluation &eval,
                                   mlir::Location loc);

namespace {
/// Structure holding information that is needed to pass host-evaluated
/// information to later lowering stages.
class HostEvalInfo {
public:
  // Allow this function access to private members in order to initialize them.
  friend void ::processHostEvalClauses(lower::AbstractConverter &,
                                       semantics::SemanticsContext &,
                                       lower::StatementContext &,
                                       lower::pft::Evaluation &,
                                       mlir::Location);

  /// Fill \c vars with values stored in \c ops.
  ///
  /// The order in which values are stored matches the one expected by \see
  /// bindOperands().
  void collectValues(llvm::SmallVectorImpl<mlir::Value> &vars) const {
    vars.append(ops.loopLowerBounds);
    vars.append(ops.loopUpperBounds);
    vars.append(ops.loopSteps);

    if (ops.numTeamsLower)
      vars.push_back(ops.numTeamsLower);

    if (ops.numTeamsUpper)
      vars.push_back(ops.numTeamsUpper);

    if (ops.numThreads)
      vars.push_back(ops.numThreads);

    if (ops.threadLimit)
      vars.push_back(ops.threadLimit);
  }

  /// Update \c ops, replacing all values with the corresponding block argument
  /// in \c args.
  ///
  /// The order in which values are stored in \c args is the same as the one
  /// used by \see collectValues().
  void bindOperands(llvm::ArrayRef<mlir::BlockArgument> args) {
    assert(args.size() ==
               ops.loopLowerBounds.size() + ops.loopUpperBounds.size() +
                   ops.loopSteps.size() + (ops.numTeamsLower ? 1 : 0) +
                   (ops.numTeamsUpper ? 1 : 0) + (ops.numThreads ? 1 : 0) +
                   (ops.threadLimit ? 1 : 0) &&
           "invalid block argument list");
    int argIndex = 0;
    for (size_t i = 0; i < ops.loopLowerBounds.size(); ++i)
      ops.loopLowerBounds[i] = args[argIndex++];

    for (size_t i = 0; i < ops.loopUpperBounds.size(); ++i)
      ops.loopUpperBounds[i] = args[argIndex++];

    for (size_t i = 0; i < ops.loopSteps.size(); ++i)
      ops.loopSteps[i] = args[argIndex++];

    if (ops.numTeamsLower)
      ops.numTeamsLower = args[argIndex++];

    if (ops.numTeamsUpper)
      ops.numTeamsUpper = args[argIndex++];

    if (ops.numThreads)
      ops.numThreads = args[argIndex++];

    if (ops.threadLimit)
      ops.threadLimit = args[argIndex++];
  }

  /// Update \p clauseOps and \p ivOut with the corresponding host-evaluated
  /// values and Fortran symbols, respectively, if they have already been
  /// initialized but not yet applied.
  ///
  /// \returns whether an update was performed. If not, these clauses were not
  ///          evaluated in the host device.
  bool apply(mlir::omp::LoopNestOperands &clauseOps,
             llvm::SmallVectorImpl<const semantics::Symbol *> &ivOut) {
    if (iv.empty() || loopNestApplied) {
      loopNestApplied = true;
      return false;
    }

    loopNestApplied = true;
    clauseOps.loopLowerBounds = ops.loopLowerBounds;
    clauseOps.loopUpperBounds = ops.loopUpperBounds;
    clauseOps.loopSteps = ops.loopSteps;
    ivOut.append(iv);
    return true;
  }

  /// Update \p clauseOps with the corresponding host-evaluated values if they
  /// have already been initialized but not yet applied.
  ///
  /// \returns whether an update was performed. If not, these clauses were not
  ///          evaluated in the host device.
  bool apply(mlir::omp::ParallelOperands &clauseOps) {
    if (!ops.numThreads || parallelApplied) {
      parallelApplied = true;
      return false;
    }

    parallelApplied = true;
    clauseOps.numThreads = ops.numThreads;
    return true;
  }

  /// Update \p clauseOps with the corresponding host-evaluated values if they
  /// have already been initialized.
  ///
  /// \returns whether an update was performed. If not, these clauses were not
  ///          evaluated in the host device.
  bool apply(mlir::omp::TeamsOperands &clauseOps) {
    if (!ops.numTeamsLower && !ops.numTeamsUpper && !ops.threadLimit)
      return false;

    clauseOps.numTeamsLower = ops.numTeamsLower;
    clauseOps.numTeamsUpper = ops.numTeamsUpper;
    clauseOps.threadLimit = ops.threadLimit;
    return true;
  }

private:
  mlir::omp::HostEvaluatedOperands ops;
  llvm::SmallVector<const semantics::Symbol *> iv;
  bool loopNestApplied = false, parallelApplied = false;
};
} // namespace

/// Stack of \see HostEvalInfo to represent the current nest of \c omp.target
/// operations being created.
///
/// The current implementation prevents nested 'target' regions from breaking
/// the handling of the outer region by keeping a stack of information
/// structures, but it will probably still require some further work to support
/// reverse offloading.
static llvm::SmallVector<HostEvalInfo, 0> hostEvalInfo;

/// Bind symbols to their corresponding entry block arguments.
///
/// The binding will be performed inside of the current block, which does not
/// necessarily have to be part of the operation for which the binding is done.
/// However, block arguments must be accessible. This enables controlling the
/// insertion point of any new MLIR operations related to the binding of
/// arguments of a loop wrapper operation.
///
/// \param [in] converter - PFT to MLIR conversion interface.
/// \param [in]        op - owner operation of the block arguments to bind.
/// \param [in]      args - entry block arguments information for the given
///                         operation.
static void bindEntryBlockArgs(lower::AbstractConverter &converter,
                               mlir::omp::BlockArgOpenMPOpInterface op,
                               const EntryBlockArgs &args) {
  assert(op != nullptr && "invalid block argument-defining operation");
  assert(args.isValid() && "invalid args");
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  auto bindSingleMapLike = [&converter,
                            &firOpBuilder](const semantics::Symbol &sym,
                                           const mlir::BlockArgument &arg) {
    // Clones the `bounds` placing them inside the entry block and returns
    // them.
    auto cloneBound = [&](mlir::Value bound) {
      if (mlir::isMemoryEffectFree(bound.getDefiningOp())) {
        mlir::Operation *clonedOp = firOpBuilder.clone(*bound.getDefiningOp());
        return clonedOp->getResult(0);
      }
      TODO(converter.getCurrentLocation(),
           "target map-like clause operand unsupported bound type");
    };

    auto cloneBounds = [cloneBound](llvm::ArrayRef<mlir::Value> bounds) {
      llvm::SmallVector<mlir::Value> clonedBounds;
      llvm::transform(bounds, std::back_inserter(clonedBounds),
                      [&](mlir::Value bound) { return cloneBound(bound); });
      return clonedBounds;
    };

    fir::ExtendedValue extVal = converter.getSymbolExtendedValue(sym);
    auto refType = mlir::dyn_cast<fir::ReferenceType>(arg.getType());
    if (refType && fir::isa_builtin_cptr_type(refType.getElementType())) {
      converter.bindSymbol(sym, arg);
    } else {
      extVal.match(
          [&](const fir::BoxValue &v) {
            converter.bindSymbol(sym,
                                 fir::BoxValue(arg, cloneBounds(v.getLBounds()),
                                               v.getExplicitParameters(),
                                               v.getExplicitExtents()));
          },
          [&](const fir::MutableBoxValue &v) {
            converter.bindSymbol(
                sym, fir::MutableBoxValue(arg, cloneBounds(v.getLBounds()),
                                          v.getMutableProperties()));
          },
          [&](const fir::ArrayBoxValue &v) {
            converter.bindSymbol(
                sym, fir::ArrayBoxValue(arg, cloneBounds(v.getExtents()),
                                        cloneBounds(v.getLBounds()),
                                        v.getSourceBox()));
          },
          [&](const fir::CharArrayBoxValue &v) {
            converter.bindSymbol(
                sym, fir::CharArrayBoxValue(arg, cloneBound(v.getLen()),
                                            cloneBounds(v.getExtents()),
                                            cloneBounds(v.getLBounds())));
          },
          [&](const fir::CharBoxValue &v) {
            converter.bindSymbol(
                sym, fir::CharBoxValue(arg, cloneBound(v.getLen())));
          },
          [&](const fir::UnboxedValue &v) { converter.bindSymbol(sym, arg); },
          [&](const auto &) {
            TODO(converter.getCurrentLocation(),
                 "target map clause operand unsupported type");
          });
    }
  };

  auto bindMapLike =
      [&bindSingleMapLike](llvm::ArrayRef<const semantics::Symbol *> syms,
                           llvm::ArrayRef<mlir::BlockArgument> args) {
        // Structure component symbols don't have bindings, and can only be
        // explicitly mapped individually. If a member is captured implicitly
        // we map the entirety of the derived type when we find its symbol.
        llvm::SmallVector<const semantics::Symbol *> processedSyms;
        llvm::copy_if(syms, std::back_inserter(processedSyms),
                      [](auto *sym) { return !sym->owner().IsDerivedType(); });

        for (auto [sym, arg] : llvm::zip_equal(processedSyms, args))
          bindSingleMapLike(*sym, arg);
      };

  auto bindPrivateLike = [&converter, &firOpBuilder](
                             llvm::ArrayRef<const semantics::Symbol *> syms,
                             llvm::ArrayRef<mlir::Value> vars,
                             llvm::ArrayRef<mlir::BlockArgument> args) {
    llvm::SmallVector<const semantics::Symbol *> processedSyms;
    for (auto *sym : syms) {
      if (const auto *commonDet =
              sym->detailsIf<semantics::CommonBlockDetails>()) {
        llvm::transform(commonDet->objects(), std::back_inserter(processedSyms),
                        [&](const auto &mem) { return &*mem; });
      } else {
        processedSyms.push_back(sym);
      }
    }

    for (auto [sym, var, arg] : llvm::zip_equal(processedSyms, vars, args))
      converter.bindSymbol(
          *sym,
          hlfir::translateToExtendedValue(
              var.getLoc(), firOpBuilder, hlfir::Entity{arg},
              /*contiguousHint=*/
              evaluate::IsSimplyContiguous(*sym, converter.getFoldingContext()))
              .first);
  };

  // Process in clause name alphabetical order to match block arguments order.
  // Do not bind host_eval variables because they cannot be used inside of the
  // corresponding region, except for very specific cases handled separately.
  bindPrivateLike(args.inReduction.syms, args.inReduction.vars,
                  op.getInReductionBlockArgs());
  bindMapLike(args.map.syms, op.getMapBlockArgs());
  bindPrivateLike(args.priv.syms, args.priv.vars, op.getPrivateBlockArgs());
  bindPrivateLike(args.reduction.syms, args.reduction.vars,
                  op.getReductionBlockArgs());
  bindPrivateLike(args.taskReduction.syms, args.taskReduction.vars,
                  op.getTaskReductionBlockArgs());
  bindMapLike(args.useDeviceAddr.syms, op.getUseDeviceAddrBlockArgs());
  bindMapLike(args.useDevicePtr.syms, op.getUseDevicePtrBlockArgs());
}

/// Get the list of base values that the specified map-like variables point to.
///
/// This function must be kept in sync with changes to the `createMapInfoOp`
/// utility function, since it must take into account the potential introduction
/// of levels of indirection (i.e. intermediate ops).
///
/// \param [in]     vars - list of values passed to map-like clauses, returned
///                        by an `omp.map.info` operation.
/// \param [out] baseOps - populated with the `var_ptr` values of the
///                        corresponding defining operations.
static void
extractMappedBaseValues(llvm::ArrayRef<mlir::Value> vars,
                        llvm::SmallVectorImpl<mlir::Value> &baseOps) {
  llvm::transform(vars, std::back_inserter(baseOps), [](mlir::Value map) {
    auto mapInfo = map.getDefiningOp<mlir::omp::MapInfoOp>();
    assert(mapInfo && "expected all map vars to be defined by omp.map.info");

    mlir::Value varPtr = mapInfo.getVarPtr();
    if (auto boxAddr = varPtr.getDefiningOp<fir::BoxAddrOp>())
      return boxAddr.getVal();

    return varPtr;
  });
}

/// Get the directive enumeration value corresponding to the given OpenMP
/// construct PFT node.
llvm::omp::Directive
extractOmpDirective(const parser::OpenMPConstruct &ompConstruct) {
  return common::visit(
      common::visitors{
          [](const parser::OpenMPAllocatorsConstruct &c) {
            return llvm::omp::OMPD_allocators;
          },
          [](const parser::OpenMPAtomicConstruct &c) {
            return llvm::omp::OMPD_atomic;
          },
          [](const parser::OpenMPBlockConstruct &c) {
            return std::get<parser::OmpBlockDirective>(
                       std::get<parser::OmpBeginBlockDirective>(c.t).t)
                .v;
          },
          [](const parser::OpenMPCriticalConstruct &c) {
            return llvm::omp::OMPD_critical;
          },
          [](const parser::OpenMPDeclarativeAllocate &c) {
            return llvm::omp::OMPD_allocate;
          },
          [](const parser::OpenMPDispatchConstruct &c) {
            return llvm::omp::OMPD_dispatch;
          },
          [](const parser::OpenMPExecutableAllocate &c) {
            return llvm::omp::OMPD_allocate;
          },
          [](const parser::OpenMPLoopConstruct &c) {
            return std::get<parser::OmpLoopDirective>(
                       std::get<parser::OmpBeginLoopDirective>(c.t).t)
                .v;
          },
          [](const parser::OpenMPSectionConstruct &c) {
            return llvm::omp::OMPD_section;
          },
          [](const parser::OpenMPSectionsConstruct &c) {
            return std::get<parser::OmpSectionsDirective>(
                       std::get<parser::OmpBeginSectionsDirective>(c.t).t)
                .v;
          },
          [](const parser::OpenMPStandaloneConstruct &c) {
            return common::visit(
                common::visitors{
                    [](const parser::OpenMPSimpleStandaloneConstruct &c) {
                      return std::get<parser::OmpSimpleStandaloneDirective>(c.t)
                          .v;
                    },
                    [](const parser::OpenMPFlushConstruct &c) {
                      return llvm::omp::OMPD_flush;
                    },
                    [](const parser::OpenMPCancelConstruct &c) {
                      return llvm::omp::OMPD_cancel;
                    },
                    [](const parser::OpenMPCancellationPointConstruct &c) {
                      return llvm::omp::OMPD_cancellation_point;
                    },
                    [](const parser::OmpMetadirectiveDirective &c) {
                      return llvm::omp::OMPD_metadirective;
                    },
                    [](const parser::OpenMPDepobjConstruct &c) {
                      return llvm::omp::OMPD_depobj;
                    }},
                c.u);
          },
          [](const parser::OpenMPUtilityConstruct &c) {
            return common::visit(
                common::visitors{[](const parser::OmpErrorDirective &c) {
                                   return llvm::omp::OMPD_error;
                                 },
                                 [](const parser::OmpNothingDirective &c) {
                                   return llvm::omp::OMPD_nothing;
                                 }},
                c.u);
          }},
      ompConstruct.u);
}

/// Populate the global \see hostEvalInfo after processing clauses for the given
/// \p eval OpenMP target construct, or nested constructs, if these must be
/// evaluated outside of the target region per the spec.
///
/// In particular, this will ensure that in 'target teams' and equivalent nested
/// constructs, the \c thread_limit and \c num_teams clauses will be evaluated
/// in the host. Additionally, loop bounds, steps and the \c num_threads clause
/// will also be evaluated in the host if a target SPMD construct is detected
/// (i.e. 'target teams distribute parallel do [simd]' or equivalent nesting).
///
/// The result, stored as a global, is intended to be used to populate the \c
/// host_eval operands of the associated \c omp.target operation, and also to be
/// checked and used by later lowering steps to populate the corresponding
/// operands of the \c omp.teams, \c omp.parallel or \c omp.loop_nest
/// operations.
static void processHostEvalClauses(lower::AbstractConverter &converter,
                                   semantics::SemanticsContext &semaCtx,
                                   lower::StatementContext &stmtCtx,
                                   lower::pft::Evaluation &eval,
                                   mlir::Location loc) {
  // Obtain the list of clauses of the given OpenMP block or loop construct
  // evaluation. Other evaluations passed to this lambda keep `clauses`
  // unchanged.
  auto extractClauses = [&semaCtx](lower::pft::Evaluation &eval,
                                   List<Clause> &clauses) {
    const auto *ompEval = eval.getIf<parser::OpenMPConstruct>();
    if (!ompEval)
      return;

    const parser::OmpClauseList *beginClauseList = nullptr;
    const parser::OmpClauseList *endClauseList = nullptr;
    common::visit(
        common::visitors{
            [&](const parser::OpenMPBlockConstruct &ompConstruct) {
              const auto &beginDirective =
                  std::get<parser::OmpBeginBlockDirective>(ompConstruct.t);
              beginClauseList =
                  &std::get<parser::OmpClauseList>(beginDirective.t);
              endClauseList = &std::get<parser::OmpClauseList>(
                  std::get<parser::OmpEndBlockDirective>(ompConstruct.t).t);
            },
            [&](const parser::OpenMPLoopConstruct &ompConstruct) {
              const auto &beginDirective =
                  std::get<parser::OmpBeginLoopDirective>(ompConstruct.t);
              beginClauseList =
                  &std::get<parser::OmpClauseList>(beginDirective.t);

              if (auto &endDirective =
                      std::get<std::optional<parser::OmpEndLoopDirective>>(
                          ompConstruct.t))
                endClauseList =
                    &std::get<parser::OmpClauseList>(endDirective->t);
            },
            [&](const auto &) {}},
        ompEval->u);

    assert(beginClauseList && "expected begin directive");
    clauses.append(makeClauses(*beginClauseList, semaCtx));

    if (endClauseList)
      clauses.append(makeClauses(*endClauseList, semaCtx));
  };

  // Return the directive that is immediately nested inside of the given
  // `parent` evaluation, if it is its only non-end-statement nested evaluation
  // and it represents an OpenMP construct.
  auto extractOnlyOmpNestedDir = [](lower::pft::Evaluation &parent)
      -> std::optional<llvm::omp::Directive> {
    if (!parent.hasNestedEvaluations())
      return std::nullopt;

    llvm::omp::Directive dir;
    auto &nested = parent.getFirstNestedEvaluation();
    if (const auto *ompEval = nested.getIf<parser::OpenMPConstruct>())
      dir = extractOmpDirective(*ompEval);
    else
      return std::nullopt;

    for (auto &sibling : parent.getNestedEvaluations())
      if (&sibling != &nested && !sibling.isEndStmt())
        return std::nullopt;

    return dir;
  };

  // Process the given evaluation assuming it's part of a 'target' construct or
  // captured by one, and store results in the global `hostEvalInfo`.
  std::function<void(lower::pft::Evaluation &, const List<Clause> &)>
      processEval;
  processEval = [&](lower::pft::Evaluation &eval, const List<Clause> &clauses) {
    using namespace llvm::omp;
    ClauseProcessor cp(converter, semaCtx, clauses);

    // Call `processEval` recursively with the immediately nested evaluation and
    // its corresponding clauses if there is a single nested evaluation
    // representing an OpenMP directive that passes the given test.
    auto processSingleNestedIf = [&](llvm::function_ref<bool(Directive)> test) {
      std::optional<Directive> nestedDir = extractOnlyOmpNestedDir(eval);
      if (!nestedDir || !test(*nestedDir))
        return;

      lower::pft::Evaluation &nestedEval = eval.getFirstNestedEvaluation();
      List<lower::omp::Clause> nestedClauses;
      extractClauses(nestedEval, nestedClauses);
      processEval(nestedEval, nestedClauses);
    };

    const auto *ompEval = eval.getIf<parser::OpenMPConstruct>();
    if (!ompEval)
      return;

    HostEvalInfo &hostInfo = hostEvalInfo.back();

    switch (extractOmpDirective(*ompEval)) {
    // Cases where 'teams' and target SPMD clauses might be present.
    case OMPD_teams_distribute_parallel_do:
    case OMPD_teams_distribute_parallel_do_simd:
      cp.processThreadLimit(stmtCtx, hostInfo.ops);
      [[fallthrough]];
    case OMPD_target_teams_distribute_parallel_do:
    case OMPD_target_teams_distribute_parallel_do_simd:
      cp.processNumTeams(stmtCtx, hostInfo.ops);
      [[fallthrough]];
    case OMPD_distribute_parallel_do:
    case OMPD_distribute_parallel_do_simd:
      cp.processCollapse(loc, eval, hostInfo.ops, hostInfo.iv);
      cp.processNumThreads(stmtCtx, hostInfo.ops);
      break;

    // Cases where 'teams' clauses might be present, and target SPMD is
    // possible by looking at nested evaluations.
    case OMPD_teams:
      cp.processThreadLimit(stmtCtx, hostInfo.ops);
      [[fallthrough]];
    case OMPD_target_teams:
      cp.processNumTeams(stmtCtx, hostInfo.ops);
      processSingleNestedIf([](Directive nestedDir) {
        return nestedDir == OMPD_distribute_parallel_do ||
               nestedDir == OMPD_distribute_parallel_do_simd;
      });
      break;

    // Cases where only 'teams' host-evaluated clauses might be present.
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
      cp.processThreadLimit(stmtCtx, hostInfo.ops);
      [[fallthrough]];
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
      cp.processNumTeams(stmtCtx, hostInfo.ops);
      break;

    // Standalone 'target' case.
    case OMPD_target: {
      processSingleNestedIf(
          [](Directive nestedDir) { return topTeamsSet.test(nestedDir); });
      break;
    }
    default:
      break;
    }
  };

  assert(!hostEvalInfo.empty() && "expected HOST_EVAL info structure");

  const auto *ompEval = eval.getIf<parser::OpenMPConstruct>();
  assert(ompEval &&
         llvm::omp::allTargetSet.test(extractOmpDirective(*ompEval)) &&
         "expected TARGET construct evaluation");
  (void)ompEval;

  // Use the whole list of clauses passed to the construct here, rather than the
  // ones only applied to omp.target.
  List<lower::omp::Clause> clauses;
  extractClauses(eval, clauses);
  processEval(eval, clauses);
}

static lower::pft::Evaluation *
getCollapsedLoopEval(lower::pft::Evaluation &eval, int collapseValue) {
  // Return the Evaluation of the innermost collapsed loop, or the current one
  // if there was no COLLAPSE.
  if (collapseValue == 0)
    return &eval;

  lower::pft::Evaluation *curEval = &eval.getFirstNestedEvaluation();
  for (int i = 1; i < collapseValue; i++) {
    // The nested evaluations should be DoConstructs (i.e. they should form
    // a loop nest). Each DoConstruct is a tuple <NonLabelDoStmt, Block,
    // EndDoStmt>.
    assert(curEval->isA<parser::DoConstruct>());
    curEval = &*std::next(curEval->getNestedEvaluations().begin());
  }
  return curEval;
}

static void genNestedEvaluations(lower::AbstractConverter &converter,
                                 lower::pft::Evaluation &eval,
                                 int collapseValue = 0) {
  lower::pft::Evaluation *curEval = getCollapsedLoopEval(eval, collapseValue);

  for (lower::pft::Evaluation &e : curEval->getNestedEvaluations())
    converter.genEval(e);
}

static fir::GlobalOp globalInitialization(lower::AbstractConverter &converter,
                                          fir::FirOpBuilder &firOpBuilder,
                                          const semantics::Symbol &sym,
                                          const lower::pft::Variable &var,
                                          mlir::Location currentLocation) {
  mlir::Type ty = converter.genType(sym);
  std::string globalName = converter.mangleName(sym);
  mlir::StringAttr linkage = firOpBuilder.createInternalLinkage();
  fir::GlobalOp global =
      firOpBuilder.createGlobal(currentLocation, ty, globalName, linkage);

  // Create default initialization for non-character scalar.
  if (semantics::IsAllocatableOrObjectPointer(&sym)) {
    mlir::Type baseAddrType = mlir::dyn_cast<fir::BoxType>(ty).getEleTy();
    lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value nullAddr =
              b.createNullConstant(currentLocation, baseAddrType);
          mlir::Value box =
              b.create<fir::EmboxOp>(currentLocation, ty, nullAddr);
          b.create<fir::HasValueOp>(currentLocation, box);
        });
  } else {
    lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value undef = b.create<fir::UndefOp>(currentLocation, ty);
          b.create<fir::HasValueOp>(currentLocation, undef);
        });
  }

  return global;
}

// Get the extended value for \p val by extracting additional variable
// information from \p base.
static fir::ExtendedValue getExtendedValue(fir::ExtendedValue base,
                                           mlir::Value val) {
  return base.match(
      [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
        return fir::MutableBoxValue(val, box.nonDeferredLenParams(), {});
      },
      [&](const auto &) -> fir::ExtendedValue {
        return fir::substBase(base, val);
      });
}

#ifndef NDEBUG
static bool isThreadPrivate(lower::SymbolRef sym) {
  if (const auto *details = sym->detailsIf<semantics::CommonBlockDetails>()) {
    for (const auto &obj : details->objects())
      if (!obj->test(semantics::Symbol::Flag::OmpThreadprivate))
        return false;
    return true;
  }
  return sym->test(semantics::Symbol::Flag::OmpThreadprivate);
}
#endif

static void threadPrivatizeVars(lower::AbstractConverter &converter,
                                lower::pft::Evaluation &eval) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  // If the symbol corresponds to the original ThreadprivateOp, use the symbol
  // value from that operation to create one ThreadprivateOp copy operation
  // inside the parallel region.
  // In some cases, however, the symbol will correspond to the original,
  // non-threadprivate variable. This can happen, for instance, with a common
  // block, declared in a separate module, used by a parent procedure and
  // privatized in its child procedure.
  auto genThreadprivateOp = [&](lower::SymbolRef sym) -> mlir::Value {
    assert(isThreadPrivate(sym));
    mlir::Value symValue = converter.getSymbolAddress(sym);
    mlir::Operation *op = symValue.getDefiningOp();
    if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(op))
      op = declOp.getMemref().getDefiningOp();
    if (mlir::isa<mlir::omp::ThreadprivateOp>(op))
      symValue = mlir::dyn_cast<mlir::omp::ThreadprivateOp>(op).getSymAddr();
    return firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  };

  llvm::SetVector<const semantics::Symbol *> threadprivateSyms;
  converter.collectSymbolSet(eval, threadprivateSyms,
                             semantics::Symbol::Flag::OmpThreadprivate,
                             /*collectSymbols=*/true,
                             /*collectHostAssociatedSymbols=*/true);
  std::set<semantics::SourceName> threadprivateSymNames;

  // For a COMMON block, the ThreadprivateOp is generated for itself instead of
  // its members, so only bind the value of the new copied ThreadprivateOp
  // inside the parallel region to the common block symbol only once for
  // multiple members in one COMMON block.
  llvm::SetVector<const semantics::Symbol *> commonSyms;
  for (std::size_t i = 0; i < threadprivateSyms.size(); i++) {
    const semantics::Symbol *sym = threadprivateSyms[i];
    mlir::Value symThreadprivateValue;
    // The variable may be used more than once, and each reference has one
    // symbol with the same name. Only do once for references of one variable.
    if (threadprivateSymNames.find(sym->name()) != threadprivateSymNames.end())
      continue;
    threadprivateSymNames.insert(sym->name());
    if (const semantics::Symbol *common =
            semantics::FindCommonBlockContaining(sym->GetUltimate())) {
      mlir::Value commonThreadprivateValue;
      if (commonSyms.contains(common)) {
        commonThreadprivateValue = converter.getSymbolAddress(*common);
      } else {
        commonThreadprivateValue = genThreadprivateOp(*common);
        converter.bindSymbol(*common, commonThreadprivateValue);
        commonSyms.insert(common);
      }
      symThreadprivateValue = lower::genCommonBlockMember(
          converter, currentLocation, *sym, commonThreadprivateValue);
    } else {
      symThreadprivateValue = genThreadprivateOp(*sym);
    }

    fir::ExtendedValue sexv = converter.getSymbolExtendedValue(*sym);
    fir::ExtendedValue symThreadprivateExv =
        getExtendedValue(sexv, symThreadprivateValue);
    converter.bindSymbol(*sym, symThreadprivateExv);
  }
}

static mlir::Operation *
createAndSetPrivatizedLoopVar(lower::AbstractConverter &converter,
                              mlir::Location loc, mlir::Value indexVal,
                              const semantics::Symbol *sym) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  mlir::Type tempTy = converter.genType(*sym);

  assert(converter.isPresentShallowLookup(*sym) &&
         "Expected symbol to be in symbol table.");

  firOpBuilder.restoreInsertionPoint(insPt);
  mlir::Value cvtVal = firOpBuilder.createConvert(loc, tempTy, indexVal);
  mlir::Operation *storeOp = firOpBuilder.create<fir::StoreOp>(
      loc, cvtVal, converter.getSymbolAddress(*sym));
  return storeOp;
}

// This helper function implements the functionality of "promoting" non-CPTR
// arguments of use_device_ptr to use_device_addr arguments (automagic
// conversion of use_device_ptr -> use_device_addr in these cases). The way we
// do so currently is through the shuffling of operands from the
// devicePtrOperands to deviceAddrOperands, as well as the types, locations and
// symbols.
//
// This effectively implements some deprecated OpenMP functionality that some
// legacy applications unfortunately depend on (deprecated in specification
// version 5.2):
//
// "If a list item in a use_device_ptr clause is not of type C_PTR, the behavior
//  is as if the list item appeared in a use_device_addr clause. Support for
//  such list items in a use_device_ptr clause is deprecated."
static void promoteNonCPtrUseDevicePtrArgsToUseDeviceAddr(
    llvm::SmallVectorImpl<mlir::Value> &useDeviceAddrVars,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceAddrSyms,
    llvm::SmallVectorImpl<mlir::Value> &useDevicePtrVars,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDevicePtrSyms) {
  // Iterate over our use_device_ptr list and shift all non-cptr arguments into
  // use_device_addr.
  auto *varIt = useDevicePtrVars.begin();
  auto *symIt = useDevicePtrSyms.begin();
  while (varIt != useDevicePtrVars.end()) {
    if (fir::isa_builtin_cptr_type(fir::unwrapRefType(varIt->getType()))) {
      ++varIt;
      ++symIt;
      continue;
    }

    useDeviceAddrVars.push_back(*varIt);
    useDeviceAddrSyms.push_back(*symIt);

    varIt = useDevicePtrVars.erase(varIt);
    symIt = useDevicePtrSyms.erase(symIt);
  }
}

/// Extract the list of function and variable symbols affected by the given
/// 'declare target' directive and return the intended device type for them.
static void getDeclareTargetInfo(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct,
    mlir::omp::DeclareTargetOperands &clauseOps,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {
  const auto &spec =
      std::get<parser::OmpDeclareTargetSpecifier>(declareTargetConstruct.t);
  if (const auto *objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
    ObjectList objects{makeObjects(*objectList, semaCtx)};
    // Case: declare target(func, var1, var2)
    gatherFuncAndVarSyms(objects, mlir::omp::DeclareTargetCaptureClause::to,
                         symbolAndClause);
  } else if (const auto *clauseList{
                 parser::Unwrap<parser::OmpClauseList>(spec.u)}) {
    List<Clause> clauses = makeClauses(*clauseList, semaCtx);
    if (clauses.empty()) {
      Fortran::lower::pft::FunctionLikeUnit *owningProc =
          eval.getOwningProcedure();
      if (owningProc && (!owningProc->isMainProgram() ||
                         owningProc->getMainProgramSymbol())) {
        // Case: declare target, implicit capture of function
        symbolAndClause.emplace_back(mlir::omp::DeclareTargetCaptureClause::to,
                                     owningProc->getSubprogramSymbol());
      }
    }

    ClauseProcessor cp(converter, semaCtx, clauses);
    cp.processDeviceType(clauseOps);
    cp.processEnter(symbolAndClause);
    cp.processLink(symbolAndClause);
    cp.processTo(symbolAndClause);

    cp.processTODO<clause::Indirect>(converter.getCurrentLocation(),
                                     llvm::omp::Directive::OMPD_declare_target);
  }
}

static void collectDeferredDeclareTargets(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct,
    llvm::SmallVectorImpl<lower::OMPDeferredDeclareTargetInfo>
        &deferredDeclareTarget) {
  mlir::omp::DeclareTargetOperands clauseOps;
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
  getDeclareTargetInfo(converter, semaCtx, eval, declareTargetConstruct,
                       clauseOps, symbolAndClause);
  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();

  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(
        converter.mangleName(std::get<const semantics::Symbol &>(symClause)));

    if (!op) {
      deferredDeclareTarget.push_back({std::get<0>(symClause),
                                       clauseOps.deviceType,
                                       std::get<1>(symClause)});
    }
  }
}

static std::optional<mlir::omp::DeclareTargetDeviceType>
getDeclareTargetFunctionDevice(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct) {
  mlir::omp::DeclareTargetOperands clauseOps;
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
  getDeclareTargetInfo(converter, semaCtx, eval, declareTargetConstruct,
                       clauseOps, symbolAndClause);

  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(
        converter.mangleName(std::get<const semantics::Symbol &>(symClause)));

    if (mlir::isa_and_nonnull<mlir::func::FuncOp>(op))
      return clauseOps.deviceType;
  }

  return std::nullopt;
}

/// Set up the entry block of the given `omp.loop_nest` operation, adding a
/// block argument for each loop induction variable and allocating and
/// initializing a private value to hold each of them.
///
/// This function can also bind the symbols of any variables that should match
/// block arguments on parent loop wrapper operations attached to the same
/// loop. This allows the introduction of any necessary `hlfir.declare`
/// operations inside of the entry block of the `omp.loop_nest` operation and
/// not directly under any of the wrappers, which would invalidate them.
///
/// \param [in]          op - the loop nest operation.
/// \param [in]   converter - PFT to MLIR conversion interface.
/// \param [in]         loc - location.
/// \param [in]        args - symbols of induction variables.
/// \param [in] wrapperArgs - list of parent loop wrappers and their associated
///                           entry block arguments.
static void genLoopVars(
    mlir::Operation *op, lower::AbstractConverter &converter,
    mlir::Location &loc, llvm::ArrayRef<const semantics::Symbol *> args,
    llvm::ArrayRef<
        std::pair<mlir::omp::BlockArgOpenMPOpInterface, const EntryBlockArgs &>>
        wrapperArgs = {}) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto &region = op->getRegion(0);

  std::size_t loopVarTypeSize = 0;
  for (const semantics::Symbol *arg : args)
    loopVarTypeSize = std::max(loopVarTypeSize, arg->GetUltimate().size());
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  llvm::SmallVector<mlir::Type> tiv(args.size(), loopVarType);
  llvm::SmallVector<mlir::Location> locs(args.size(), loc);
  firOpBuilder.createBlock(&region, {}, tiv, locs);

  // Update nested wrapper operands if parent wrappers have mapped these values
  // to block arguments.
  //
  // Binding these values earlier would take care of this, but we cannot rely on
  // that approach because binding in between the creation of a wrapper and the
  // next one would result in 'hlfir.declare' operations being introduced inside
  // of a wrapper, which is illegal.
  mlir::IRMapping mapper;
  for (auto [argGeneratingOp, blockArgs] : wrapperArgs) {
    for (mlir::OpOperand &operand : argGeneratingOp->getOpOperands())
      operand.set(mapper.lookupOrDefault(operand.get()));

    for (const auto [arg, var] : llvm::zip_equal(
             argGeneratingOp->getRegion(0).getArguments(), blockArgs.getVars()))
      mapper.map(var, arg);
  }

  // Bind the entry block arguments of parent wrappers to the corresponding
  // symbols.
  for (auto [argGeneratingOp, blockArgs] : wrapperArgs)
    bindEntryBlockArgs(converter, argGeneratingOp, blockArgs);

  // The argument is not currently in memory, so make a temporary for the
  // argument, and store it there, then bind that location to the argument.
  mlir::Operation *storeOp = nullptr;
  for (auto [argIndex, argSymbol] : llvm::enumerate(args)) {
    mlir::Value indexVal = fir::getBase(region.front().getArgument(argIndex));
    storeOp =
        createAndSetPrivatizedLoopVar(converter, loc, indexVal, argSymbol);
  }
  firOpBuilder.setInsertionPointAfter(storeOp);
}

static void
markDeclareTarget(mlir::Operation *op, lower::AbstractConverter &converter,
                  mlir::omp::DeclareTargetCaptureClause captureClause,
                  mlir::omp::DeclareTargetDeviceType deviceType) {
  // TODO: Add support for program local variables with declare target applied
  auto declareTargetOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(op);
  if (!declareTargetOp)
    fir::emitFatalError(
        converter.getCurrentLocation(),
        "Attempt to apply declare target on unsupported operation");

  // The function or global already has a declare target applied to it, very
  // likely through implicit capture (usage in another declare target
  // function/subroutine). It should be marked as any if it has been assigned
  // both host and nohost, else we skip, as there is no change
  if (declareTargetOp.isDeclareTarget()) {
    if (declareTargetOp.getDeclareTargetDeviceType() != deviceType)
      declareTargetOp.setDeclareTarget(mlir::omp::DeclareTargetDeviceType::any,
                                       captureClause);
    return;
  }

  declareTargetOp.setDeclareTarget(deviceType, captureClause);
}

//===----------------------------------------------------------------------===//
// Op body generation helper structures and functions
//===----------------------------------------------------------------------===//

struct OpWithBodyGenInfo {
  /// A type for a code-gen callback function. This takes as argument the op for
  /// which the code is being generated and returns the arguments of the op's
  /// region.
  using GenOMPRegionEntryCBFn =
      std::function<llvm::SmallVector<const semantics::Symbol *>(
          mlir::Operation *)>;

  OpWithBodyGenInfo(lower::AbstractConverter &converter,
                    lower::SymMap &symTable,
                    semantics::SemanticsContext &semaCtx, mlir::Location loc,
                    lower::pft::Evaluation &eval, llvm::omp::Directive dir)
      : converter(converter), symTable(symTable), semaCtx(semaCtx), loc(loc),
        eval(eval), dir(dir) {}

  OpWithBodyGenInfo &setClauses(const List<Clause> *value) {
    clauses = value;
    return *this;
  }

  OpWithBodyGenInfo &setDataSharingProcessor(DataSharingProcessor *value) {
    dsp = value;
    return *this;
  }

  OpWithBodyGenInfo &setGenRegionEntryCb(GenOMPRegionEntryCBFn value) {
    genRegionEntryCB = value;
    return *this;
  }

  OpWithBodyGenInfo &setGenSkeletonOnly(bool value) {
    genSkeletonOnly = value;
    return *this;
  }

  /// [inout] converter to use for the clauses.
  lower::AbstractConverter &converter;
  /// [in] Symbol table
  lower::SymMap &symTable;
  /// [in] Semantics context
  semantics::SemanticsContext &semaCtx;
  /// [in] location in source code.
  mlir::Location loc;
  /// [in] current PFT node/evaluation.
  lower::pft::Evaluation &eval;
  /// [in] leaf directive for which to generate the op body.
  llvm::omp::Directive dir;
  /// [in] list of clauses to process.
  const List<Clause> *clauses = nullptr;
  /// [in] if provided, processes the construct's data-sharing attributes.
  DataSharingProcessor *dsp = nullptr;
  /// [in] if provided, emits the op's region entry. Otherwise, an emtpy block
  /// is created in the region.
  GenOMPRegionEntryCBFn genRegionEntryCB = nullptr;
  /// [in] if set to `true`, skip generating nested evaluations and dispatching
  /// any further leaf constructs.
  bool genSkeletonOnly = false;
};

/// Create the body (block) for an OpenMP Operation.
///
/// \param [in]   op  - the operation the body belongs to.
/// \param [in] info  - options controlling code-gen for the construction.
/// \param [in] queue - work queue with nested constructs.
/// \param [in] item  - item in the queue to generate body for.
static void createBodyOfOp(mlir::Operation &op, const OpWithBodyGenInfo &info,
                           const ConstructQueue &queue,
                           ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = info.converter.getFirOpBuilder();

  auto insertMarker = [](fir::FirOpBuilder &builder) {
    mlir::Value undef = builder.create<fir::UndefOp>(builder.getUnknownLoc(),
                                                     builder.getIndexType());
    return undef.getDefiningOp();
  };

  // If an argument for the region is provided then create the block with that
  // argument. Also update the symbol's address with the mlir argument value.
  // e.g. For loops the argument is the induction variable. And all further
  // uses of the induction variable should use this mlir value.
  auto regionArgs = [&]() -> llvm::SmallVector<const semantics::Symbol *> {
    if (info.genRegionEntryCB != nullptr) {
      return info.genRegionEntryCB(&op);
    }

    firOpBuilder.createBlock(&op.getRegion(0));
    return {};
  }();
  // Mark the earliest insertion point.
  mlir::Operation *marker = insertMarker(firOpBuilder);

  // If it is an unstructured region, create empty blocks for all evaluations.
  if (lower::omp::isLastItemInQueue(item, queue) &&
      info.eval.lowerAsUnstructured()) {
    lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp, mlir::omp::YieldOp>(
        firOpBuilder, info.eval.getNestedEvaluations());
  }

  // Start with privatization, so that the lowering of the nested
  // code will use the right symbols.
  bool isLoop = llvm::omp::getDirectiveAssociation(info.dir) ==
                llvm::omp::Association::Loop;
  bool privatize = info.clauses;

  firOpBuilder.setInsertionPoint(marker);
  std::optional<DataSharingProcessor> tempDsp;
  if (privatize && !info.dsp) {
    tempDsp.emplace(info.converter, info.semaCtx, *info.clauses, info.eval,
                    Fortran::lower::omp::isLastItemInQueue(item, queue),
                    /*useDelayedPrivatization=*/false, info.symTable);
    tempDsp->processStep1();
  }

  if (info.dir == llvm::omp::Directive::OMPD_parallel) {
    threadPrivatizeVars(info.converter, info.eval);
    if (info.clauses) {
      firOpBuilder.setInsertionPoint(marker);
      ClauseProcessor(info.converter, info.semaCtx, *info.clauses)
          .processCopyin();
    }
  }

  if (!info.genSkeletonOnly) {
    if (ConstructQueue::const_iterator next = std::next(item);
        next != queue.end()) {
      genOMPDispatch(info.converter, info.symTable, info.semaCtx, info.eval,
                     info.loc, queue, next);
    } else {
      // genFIR(Evaluation&) tries to patch up unterminated blocks, causing
      // a lot of complications for our approach if the terminator generation
      // is delayed past this point. Insert a temporary terminator here, then
      // delete it.
      firOpBuilder.setInsertionPointToEnd(&op.getRegion(0).back());
      auto *temp = lower::genOpenMPTerminator(firOpBuilder, &op, info.loc);
      firOpBuilder.setInsertionPointAfter(marker);
      genNestedEvaluations(info.converter, info.eval);
      temp->erase();
    }
  }

  // Get or create a unique exiting block from the given region, or
  // return nullptr if there is no exiting block.
  auto getUniqueExit = [&](mlir::Region &region) -> mlir::Block * {
    // Find the blocks where the OMP terminator should go. In simple cases
    // it is the single block in the operation's region. When the region
    // is more complicated, especially with unstructured control flow, there
    // may be multiple blocks, and some of them may have non-OMP terminators
    // resulting from lowering of the code contained within the operation.
    // All the remaining blocks are potential exit points from the op's region.
    //
    // Explicit control flow cannot exit any OpenMP region (other than via
    // STOP), and that is enforced by semantic checks prior to lowering. STOP
    // statements are lowered to a function call.

    // Collect unterminated blocks.
    llvm::SmallVector<mlir::Block *> exits;
    for (mlir::Block &b : region) {
      if (b.empty() || !b.back().hasTrait<mlir::OpTrait::IsTerminator>())
        exits.push_back(&b);
    }

    if (exits.empty())
      return nullptr;
    // If there already is a unique exiting block, do not create another one.
    // Additionally, some ops (e.g. omp.sections) require only 1 block in
    // its region.
    if (exits.size() == 1)
      return exits[0];
    mlir::Block *exit = firOpBuilder.createBlock(&region);
    for (mlir::Block *b : exits) {
      firOpBuilder.setInsertionPointToEnd(b);
      firOpBuilder.create<mlir::cf::BranchOp>(info.loc, exit);
    }
    return exit;
  };

  if (auto *exitBlock = getUniqueExit(op.getRegion(0))) {
    firOpBuilder.setInsertionPointToEnd(exitBlock);
    auto *term = lower::genOpenMPTerminator(firOpBuilder, &op, info.loc);
    // Only insert lastprivate code when there actually is an exit block.
    // Such a block may not exist if the nested code produced an infinite
    // loop (this may not make sense in production code, but a user could
    // write that and we should handle it).
    firOpBuilder.setInsertionPoint(term);
    if (privatize) {
      // DataSharingProcessor::processStep2() may create operations before/after
      // the one passed as argument. We need to treat loop wrappers and their
      // nested loop as a unit, so we need to pass the top level wrapper (if
      // present). Otherwise, these operations will be inserted within a
      // wrapper region.
      mlir::Operation *privatizationTopLevelOp = &op;
      if (auto loopNest = llvm::dyn_cast<mlir::omp::LoopNestOp>(op)) {
        llvm::SmallVector<mlir::omp::LoopWrapperInterface> wrappers;
        loopNest.gatherWrappers(wrappers);
        if (!wrappers.empty())
          privatizationTopLevelOp = &*wrappers.back();
      }

      if (!info.dsp) {
        assert(tempDsp.has_value());
        tempDsp->processStep2(privatizationTopLevelOp, isLoop);
      } else {
        if (isLoop && regionArgs.size() > 0) {
          for (const auto &regionArg : regionArgs) {
            info.dsp->pushLoopIV(info.converter.getSymbolAddress(*regionArg));
          }
        }
        info.dsp->processStep2(privatizationTopLevelOp, isLoop);
      }
    }
  }

  firOpBuilder.setInsertionPointAfter(marker);
  marker->erase();
}

static void genBodyOfTargetDataOp(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::omp::TargetDataOp &dataOp, const EntryBlockArgs &args,
    const mlir::Location &currentLocation, const ConstructQueue &queue,
    ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  genEntryBlock(firOpBuilder, args, dataOp.getRegion());
  bindEntryBlockArgs(converter, dataOp, args);

  // Insert dummy instruction to remember the insertion position. The
  // marker will be deleted by clean up passes since there are no uses.
  // Remembering the position for further insertion is important since
  // there are hlfir.declares inserted above while setting block arguments
  // and new code from the body should be inserted after that.
  mlir::Value undefMarker = firOpBuilder.create<fir::UndefOp>(
      dataOp.getLoc(), firOpBuilder.getIndexType());

  // Create blocks for unstructured regions. This has to be done since
  // blocks are initially allocated with the function as the parent region.
  if (eval.lowerAsUnstructured()) {
    lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp, mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);

  // Set the insertion point after the marker.
  firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());

  if (ConstructQueue::const_iterator next = std::next(item);
      next != queue.end()) {
    genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                   next);
  } else {
    genNestedEvaluations(converter, eval);
  }
}

// This generates intermediate common block member accesses within a region
// and then rebinds the members symbol to the intermediate accessors we have
// generated so that subsequent code generation will utilise these instead.
//
// When the scope changes, the bindings to the intermediate accessors should
// be dropped in place of the original symbol bindings.
//
// This is for utilisation with TargetOp.
static void genIntermediateCommonBlockAccessors(
    Fortran::lower::AbstractConverter &converter,
    const mlir::Location &currentLocation,
    llvm::ArrayRef<mlir::BlockArgument> mapBlockArgs,
    llvm::ArrayRef<const Fortran::semantics::Symbol *> mapSyms) {
  // Iterate over the symbol list, which will be shorter than the list of
  // arguments if new entry block arguments were introduced to implicitly map
  // outside values used by the bounds cloned into the target region. In that
  // case, the additional block arguments do not need processing here.
  for (auto [mapSym, mapArg] : llvm::zip_first(mapSyms, mapBlockArgs)) {
    auto *details = mapSym->detailsIf<Fortran::semantics::CommonBlockDetails>();
    if (!details)
      continue;

    for (auto obj : details->objects()) {
      auto targetCBMemberBind = Fortran::lower::genCommonBlockMember(
          converter, currentLocation, *obj, mapArg);
      fir::ExtendedValue sexv = converter.getSymbolExtendedValue(*obj);
      fir::ExtendedValue targetCBExv =
          getExtendedValue(sexv, targetCBMemberBind);
      converter.bindSymbol(*obj, targetCBExv);
    }
  }
}

// This functions creates a block for the body of the targetOp's region. It adds
// all the symbols present in mapSymbols as block arguments to this block.
static void genBodyOfTargetOp(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::omp::TargetOp &targetOp, const EntryBlockArgs &args,
    const mlir::Location &currentLocation, const ConstructQueue &queue,
    ConstructQueue::const_iterator item, DataSharingProcessor &dsp) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);

  mlir::Region &region = targetOp.getRegion();
  mlir::Block *entryBlock = genEntryBlock(firOpBuilder, args, region);
  bindEntryBlockArgs(converter, targetOp, args);
  if (!hostEvalInfo.empty())
    hostEvalInfo.back().bindOperands(argIface.getHostEvalBlockArgs());

  // Check if cloning the bounds introduced any dependency on the outer region.
  // If so, then either clone them as well if they are MemoryEffectFree, or else
  // copy them to a new temporary and add them to the map and block_argument
  // lists and replace their uses with the new temporary.
  llvm::SetVector<mlir::Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  while (!valuesDefinedAbove.empty()) {
    for (mlir::Value val : valuesDefinedAbove) {
      mlir::Operation *valOp = val.getDefiningOp();
      assert(valOp != nullptr);

      // NOTE: We skip BoxDimsOp's as the lesser of two evils is to map the
      // indices separately, as the alternative is to eventually map the Box,
      // which comes with a fairly large overhead comparatively. We could be
      // more robust about this and check using a BackwardsSlice to see if we
      // run the risk of mapping a box.
      if (mlir::isMemoryEffectFree(valOp) &&
          !mlir::isa<fir::BoxDimsOp>(valOp)) {
        mlir::Operation *clonedOp = valOp->clone();
        entryBlock->push_front(clonedOp);

        auto replace = [entryBlock](mlir::OpOperand &use) {
          return use.getOwner()->getBlock() == entryBlock;
        };

        valOp->getResults().replaceUsesWithIf(clonedOp->getResults(), replace);
        valOp->replaceUsesWithIf(clonedOp, replace);
      } else {
        auto savedIP = firOpBuilder.getInsertionPoint();
        firOpBuilder.setInsertionPointAfter(valOp);
        auto copyVal =
            firOpBuilder.createTemporary(val.getLoc(), val.getType());
        firOpBuilder.createStoreWithConvert(copyVal.getLoc(), val, copyVal);

        fir::factory::AddrAndBoundsInfo info =
            fir::factory::getDataOperandBaseAddr(
                firOpBuilder, val, /*isOptional=*/false, val.getLoc());
        llvm::SmallVector<mlir::Value> bounds =
            fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
                                               mlir::omp::MapBoundsType>(
                firOpBuilder, info,
                hlfir::translateToExtendedValue(val.getLoc(), firOpBuilder,
                                                hlfir::Entity{val})
                    .first,
                /*dataExvIsAssumedSize=*/false, val.getLoc());

        std::stringstream name;
        firOpBuilder.setInsertionPoint(targetOp);

        llvm::omp::OpenMPOffloadMappingFlags mapFlag =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
        mlir::omp::VariableCaptureKind captureKind =
            mlir::omp::VariableCaptureKind::ByRef;

        mlir::Type eleType = copyVal.getType();
        if (auto refType =
                mlir::dyn_cast<fir::ReferenceType>(copyVal.getType()))
          eleType = refType.getElementType();

        if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
          captureKind = mlir::omp::VariableCaptureKind::ByCopy;
        } else if (!fir::isa_builtin_cptr_type(eleType)) {
          mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
        }

        mlir::Value mapOp = createMapInfoOp(
            firOpBuilder, copyVal.getLoc(), copyVal,
            /*varPtrPtr=*/mlir::Value{}, name.str(), bounds,
            /*members=*/llvm::SmallVector<mlir::Value>{},
            /*membersIndex=*/mlir::ArrayAttr{},
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                mapFlag),
            captureKind, copyVal.getType());

        // Get the index of the first non-map argument before modifying mapVars,
        // then append an element to mapVars and an associated entry block
        // argument at that index.
        unsigned insertIndex =
            argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs();
        targetOp.getMapVarsMutable().append(mapOp);
        mlir::Value clonedValArg = region.insertArgument(
            insertIndex, copyVal.getType(), copyVal.getLoc());

        firOpBuilder.setInsertionPointToStart(entryBlock);
        auto loadOp = firOpBuilder.create<fir::LoadOp>(clonedValArg.getLoc(),
                                                       clonedValArg);
        val.replaceUsesWithIf(loadOp->getResult(0),
                              [entryBlock](mlir::OpOperand &use) {
                                return use.getOwner()->getBlock() == entryBlock;
                              });
        firOpBuilder.setInsertionPoint(entryBlock, savedIP);
      }
    }
    valuesDefinedAbove.clear();
    mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  }

  // Insert dummy instruction to remember the insertion position. The
  // marker will be deleted since there are not uses.
  // In the HLFIR flow there are hlfir.declares inserted above while
  // setting block arguments.
  mlir::Value undefMarker = firOpBuilder.create<fir::UndefOp>(
      targetOp.getLoc(), firOpBuilder.getIndexType());

  // Create blocks for unstructured regions. This has to be done since
  // blocks are initially allocated with the function as the parent region.
  if (lower::omp::isLastItemInQueue(item, queue) &&
      eval.lowerAsUnstructured()) {
    lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp, mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);

  // Create the insertion point after the marker.
  firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());

  // If we map a common block using it's symbol e.g. map(tofrom: /common_block/)
  // and accessing its members within the target region, there is a large
  // chance we will end up with uses external to the region accessing the common
  // resolve these, we do so by generating new common block member accesses
  // within the region, binding them to the member symbol for the scope of the
  // region so that subsequent code generation within the region will utilise
  // our new member accesses we have created.
  genIntermediateCommonBlockAccessors(
      converter, currentLocation, argIface.getMapBlockArgs(), args.map.syms);

  if (ConstructQueue::const_iterator next = std::next(item);
      next != queue.end()) {
    genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                   next);
  } else {
    genNestedEvaluations(converter, eval);
  }

  dsp.processStep2(targetOp, /*isLoop=*/false);
}

template <typename OpTy, typename... Args>
static OpTy genOpWithBody(const OpWithBodyGenInfo &info,
                          const ConstructQueue &queue,
                          ConstructQueue::const_iterator item, Args &&...args) {
  auto op = info.converter.getFirOpBuilder().create<OpTy>(
      info.loc, std::forward<Args>(args)...);
  createBodyOfOp(*op, info, queue, item);
  return op;
}

template <typename OpTy, typename ClauseOpsTy>
static OpTy genWrapperOp(lower::AbstractConverter &converter,
                         mlir::Location loc, const ClauseOpsTy &clauseOps,
                         const EntryBlockArgs &args) {
  static_assert(
      OpTy::template hasTrait<mlir::omp::LoopWrapperInterface::Trait>(),
      "expected a loop wrapper");
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Create wrapper.
  auto op = firOpBuilder.create<OpTy>(loc, clauseOps);

  // Create entry block with arguments.
  genEntryBlock(firOpBuilder, args, op.getRegion());

  return op;
}

//===----------------------------------------------------------------------===//
// Code generation functions for clauses
//===----------------------------------------------------------------------===//

static void genCriticalDeclareClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    const List<Clause> &clauses, mlir::Location loc,
    mlir::omp::CriticalDeclareOperands &clauseOps, llvm::StringRef name) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processHint(clauseOps);
  clauseOps.symName =
      mlir::StringAttr::get(converter.getFirOpBuilder().getContext(), name);
}

static void genDistributeClauses(lower::AbstractConverter &converter,
                                 semantics::SemanticsContext &semaCtx,
                                 lower::StatementContext &stmtCtx,
                                 const List<Clause> &clauses,
                                 mlir::Location loc,
                                 mlir::omp::DistributeOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processDistSchedule(stmtCtx, clauseOps);
  cp.processOrder(clauseOps);
}

static void genFlushClauses(lower::AbstractConverter &converter,
                            semantics::SemanticsContext &semaCtx,
                            const ObjectList &objects,
                            const List<Clause> &clauses, mlir::Location loc,
                            llvm::SmallVectorImpl<mlir::Value> &operandRange) {
  if (!objects.empty())
    genObjectList(objects, converter, operandRange);

  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processTODO<clause::AcqRel, clause::Acquire, clause::Release,
                 clause::SeqCst>(loc, llvm::omp::OMPD_flush);
}

static void
genLoopNestClauses(lower::AbstractConverter &converter,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval, const List<Clause> &clauses,
                   mlir::Location loc, mlir::omp::LoopNestOperands &clauseOps,
                   llvm::SmallVectorImpl<const semantics::Symbol *> &iv) {
  ClauseProcessor cp(converter, semaCtx, clauses);

  if (hostEvalInfo.empty() || !hostEvalInfo.back().apply(clauseOps, iv))
    cp.processCollapse(loc, eval, clauseOps, iv);

  clauseOps.loopInclusive = converter.getFirOpBuilder().getUnitAttr();
}

static void genLoopClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    const List<Clause> &clauses, mlir::Location loc,
    mlir::omp::LoopOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processBind(clauseOps);
  cp.processOrder(clauseOps);
  cp.processReduction(loc, clauseOps, reductionSyms);
  cp.processTODO<clause::Lastprivate>(loc, llvm::omp::Directive::OMPD_loop);
}

static void genMaskedClauses(lower::AbstractConverter &converter,
                             semantics::SemanticsContext &semaCtx,
                             lower::StatementContext &stmtCtx,
                             const List<Clause> &clauses, mlir::Location loc,
                             mlir::omp::MaskedOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processFilter(stmtCtx, clauseOps);
}

static void
genOrderedRegionClauses(lower::AbstractConverter &converter,
                        semantics::SemanticsContext &semaCtx,
                        const List<Clause> &clauses, mlir::Location loc,
                        mlir::omp::OrderedRegionOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processTODO<clause::Simd>(loc, llvm::omp::Directive::OMPD_ordered);
}

static void genParallelClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::ParallelOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_parallel, clauseOps);

  if (hostEvalInfo.empty() || !hostEvalInfo.back().apply(clauseOps))
    cp.processNumThreads(stmtCtx, clauseOps);

  cp.processProcBind(clauseOps);
  cp.processReduction(loc, clauseOps, reductionSyms);
}

static void genScanClauses(lower::AbstractConverter &converter,
                           semantics::SemanticsContext &semaCtx,
                           const List<Clause> &clauses, mlir::Location loc,
                           mlir::omp::ScanOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processInclusive(loc, clauseOps);
  cp.processExclusive(loc, clauseOps);
}

static void genSectionsClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    const List<Clause> &clauses, mlir::Location loc,
    mlir::omp::SectionsOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processNowait(clauseOps);
  cp.processReduction(loc, clauseOps, reductionSyms);
  // TODO Support delayed privatization.
}

static void genSimdClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    const List<Clause> &clauses, mlir::Location loc,
    mlir::omp::SimdOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAligned(clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_simd, clauseOps);
  cp.processNontemporal(clauseOps);
  cp.processOrder(clauseOps);
  cp.processReduction(loc, clauseOps, reductionSyms);
  cp.processSafelen(clauseOps);
  cp.processSimdlen(clauseOps);

  cp.processTODO<clause::Linear>(loc, llvm::omp::Directive::OMPD_simd);
}

static void genSingleClauses(lower::AbstractConverter &converter,
                             semantics::SemanticsContext &semaCtx,
                             const List<Clause> &clauses, mlir::Location loc,
                             mlir::omp::SingleOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processCopyprivate(loc, clauseOps);
  cp.processNowait(clauseOps);
  // TODO Support delayed privatization.
}

static void genTargetClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, lower::pft::Evaluation &eval,
    const List<Clause> &clauses, mlir::Location loc,
    mlir::omp::TargetOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &hasDeviceAddrSyms,
    llvm::SmallVectorImpl<const semantics::Symbol *> &isDevicePtrSyms,
    llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processBare(clauseOps);
  cp.processDepend(clauseOps);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processHasDeviceAddr(clauseOps, hasDeviceAddrSyms);
  if (!hostEvalInfo.empty()) {
    // Only process host_eval if compiling for the host device.
    processHostEvalClauses(converter, semaCtx, stmtCtx, eval, loc);
    hostEvalInfo.back().collectValues(clauseOps.hostEvalVars);
  }
  cp.processIf(llvm::omp::Directive::OMPD_target, clauseOps);
  cp.processIsDevicePtr(clauseOps, isDevicePtrSyms);
  cp.processMap(loc, stmtCtx, clauseOps, &mapSyms);
  cp.processNowait(clauseOps);
  cp.processThreadLimit(stmtCtx, clauseOps);

  cp.processTODO<clause::Allocate, clause::Defaultmap, clause::Firstprivate,
                 clause::InReduction, clause::UsesAllocators>(
      loc, llvm::omp::Directive::OMPD_target);

  // `target private(..)` is only supported in delayed privatization mode.
  if (!enableDelayedPrivatizationStaging)
    cp.processTODO<clause::Private>(loc, llvm::omp::Directive::OMPD_target);
}

static void genTargetDataClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::TargetDataOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceAddrSyms,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDevicePtrSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_target_data, clauseOps);
  cp.processMap(loc, stmtCtx, clauseOps);
  cp.processUseDeviceAddr(stmtCtx, clauseOps, useDeviceAddrSyms);
  cp.processUseDevicePtr(stmtCtx, clauseOps, useDevicePtrSyms);

  // This function implements the deprecated functionality of use_device_ptr
  // that allows users to provide non-CPTR arguments to it with the caveat
  // that the compiler will treat them as use_device_addr. A lot of legacy
  // code may still depend on this functionality, so we should support it
  // in some manner. We do so currently by simply shifting non-cptr operands
  // from the use_device_ptr lists into the use_device_addr lists.
  // TODO: Perhaps create a user provideable compiler option that will
  // re-introduce a hard-error rather than a warning in these cases.
  promoteNonCPtrUseDevicePtrArgsToUseDeviceAddr(
      clauseOps.useDeviceAddrVars, useDeviceAddrSyms,
      clauseOps.useDevicePtrVars, useDevicePtrSyms);
}

static void genTargetEnterExitUpdateDataClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, llvm::omp::Directive directive,
    mlir::omp::TargetEnterExitUpdateDataOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDepend(clauseOps);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processIf(directive, clauseOps);

  if (directive == llvm::omp::Directive::OMPD_target_update)
    cp.processMotionClauses(stmtCtx, clauseOps);
  else
    cp.processMap(loc, stmtCtx, clauseOps);

  cp.processNowait(clauseOps);
}

static void genTaskClauses(lower::AbstractConverter &converter,
                           semantics::SemanticsContext &semaCtx,
                           lower::StatementContext &stmtCtx,
                           const List<Clause> &clauses, mlir::Location loc,
                           mlir::omp::TaskOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processDepend(clauseOps);
  cp.processFinal(stmtCtx, clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_task, clauseOps);
  cp.processMergeable(clauseOps);
  cp.processPriority(stmtCtx, clauseOps);
  cp.processUntied(clauseOps);
  cp.processDetach(clauseOps);
  // TODO Support delayed privatization.

  cp.processTODO<clause::Affinity, clause::InReduction>(
      loc, llvm::omp::Directive::OMPD_task);
}

static void genTaskgroupClauses(lower::AbstractConverter &converter,
                                semantics::SemanticsContext &semaCtx,
                                const List<Clause> &clauses, mlir::Location loc,
                                mlir::omp::TaskgroupOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processTODO<clause::TaskReduction>(loc,
                                        llvm::omp::Directive::OMPD_taskgroup);
}

static void genTaskwaitClauses(lower::AbstractConverter &converter,
                               semantics::SemanticsContext &semaCtx,
                               const List<Clause> &clauses, mlir::Location loc,
                               mlir::omp::TaskwaitOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processTODO<clause::Depend, clause::Nowait>(
      loc, llvm::omp::Directive::OMPD_taskwait);
}

static void genWorkshareClauses(lower::AbstractConverter &converter,
                                semantics::SemanticsContext &semaCtx,
                                lower::StatementContext &stmtCtx,
                                const List<Clause> &clauses, mlir::Location loc,
                                mlir::omp::WorkshareOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processNowait(clauseOps);
}

static void genTeamsClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::TeamsOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_teams, clauseOps);

  if (hostEvalInfo.empty() || !hostEvalInfo.back().apply(clauseOps)) {
    cp.processNumTeams(stmtCtx, clauseOps);
    cp.processThreadLimit(stmtCtx, clauseOps);
  }

  cp.processReduction(loc, clauseOps, reductionSyms);
  // TODO Support delayed privatization.
}

static void genWsloopClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::WsloopOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processNowait(clauseOps);
  cp.processOrder(clauseOps);
  cp.processOrdered(clauseOps);
  cp.processReduction(loc, clauseOps, reductionSyms);
  cp.processSchedule(stmtCtx, clauseOps);

  cp.processTODO<clause::Allocate, clause::Linear>(
      loc, llvm::omp::Directive::OMPD_do);
}

//===----------------------------------------------------------------------===//
// Code generation functions for leaf constructs
//===----------------------------------------------------------------------===//

static mlir::omp::BarrierOp
genBarrierOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
             semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
             mlir::Location loc, const ConstructQueue &queue,
             ConstructQueue::const_iterator item) {
  return converter.getFirOpBuilder().create<mlir::omp::BarrierOp>(loc);
}

static mlir::omp::CriticalOp
genCriticalOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue, ConstructQueue::const_iterator item,
              const std::optional<parser::Name> &name) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::FlatSymbolRefAttr nameAttr;

  if (name) {
    std::string nameStr = name->ToString();
    mlir::ModuleOp mod = firOpBuilder.getModule();
    auto global = mod.lookupSymbol<mlir::omp::CriticalDeclareOp>(nameStr);
    if (!global) {
      mlir::omp::CriticalDeclareOperands clauseOps;
      genCriticalDeclareClauses(converter, semaCtx, item->clauses, loc,
                                clauseOps, nameStr);

      mlir::OpBuilder modBuilder(mod.getBodyRegion());
      global = modBuilder.create<mlir::omp::CriticalDeclareOp>(loc, clauseOps);
    }
    nameAttr = mlir::FlatSymbolRefAttr::get(firOpBuilder.getContext(),
                                            global.getSymName());
  }

  return genOpWithBody<mlir::omp::CriticalOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_critical),
      queue, item, nameAttr);
}

static mlir::omp::FlushOp
genFlushOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
           semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
           mlir::Location loc, const ObjectList &objects,
           const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  llvm::SmallVector<mlir::Value> operandRange;
  genFlushClauses(converter, semaCtx, objects, item->clauses, loc,
                  operandRange);

  return converter.getFirOpBuilder().create<mlir::omp::FlushOp>(
      converter.getCurrentLocation(), operandRange);
}

static mlir::omp::LoopNestOp genLoopNestOp(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::Location loc, const ConstructQueue &queue,
    ConstructQueue::const_iterator item, mlir::omp::LoopNestOperands &clauseOps,
    llvm::ArrayRef<const semantics::Symbol *> iv,
    llvm::ArrayRef<
        std::pair<mlir::omp::BlockArgOpenMPOpInterface, const EntryBlockArgs &>>
        wrapperArgs,
    llvm::omp::Directive directive, DataSharingProcessor &dsp) {
  auto ivCallback = [&](mlir::Operation *op) {
    genLoopVars(op, converter, loc, iv, wrapperArgs);
    return llvm::SmallVector<const semantics::Symbol *>(iv);
  };

  auto *nestedEval =
      getCollapsedLoopEval(eval, getCollapseValue(item->clauses));

  return genOpWithBody<mlir::omp::LoopNestOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, *nestedEval,
                        directive)
          .setClauses(&item->clauses)
          .setDataSharingProcessor(&dsp)
          .setGenRegionEntryCb(ivCallback),
      queue, item, clauseOps);
}

static void genLoopOp(lower::AbstractConverter &converter,
                      lower::SymMap &symTable,
                      semantics::SemanticsContext &semaCtx,
                      lower::pft::Evaluation &eval, mlir::Location loc,
                      const ConstructQueue &queue,
                      ConstructQueue::const_iterator item) {
  mlir::omp::LoopOperands loopClauseOps;
  llvm::SmallVector<const semantics::Symbol *> loopReductionSyms;
  genLoopClauses(converter, semaCtx, item->clauses, loc, loopClauseOps,
                 loopReductionSyms);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           /*useDelayedPrivatization=*/true, symTable);
  dsp.processStep1(&loopClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);

  EntryBlockArgs loopArgs;
  loopArgs.priv.syms = dsp.getDelayedPrivSymbols();
  loopArgs.priv.vars = loopClauseOps.privateVars;
  loopArgs.reduction.syms = loopReductionSyms;
  loopArgs.reduction.vars = loopClauseOps.reductionVars;

  auto loopOp =
      genWrapperOp<mlir::omp::LoopOp>(converter, loc, loopClauseOps, loopArgs);
  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, item,
                loopNestClauseOps, iv, {{loopOp, loopArgs}},
                llvm::omp::Directive::OMPD_loop, dsp);
}

static mlir::omp::MaskedOp
genMaskedOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;
  mlir::omp::MaskedOperands clauseOps;
  genMaskedClauses(converter, semaCtx, stmtCtx, item->clauses, loc, clauseOps);

  return genOpWithBody<mlir::omp::MaskedOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_masked),
      queue, item, clauseOps);
}

static mlir::omp::MasterOp
genMasterOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::const_iterator item) {
  return genOpWithBody<mlir::omp::MasterOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_master),
      queue, item);
}

static mlir::omp::OrderedOp
genOrderedOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
             semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
             mlir::Location loc, const ConstructQueue &queue,
             ConstructQueue::const_iterator item) {
  TODO(loc, "OMPD_ordered");
  return nullptr;
}

static mlir::omp::OrderedRegionOp
genOrderedRegionOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval, mlir::Location loc,
                   const ConstructQueue &queue,
                   ConstructQueue::const_iterator item) {
  mlir::omp::OrderedRegionOperands clauseOps;
  genOrderedRegionClauses(converter, semaCtx, item->clauses, loc, clauseOps);

  return genOpWithBody<mlir::omp::OrderedRegionOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_ordered),
      queue, item, clauseOps);
}

static mlir::omp::ParallelOp
genParallelOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue, ConstructQueue::const_iterator item,
              mlir::omp::ParallelOperands &clauseOps,
              const EntryBlockArgs &args, DataSharingProcessor *dsp,
              bool isComposite = false) {
  auto genRegionEntryCB = [&](mlir::Operation *op) {
    genEntryBlock(converter.getFirOpBuilder(), args, op->getRegion(0));
    bindEntryBlockArgs(
        converter, llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(op), args);
    return llvm::to_vector(args.getSyms());
  };

  assert((!enableDelayedPrivatization || dsp) &&
         "expected valid DataSharingProcessor");
  OpWithBodyGenInfo genInfo =
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_parallel)
          .setClauses(&item->clauses)
          .setGenRegionEntryCb(genRegionEntryCB)
          .setGenSkeletonOnly(isComposite)
          .setDataSharingProcessor(dsp);

  auto parallelOp =
      genOpWithBody<mlir::omp::ParallelOp>(genInfo, queue, item, clauseOps);
  parallelOp.setComposite(isComposite);
  return parallelOp;
}

static mlir::omp::ScanOp
genScanOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
          semantics::SemanticsContext &semaCtx, mlir::Location loc,
          const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  mlir::omp::ScanOperands clauseOps;
  genScanClauses(converter, semaCtx, item->clauses, loc, clauseOps);
  return converter.getFirOpBuilder().create<mlir::omp::ScanOp>(
      converter.getCurrentLocation(), clauseOps);
}

/// This breaks the normal prototype of the gen*Op functions: adding the
/// sectionBlocks argument so that the enclosed section constructs can be
/// lowered here with correct reduction symbol remapping.
static mlir::omp::SectionsOp
genSectionsOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue, ConstructQueue::const_iterator item,
              const parser::OmpSectionBlocks &sectionBlocks) {
  mlir::omp::SectionsOperands clauseOps;
  llvm::SmallVector<const semantics::Symbol *> reductionSyms;
  genSectionsClauses(converter, semaCtx, item->clauses, loc, clauseOps,
                     reductionSyms);

  auto &builder = converter.getFirOpBuilder();

  // Insert privatizations before SECTIONS
  lower::SymMapScope scope(symTable);
  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           lower::omp::isLastItemInQueue(item, queue),
                           /*useDelayedPrivatization=*/false, symTable);
  dsp.processStep1();

  List<Clause> nonDsaClauses;
  List<const clause::Lastprivate *> lastprivates;

  for (const Clause &clause : item->clauses) {
    if (clause.id == llvm::omp::Clause::OMPC_lastprivate) {
      auto &lastp = std::get<clause::Lastprivate>(clause.u);
      lastprivateModifierNotSupported(lastp, converter.getCurrentLocation());
      lastprivates.push_back(&lastp);
    } else {
      switch (clause.id) {
      case llvm::omp::Clause::OMPC_firstprivate:
      case llvm::omp::Clause::OMPC_private:
      case llvm::omp::Clause::OMPC_shared:
        break;
      default:
        nonDsaClauses.push_back(clause);
      }
    }
  }

  // SECTIONS construct.
  auto sectionsOp = builder.create<mlir::omp::SectionsOp>(loc, clauseOps);

  // Create entry block with reduction variables as arguments.
  EntryBlockArgs args;
  // TODO: Add private syms and vars.
  args.reduction.syms = reductionSyms;
  args.reduction.vars = clauseOps.reductionVars;

  genEntryBlock(builder, args, sectionsOp.getRegion());
  mlir::Operation *terminator =
      lower::genOpenMPTerminator(builder, sectionsOp, loc);

  auto genRegionEntryCB = [&](mlir::Operation *op) {
    genEntryBlock(builder, args, op->getRegion(0));
    bindEntryBlockArgs(
        converter, llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(op), args);
    return llvm::to_vector(args.getSyms());
  };

  // Generate nested SECTION constructs.
  // This is done here rather than in genOMP([...], OpenMPSectionConstruct )
  // because we need to run genReductionVars on each omp.section so that the
  // reduction variable gets mapped to the private version
  for (auto [construct, nestedEval] :
       llvm::zip(sectionBlocks.v, eval.getNestedEvaluations())) {
    const auto *sectionConstruct =
        std::get_if<parser::OpenMPSectionConstruct>(&construct.u);
    if (!sectionConstruct) {
      assert(false &&
             "unexpected construct nested inside of SECTIONS construct");
      continue;
    }

    ConstructQueue sectionQueue{buildConstructQueue(
        converter.getFirOpBuilder().getModule(), semaCtx, nestedEval,
        sectionConstruct->source, llvm::omp::Directive::OMPD_section, {})};

    builder.setInsertionPoint(terminator);
    genOpWithBody<mlir::omp::SectionOp>(
        OpWithBodyGenInfo(converter, symTable, semaCtx, loc, nestedEval,
                          llvm::omp::Directive::OMPD_section)
            .setClauses(&sectionQueue.begin()->clauses)
            .setGenRegionEntryCb(genRegionEntryCB),
        sectionQueue, sectionQueue.begin());
  }

  if (!lastprivates.empty()) {
    mlir::Region &sectionsBody = sectionsOp.getRegion();
    assert(sectionsBody.hasOneBlock());
    mlir::Block &body = sectionsBody.front();

    auto lastSectionOp = llvm::find_if(
        llvm::reverse(body.getOperations()), [](const mlir::Operation &op) {
          return llvm::isa<mlir::omp::SectionOp>(op);
        });
    assert(lastSectionOp != body.rend());

    for (const clause::Lastprivate *lastp : lastprivates) {
      builder.setInsertionPoint(
          lastSectionOp->getRegion(0).back().getTerminator());
      mlir::OpBuilder::InsertPoint insp = builder.saveInsertionPoint();
      const auto &objList = std::get<ObjectList>(lastp->t);
      for (const Object &object : objList) {
        semantics::Symbol *sym = object.sym();
        if (const auto *common =
                sym->detailsIf<semantics::CommonBlockDetails>()) {
          for (const auto &obj : common->objects())
            converter.copyHostAssociateVar(*obj, &insp, /*hostIsSource=*/false);
        } else {
          converter.copyHostAssociateVar(*sym, &insp, /*hostIsSource=*/false);
        }
      }
    }
  }

  // Perform DataSharingProcessor's step2 out of SECTIONS
  builder.setInsertionPointAfter(sectionsOp.getOperation());
  dsp.processStep2(sectionsOp, false);
  // Emit implicit barrier to synchronize threads and avoid data
  // races on post-update of lastprivate variables when `nowait`
  // clause is present.
  if (clauseOps.nowait && !lastprivates.empty())
    builder.create<mlir::omp::BarrierOp>(loc);

  return sectionsOp;
}

static void genScopeOp(lower::AbstractConverter &converter,
                       lower::SymMap &symTable,
                       semantics::SemanticsContext &semaCtx,
                       lower::pft::Evaluation &eval, mlir::Location loc,
                       const ConstructQueue &queue,
                       ConstructQueue::const_iterator item) {
  TODO(loc, "Scope construct");
}

static mlir::omp::SingleOp
genSingleOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::const_iterator item) {
  mlir::omp::SingleOperands clauseOps;
  genSingleClauses(converter, semaCtx, item->clauses, loc, clauseOps);

  return genOpWithBody<mlir::omp::SingleOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_single)
          .setClauses(&item->clauses),
      queue, item, clauseOps);
}

static mlir::omp::TargetOp
genTargetOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  lower::StatementContext stmtCtx;
  bool isTargetDevice =
      llvm::cast<mlir::omp::OffloadModuleInterface>(*converter.getModuleOp())
          .getIsTargetDevice();

  // Introduce a new host_eval information structure for this target region.
  if (!isTargetDevice)
    hostEvalInfo.emplace_back();

  mlir::omp::TargetOperands clauseOps;
  llvm::SmallVector<const semantics::Symbol *> mapSyms, isDevicePtrSyms,
      hasDeviceAddrSyms;
  genTargetClauses(converter, semaCtx, stmtCtx, eval, item->clauses, loc,
                   clauseOps, hasDeviceAddrSyms, isDevicePtrSyms, mapSyms);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/
                           lower::omp::isLastItemInQueue(item, queue),
                           /*useDelayedPrivatization=*/true, symTable);
  dsp.processStep1(&clauseOps);

  // 5.8.1 Implicit Data-Mapping Attribute Rules
  // The following code follows the implicit data-mapping rules to map all the
  // symbols used inside the region that do not have explicit data-environment
  // attribute clauses (neither data-sharing; e.g. `private`, nor `map`
  // clauses).
  auto captureImplicitMap = [&](const semantics::Symbol &sym) {
    if (dsp.getAllSymbolsToPrivatize().contains(&sym))
      return;

    // Structure component symbols don't have bindings, and can only be
    // explicitly mapped individually. If a member is captured implicitly
    // we map the entirety of the derived type when we find its symbol.
    if (sym.owner().IsDerivedType())
      return;

    // if the symbol is part of an already mapped common block, do not make a
    // map for it.
    if (const Fortran::semantics::Symbol *common =
            Fortran::semantics::FindCommonBlockContaining(sym.GetUltimate()))
      if (llvm::is_contained(mapSyms, common))
        return;

    // If we come across a symbol without a symbol address, we
    // return as we cannot process it, this is intended as a
    // catch all early exit for symbols that do not have a
    // corresponding extended value. Such as subroutines,
    // interfaces and named blocks.
    if (!converter.getSymbolAddress(sym))
      return;

    if (!llvm::is_contained(mapSyms, &sym)) {
      if (const auto *details =
              sym.template detailsIf<semantics::HostAssocDetails>())
        converter.copySymbolBinding(details->symbol(), sym);
      std::stringstream name;
      fir::ExtendedValue dataExv = converter.getSymbolExtendedValue(sym);
      name << sym.name().ToString();

      fir::factory::AddrAndBoundsInfo info =
          Fortran::lower::getDataOperandBaseAddr(
              converter, firOpBuilder, sym, converter.getCurrentLocation());
      llvm::SmallVector<mlir::Value> bounds =
          fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
                                             mlir::omp::MapBoundsType>(
              firOpBuilder, info, dataExv,
              semantics::IsAssumedSizeArray(sym.GetUltimate()),
              converter.getCurrentLocation());

      llvm::omp::OpenMPOffloadMappingFlags mapFlag =
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
      mlir::omp::VariableCaptureKind captureKind =
          mlir::omp::VariableCaptureKind::ByRef;

      mlir::Value baseOp = info.rawInput;
      mlir::Type eleType = baseOp.getType();
      if (auto refType = mlir::dyn_cast<fir::ReferenceType>(baseOp.getType()))
        eleType = refType.getElementType();

      // If a variable is specified in declare target link and if device
      // type is not specified as `nohost`, it needs to be mapped tofrom
      mlir::ModuleOp mod = firOpBuilder.getModule();
      mlir::Operation *op = mod.lookupSymbol(converter.mangleName(sym));
      auto declareTargetOp =
          llvm::dyn_cast_if_present<mlir::omp::DeclareTargetInterface>(op);
      if (declareTargetOp && declareTargetOp.isDeclareTarget()) {
        if (declareTargetOp.getDeclareTargetCaptureClause() ==
                mlir::omp::DeclareTargetCaptureClause::link &&
            declareTargetOp.getDeclareTargetDeviceType() !=
                mlir::omp::DeclareTargetDeviceType::nohost) {
          mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
          mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
        }
      } else if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
        captureKind = mlir::omp::VariableCaptureKind::ByCopy;
      } else if (!fir::isa_builtin_cptr_type(eleType)) {
        mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
        mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
      }
      auto location =
          mlir::NameLoc::get(mlir::StringAttr::get(firOpBuilder.getContext(),
                                                   sym.name().ToString()),
                             baseOp.getLoc());
      mlir::Value mapOp = createMapInfoOp(
          firOpBuilder, location, baseOp, /*varPtrPtr=*/mlir::Value{},
          name.str(), bounds, /*members=*/{},
          /*membersIndex=*/mlir::ArrayAttr{},
          static_cast<
              std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
              mapFlag),
          captureKind, baseOp.getType());

      clauseOps.mapVars.push_back(mapOp);
      mapSyms.push_back(&sym);
    }
  };
  lower::pft::visitAllSymbols(eval, captureImplicitMap);

  auto targetOp = firOpBuilder.create<mlir::omp::TargetOp>(loc, clauseOps);

  llvm::SmallVector<mlir::Value> mapBaseValues;
  extractMappedBaseValues(clauseOps.mapVars, mapBaseValues);

  EntryBlockArgs args;
  args.hostEvalVars = clauseOps.hostEvalVars;
  // TODO: Add in_reduction syms and vars.
  args.map.syms = mapSyms;
  args.map.vars = mapBaseValues;
  args.priv.syms = dsp.getDelayedPrivSymbols();
  args.priv.vars = clauseOps.privateVars;

  genBodyOfTargetOp(converter, symTable, semaCtx, eval, targetOp, args, loc,
                    queue, item, dsp);

  // Remove the host_eval information structure created for this target region.
  if (!isTargetDevice)
    hostEvalInfo.pop_back();
  return targetOp;
}

static mlir::omp::TargetDataOp
genTargetDataOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
                semantics::SemanticsContext &semaCtx,
                lower::pft::Evaluation &eval, mlir::Location loc,
                const ConstructQueue &queue,
                ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;
  mlir::omp::TargetDataOperands clauseOps;
  llvm::SmallVector<const semantics::Symbol *> useDeviceAddrSyms,
      useDevicePtrSyms;
  genTargetDataClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                       clauseOps, useDeviceAddrSyms, useDevicePtrSyms);

  auto targetDataOp =
      converter.getFirOpBuilder().create<mlir::omp::TargetDataOp>(loc,
                                                                  clauseOps);

  llvm::SmallVector<mlir::Value> useDeviceAddrBaseValues,
      useDevicePtrBaseValues;
  extractMappedBaseValues(clauseOps.useDeviceAddrVars, useDeviceAddrBaseValues);
  extractMappedBaseValues(clauseOps.useDevicePtrVars, useDevicePtrBaseValues);

  EntryBlockArgs args;
  args.useDeviceAddr.syms = useDeviceAddrSyms;
  args.useDeviceAddr.vars = useDeviceAddrBaseValues;
  args.useDevicePtr.syms = useDevicePtrSyms;
  args.useDevicePtr.vars = useDevicePtrBaseValues;

  genBodyOfTargetDataOp(converter, symTable, semaCtx, eval, targetDataOp, args,
                        loc, queue, item);
  return targetDataOp;
}

template <typename OpTy>
static OpTy genTargetEnterExitUpdateDataOp(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  lower::StatementContext stmtCtx;

  // GCC 9.3.0 emits a (probably) bogus warning about an unused variable.
  [[maybe_unused]] llvm::omp::Directive directive;
  if constexpr (std::is_same_v<OpTy, mlir::omp::TargetEnterDataOp>) {
    directive = llvm::omp::Directive::OMPD_target_enter_data;
  } else if constexpr (std::is_same_v<OpTy, mlir::omp::TargetExitDataOp>) {
    directive = llvm::omp::Directive::OMPD_target_exit_data;
  } else if constexpr (std::is_same_v<OpTy, mlir::omp::TargetUpdateOp>) {
    directive = llvm::omp::Directive::OMPD_target_update;
  } else {
    llvm_unreachable("Unexpected TARGET DATA construct");
  }

  mlir::omp::TargetEnterExitUpdateDataOperands clauseOps;
  genTargetEnterExitUpdateDataClauses(converter, semaCtx, stmtCtx,
                                      item->clauses, loc, directive, clauseOps);

  return firOpBuilder.create<OpTy>(loc, clauseOps);
}

static mlir::omp::TaskOp
genTaskOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
          semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
          mlir::Location loc, const ConstructQueue &queue,
          ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;
  mlir::omp::TaskOperands clauseOps;
  genTaskClauses(converter, semaCtx, stmtCtx, item->clauses, loc, clauseOps);

  if (!enableDelayedPrivatization)
    return genOpWithBody<mlir::omp::TaskOp>(
        OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                          llvm::omp::Directive::OMPD_task)
            .setClauses(&item->clauses),
        queue, item, clauseOps);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           lower::omp::isLastItemInQueue(item, queue),
                           /*useDelayedPrivatization=*/true, symTable);
  dsp.processStep1(&clauseOps);

  EntryBlockArgs taskArgs;
  taskArgs.priv.syms = dsp.getDelayedPrivSymbols();
  taskArgs.priv.vars = clauseOps.privateVars;

  auto genRegionEntryCB = [&](mlir::Operation *op) {
    genEntryBlock(converter.getFirOpBuilder(), taskArgs, op->getRegion(0));
    bindEntryBlockArgs(converter,
                       llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(op),
                       taskArgs);
    return llvm::to_vector(taskArgs.priv.syms);
  };

  return genOpWithBody<mlir::omp::TaskOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_task)
          .setClauses(&item->clauses)
          .setDataSharingProcessor(&dsp)
          .setGenRegionEntryCb(genRegionEntryCB),
      queue, item, clauseOps);
}

static mlir::omp::TaskgroupOp
genTaskgroupOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
               semantics::SemanticsContext &semaCtx,
               lower::pft::Evaluation &eval, mlir::Location loc,
               const ConstructQueue &queue,
               ConstructQueue::const_iterator item) {
  mlir::omp::TaskgroupOperands clauseOps;
  genTaskgroupClauses(converter, semaCtx, item->clauses, loc, clauseOps);

  return genOpWithBody<mlir::omp::TaskgroupOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_taskgroup)
          .setClauses(&item->clauses),
      queue, item, clauseOps);
}

static mlir::omp::TaskwaitOp
genTaskwaitOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue,
              ConstructQueue::const_iterator item) {
  mlir::omp::TaskwaitOperands clauseOps;
  genTaskwaitClauses(converter, semaCtx, item->clauses, loc, clauseOps);
  return converter.getFirOpBuilder().create<mlir::omp::TaskwaitOp>(loc,
                                                                   clauseOps);
}

static mlir::omp::TaskyieldOp
genTaskyieldOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
               semantics::SemanticsContext &semaCtx,
               lower::pft::Evaluation &eval, mlir::Location loc,
               const ConstructQueue &queue,
               ConstructQueue::const_iterator item) {
  return converter.getFirOpBuilder().create<mlir::omp::TaskyieldOp>(loc);
}

static mlir::omp::WorkshareOp
genWorkshareOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
               semantics::SemanticsContext &semaCtx,
               lower::pft::Evaluation &eval, mlir::Location loc,
               const ConstructQueue &queue,
               ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;
  mlir::omp::WorkshareOperands clauseOps;
  genWorkshareClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                      clauseOps);

  return genOpWithBody<mlir::omp::WorkshareOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_workshare)
          .setClauses(&item->clauses),
      queue, item, clauseOps);
}

static mlir::omp::TeamsOp
genTeamsOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
           semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
           mlir::Location loc, const ConstructQueue &queue,
           ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;

  mlir::omp::TeamsOperands clauseOps;
  llvm::SmallVector<const semantics::Symbol *> reductionSyms;
  genTeamsClauses(converter, semaCtx, stmtCtx, item->clauses, loc, clauseOps,
                  reductionSyms);

  EntryBlockArgs args;
  // TODO: Add private syms and vars.
  args.reduction.syms = reductionSyms;
  args.reduction.vars = clauseOps.reductionVars;

  auto genRegionEntryCB = [&](mlir::Operation *op) {
    genEntryBlock(converter.getFirOpBuilder(), args, op->getRegion(0));
    bindEntryBlockArgs(
        converter, llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(op), args);
    return llvm::to_vector(args.getSyms());
  };

  return genOpWithBody<mlir::omp::TeamsOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_teams)
          .setClauses(&item->clauses)
          .setGenRegionEntryCb(genRegionEntryCB),
      queue, item, clauseOps);
}

//===----------------------------------------------------------------------===//
// Code generation functions for the standalone version of constructs that can
// also be a leaf of a composite construct
//===----------------------------------------------------------------------===//

static void genStandaloneDistribute(lower::AbstractConverter &converter,
                                    lower::SymMap &symTable,
                                    semantics::SemanticsContext &semaCtx,
                                    lower::pft::Evaluation &eval,
                                    mlir::Location loc,
                                    const ConstructQueue &queue,
                                    ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;

  mlir::omp::DistributeOperands distributeClauseOps;
  genDistributeClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                       distributeClauseOps);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           enableDelayedPrivatizationStaging, symTable);
  dsp.processStep1(&distributeClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);

  EntryBlockArgs distributeArgs;
  distributeArgs.priv.syms = dsp.getDelayedPrivSymbols();
  distributeArgs.priv.vars = distributeClauseOps.privateVars;
  auto distributeOp = genWrapperOp<mlir::omp::DistributeOp>(
      converter, loc, distributeClauseOps, distributeArgs);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, item,
                loopNestClauseOps, iv, {{distributeOp, distributeArgs}},
                llvm::omp::Directive::OMPD_distribute, dsp);
}

static void genStandaloneDo(lower::AbstractConverter &converter,
                            lower::SymMap &symTable,
                            semantics::SemanticsContext &semaCtx,
                            lower::pft::Evaluation &eval, mlir::Location loc,
                            const ConstructQueue &queue,
                            ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;

  mlir::omp::WsloopOperands wsloopClauseOps;
  llvm::SmallVector<const semantics::Symbol *> wsloopReductionSyms;
  genWsloopClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                   wsloopClauseOps, wsloopReductionSyms);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           enableDelayedPrivatization, symTable);
  dsp.processStep1(&wsloopClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);

  EntryBlockArgs wsloopArgs;
  wsloopArgs.priv.syms = dsp.getDelayedPrivSymbols();
  wsloopArgs.priv.vars = wsloopClauseOps.privateVars;
  wsloopArgs.reduction.syms = wsloopReductionSyms;
  wsloopArgs.reduction.vars = wsloopClauseOps.reductionVars;
  auto wsloopOp = genWrapperOp<mlir::omp::WsloopOp>(
      converter, loc, wsloopClauseOps, wsloopArgs);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, item,
                loopNestClauseOps, iv, {{wsloopOp, wsloopArgs}},
                llvm::omp::Directive::OMPD_do, dsp);
}

static void genStandaloneParallel(lower::AbstractConverter &converter,
                                  lower::SymMap &symTable,
                                  semantics::SemanticsContext &semaCtx,
                                  lower::pft::Evaluation &eval,
                                  mlir::Location loc,
                                  const ConstructQueue &queue,
                                  ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;

  mlir::omp::ParallelOperands parallelClauseOps;
  llvm::SmallVector<const semantics::Symbol *> parallelReductionSyms;
  genParallelClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                     parallelClauseOps, parallelReductionSyms);

  std::optional<DataSharingProcessor> dsp;
  if (enableDelayedPrivatization) {
    dsp.emplace(converter, semaCtx, item->clauses, eval,
                lower::omp::isLastItemInQueue(item, queue),
                /*useDelayedPrivatization=*/true, symTable);
    dsp->processStep1(&parallelClauseOps);
  }

  EntryBlockArgs parallelArgs;
  if (dsp)
    parallelArgs.priv.syms = dsp->getDelayedPrivSymbols();
  parallelArgs.priv.vars = parallelClauseOps.privateVars;
  parallelArgs.reduction.syms = parallelReductionSyms;
  parallelArgs.reduction.vars = parallelClauseOps.reductionVars;
  genParallelOp(converter, symTable, semaCtx, eval, loc, queue, item,
                parallelClauseOps, parallelArgs,
                enableDelayedPrivatization ? &dsp.value() : nullptr);
}

static void genStandaloneSimd(lower::AbstractConverter &converter,
                              lower::SymMap &symTable,
                              semantics::SemanticsContext &semaCtx,
                              lower::pft::Evaluation &eval, mlir::Location loc,
                              const ConstructQueue &queue,
                              ConstructQueue::const_iterator item) {
  mlir::omp::SimdOperands simdClauseOps;
  llvm::SmallVector<const semantics::Symbol *> simdReductionSyms;
  genSimdClauses(converter, semaCtx, item->clauses, loc, simdClauseOps,
                 simdReductionSyms);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           enableDelayedPrivatization, symTable);
  dsp.processStep1(&simdClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);

  EntryBlockArgs simdArgs;
  simdArgs.priv.syms = dsp.getDelayedPrivSymbols();
  simdArgs.priv.vars = simdClauseOps.privateVars;
  simdArgs.reduction.syms = simdReductionSyms;
  simdArgs.reduction.vars = simdClauseOps.reductionVars;
  auto simdOp =
      genWrapperOp<mlir::omp::SimdOp>(converter, loc, simdClauseOps, simdArgs);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, item,
                loopNestClauseOps, iv, {{simdOp, simdArgs}},
                llvm::omp::Directive::OMPD_simd, dsp);
}

static void genStandaloneTaskloop(lower::AbstractConverter &converter,
                                  lower::SymMap &symTable,
                                  semantics::SemanticsContext &semaCtx,
                                  lower::pft::Evaluation &eval,
                                  mlir::Location loc,
                                  const ConstructQueue &queue,
                                  ConstructQueue::const_iterator item) {
  TODO(loc, "Taskloop construct");
}

//===----------------------------------------------------------------------===//
// Code generation functions for composite constructs
//===----------------------------------------------------------------------===//

static void genCompositeDistributeParallelDo(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::Location loc, const ConstructQueue &queue,
    ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;

  assert(std::distance(item, queue.end()) == 3 && "Invalid leaf constructs");
  ConstructQueue::const_iterator distributeItem = item;
  ConstructQueue::const_iterator parallelItem = std::next(distributeItem);
  ConstructQueue::const_iterator doItem = std::next(parallelItem);

  // Create parent omp.parallel first.
  mlir::omp::ParallelOperands parallelClauseOps;
  llvm::SmallVector<const semantics::Symbol *> parallelReductionSyms;
  genParallelClauses(converter, semaCtx, stmtCtx, parallelItem->clauses, loc,
                     parallelClauseOps, parallelReductionSyms);

  DataSharingProcessor dsp(converter, semaCtx, doItem->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           /*useDelayedPrivatization=*/true, symTable);
  dsp.processStep1(&parallelClauseOps);

  EntryBlockArgs parallelArgs;
  parallelArgs.priv.syms = dsp.getDelayedPrivSymbols();
  parallelArgs.priv.vars = parallelClauseOps.privateVars;
  parallelArgs.reduction.syms = parallelReductionSyms;
  parallelArgs.reduction.vars = parallelClauseOps.reductionVars;
  genParallelOp(converter, symTable, semaCtx, eval, loc, queue, parallelItem,
                parallelClauseOps, parallelArgs, &dsp, /*isComposite=*/true);

  // Clause processing.
  mlir::omp::DistributeOperands distributeClauseOps;
  genDistributeClauses(converter, semaCtx, stmtCtx, distributeItem->clauses,
                       loc, distributeClauseOps);

  mlir::omp::WsloopOperands wsloopClauseOps;
  llvm::SmallVector<const semantics::Symbol *> wsloopReductionSyms;
  genWsloopClauses(converter, semaCtx, stmtCtx, doItem->clauses, loc,
                   wsloopClauseOps, wsloopReductionSyms);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, doItem->clauses, loc,
                     loopNestClauseOps, iv);

  // Operation creation.
  EntryBlockArgs distributeArgs;
  // TODO: Add private syms and vars.
  auto distributeOp = genWrapperOp<mlir::omp::DistributeOp>(
      converter, loc, distributeClauseOps, distributeArgs);
  distributeOp.setComposite(/*val=*/true);

  EntryBlockArgs wsloopArgs;
  // TODO: Add private syms and vars.
  wsloopArgs.reduction.syms = wsloopReductionSyms;
  wsloopArgs.reduction.vars = wsloopClauseOps.reductionVars;
  auto wsloopOp = genWrapperOp<mlir::omp::WsloopOp>(
      converter, loc, wsloopClauseOps, wsloopArgs);
  wsloopOp.setComposite(/*val=*/true);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, doItem,
                loopNestClauseOps, iv,
                {{distributeOp, distributeArgs}, {wsloopOp, wsloopArgs}},
                llvm::omp::Directive::OMPD_distribute_parallel_do, dsp);
}

static void genCompositeDistributeParallelDoSimd(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::Location loc, const ConstructQueue &queue,
    ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;

  assert(std::distance(item, queue.end()) == 4 && "Invalid leaf constructs");
  ConstructQueue::const_iterator distributeItem = item;
  ConstructQueue::const_iterator parallelItem = std::next(distributeItem);
  ConstructQueue::const_iterator doItem = std::next(parallelItem);
  ConstructQueue::const_iterator simdItem = std::next(doItem);

  // Create parent omp.parallel first.
  mlir::omp::ParallelOperands parallelClauseOps;
  llvm::SmallVector<const semantics::Symbol *> parallelReductionSyms;
  genParallelClauses(converter, semaCtx, stmtCtx, parallelItem->clauses, loc,
                     parallelClauseOps, parallelReductionSyms);

  DataSharingProcessor dsp(converter, semaCtx, simdItem->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           /*useDelayedPrivatization=*/true, symTable);
  dsp.processStep1(&parallelClauseOps);

  EntryBlockArgs parallelArgs;
  parallelArgs.priv.syms = dsp.getDelayedPrivSymbols();
  parallelArgs.priv.vars = parallelClauseOps.privateVars;
  parallelArgs.reduction.syms = parallelReductionSyms;
  parallelArgs.reduction.vars = parallelClauseOps.reductionVars;
  genParallelOp(converter, symTable, semaCtx, eval, loc, queue, parallelItem,
                parallelClauseOps, parallelArgs, &dsp, /*isComposite=*/true);

  // Clause processing.
  mlir::omp::DistributeOperands distributeClauseOps;
  genDistributeClauses(converter, semaCtx, stmtCtx, distributeItem->clauses,
                       loc, distributeClauseOps);

  mlir::omp::WsloopOperands wsloopClauseOps;
  llvm::SmallVector<const semantics::Symbol *> wsloopReductionSyms;
  genWsloopClauses(converter, semaCtx, stmtCtx, doItem->clauses, loc,
                   wsloopClauseOps, wsloopReductionSyms);

  mlir::omp::SimdOperands simdClauseOps;
  llvm::SmallVector<const semantics::Symbol *> simdReductionSyms;
  genSimdClauses(converter, semaCtx, simdItem->clauses, loc, simdClauseOps,
                 simdReductionSyms);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, simdItem->clauses, loc,
                     loopNestClauseOps, iv);

  // Operation creation.
  EntryBlockArgs distributeArgs;
  // TODO: Add private syms and vars.
  auto distributeOp = genWrapperOp<mlir::omp::DistributeOp>(
      converter, loc, distributeClauseOps, distributeArgs);
  distributeOp.setComposite(/*val=*/true);

  EntryBlockArgs wsloopArgs;
  // TODO: Add private syms and vars.
  wsloopArgs.reduction.syms = wsloopReductionSyms;
  wsloopArgs.reduction.vars = wsloopClauseOps.reductionVars;
  auto wsloopOp = genWrapperOp<mlir::omp::WsloopOp>(
      converter, loc, wsloopClauseOps, wsloopArgs);
  wsloopOp.setComposite(/*val=*/true);

  EntryBlockArgs simdArgs;
  // TODO: Add private syms and vars.
  simdArgs.reduction.syms = simdReductionSyms;
  simdArgs.reduction.vars = simdClauseOps.reductionVars;
  auto simdOp =
      genWrapperOp<mlir::omp::SimdOp>(converter, loc, simdClauseOps, simdArgs);
  simdOp.setComposite(/*val=*/true);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, simdItem,
                loopNestClauseOps, iv,
                {{distributeOp, distributeArgs},
                 {wsloopOp, wsloopArgs},
                 {simdOp, simdArgs}},
                llvm::omp::Directive::OMPD_distribute_parallel_do_simd, dsp);
}

static void genCompositeDistributeSimd(lower::AbstractConverter &converter,
                                       lower::SymMap &symTable,
                                       semantics::SemanticsContext &semaCtx,
                                       lower::pft::Evaluation &eval,
                                       mlir::Location loc,
                                       const ConstructQueue &queue,
                                       ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;

  assert(std::distance(item, queue.end()) == 2 && "Invalid leaf constructs");
  ConstructQueue::const_iterator distributeItem = item;
  ConstructQueue::const_iterator simdItem = std::next(distributeItem);

  // Clause processing.
  mlir::omp::DistributeOperands distributeClauseOps;
  genDistributeClauses(converter, semaCtx, stmtCtx, distributeItem->clauses,
                       loc, distributeClauseOps);

  mlir::omp::SimdOperands simdClauseOps;
  llvm::SmallVector<const semantics::Symbol *> simdReductionSyms;
  genSimdClauses(converter, semaCtx, simdItem->clauses, loc, simdClauseOps,
                 simdReductionSyms);

  // TODO: Support delayed privatization.
  DataSharingProcessor dsp(converter, semaCtx, simdItem->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           /*useDelayedPrivatization=*/false, symTable);
  dsp.processStep1();

  // Pass the innermost leaf construct's clauses because that's where COLLAPSE
  // is placed by construct decomposition.
  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, simdItem->clauses, loc,
                     loopNestClauseOps, iv);

  // Operation creation.
  EntryBlockArgs distributeArgs;
  // TODO: Add private syms and vars.
  auto distributeOp = genWrapperOp<mlir::omp::DistributeOp>(
      converter, loc, distributeClauseOps, distributeArgs);
  distributeOp.setComposite(/*val=*/true);

  EntryBlockArgs simdArgs;
  // TODO: Add private syms and vars.
  simdArgs.reduction.syms = simdReductionSyms;
  simdArgs.reduction.vars = simdClauseOps.reductionVars;
  auto simdOp =
      genWrapperOp<mlir::omp::SimdOp>(converter, loc, simdClauseOps, simdArgs);
  simdOp.setComposite(/*val=*/true);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, simdItem,
                loopNestClauseOps, iv,
                {{distributeOp, distributeArgs}, {simdOp, simdArgs}},
                llvm::omp::Directive::OMPD_distribute_simd, dsp);
}

static void genCompositeDoSimd(lower::AbstractConverter &converter,
                               lower::SymMap &symTable,
                               semantics::SemanticsContext &semaCtx,
                               lower::pft::Evaluation &eval, mlir::Location loc,
                               const ConstructQueue &queue,
                               ConstructQueue::const_iterator item) {
  lower::StatementContext stmtCtx;

  assert(std::distance(item, queue.end()) == 2 && "Invalid leaf constructs");
  ConstructQueue::const_iterator doItem = item;
  ConstructQueue::const_iterator simdItem = std::next(doItem);

  // Clause processing.
  mlir::omp::WsloopOperands wsloopClauseOps;
  llvm::SmallVector<const semantics::Symbol *> wsloopReductionSyms;
  genWsloopClauses(converter, semaCtx, stmtCtx, doItem->clauses, loc,
                   wsloopClauseOps, wsloopReductionSyms);

  mlir::omp::SimdOperands simdClauseOps;
  llvm::SmallVector<const semantics::Symbol *> simdReductionSyms;
  genSimdClauses(converter, semaCtx, simdItem->clauses, loc, simdClauseOps,
                 simdReductionSyms);

  // TODO: Support delayed privatization.
  DataSharingProcessor dsp(converter, semaCtx, simdItem->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           /*useDelayedPrivatization=*/false, symTable);
  dsp.processStep1();

  // Pass the innermost leaf construct's clauses because that's where COLLAPSE
  // is placed by construct decomposition.
  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, simdItem->clauses, loc,
                     loopNestClauseOps, iv);

  // Operation creation.
  EntryBlockArgs wsloopArgs;
  // TODO: Add private syms and vars.
  wsloopArgs.reduction.syms = wsloopReductionSyms;
  wsloopArgs.reduction.vars = wsloopClauseOps.reductionVars;
  auto wsloopOp = genWrapperOp<mlir::omp::WsloopOp>(
      converter, loc, wsloopClauseOps, wsloopArgs);
  wsloopOp.setComposite(/*val=*/true);

  EntryBlockArgs simdArgs;
  // TODO: Add private syms and vars.
  simdArgs.reduction.syms = simdReductionSyms;
  simdArgs.reduction.vars = simdClauseOps.reductionVars;
  auto simdOp =
      genWrapperOp<mlir::omp::SimdOp>(converter, loc, simdClauseOps, simdArgs);
  simdOp.setComposite(/*val=*/true);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, simdItem,
                loopNestClauseOps, iv,
                {{wsloopOp, wsloopArgs}, {simdOp, simdArgs}},
                llvm::omp::Directive::OMPD_do_simd, dsp);
}

static void genCompositeTaskloopSimd(lower::AbstractConverter &converter,
                                     lower::SymMap &symTable,
                                     semantics::SemanticsContext &semaCtx,
                                     lower::pft::Evaluation &eval,
                                     mlir::Location loc,
                                     const ConstructQueue &queue,
                                     ConstructQueue::const_iterator item) {
  assert(std::distance(item, queue.end()) == 2 && "Invalid leaf constructs");
  TODO(loc, "Composite TASKLOOP SIMD");
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

static bool genOMPCompositeDispatch(lower::AbstractConverter &converter,
                                    lower::SymMap &symTable,
                                    semantics::SemanticsContext &semaCtx,
                                    lower::pft::Evaluation &eval,
                                    mlir::Location loc,
                                    const ConstructQueue &queue,
                                    ConstructQueue::const_iterator item) {
  using llvm::omp::Directive;
  using lower::omp::matchLeafSequence;

  // TODO: Privatization for composite constructs is currently only done based
  // on the clauses for their last leaf construct, which may not always be
  // correct. Consider per-leaf privatization of composite constructs once
  // delayed privatization is supported by all participating ops.
  if (matchLeafSequence(item, queue, Directive::OMPD_distribute_parallel_do))
    genCompositeDistributeParallelDo(converter, symTable, semaCtx, eval, loc,
                                     queue, item);
  else if (matchLeafSequence(item, queue,
                             Directive::OMPD_distribute_parallel_do_simd))
    genCompositeDistributeParallelDoSimd(converter, symTable, semaCtx, eval,
                                         loc, queue, item);
  else if (matchLeafSequence(item, queue, Directive::OMPD_distribute_simd))
    genCompositeDistributeSimd(converter, symTable, semaCtx, eval, loc, queue,
                               item);
  else if (matchLeafSequence(item, queue, Directive::OMPD_do_simd))
    genCompositeDoSimd(converter, symTable, semaCtx, eval, loc, queue, item);
  else if (matchLeafSequence(item, queue, Directive::OMPD_taskloop_simd))
    genCompositeTaskloopSimd(converter, symTable, semaCtx, eval, loc, queue,
                             item);
  else
    return false;

  return true;
}

static void genOMPDispatch(lower::AbstractConverter &converter,
                           lower::SymMap &symTable,
                           semantics::SemanticsContext &semaCtx,
                           lower::pft::Evaluation &eval, mlir::Location loc,
                           const ConstructQueue &queue,
                           ConstructQueue::const_iterator item) {
  assert(item != queue.end());

  bool loopLeaf = llvm::omp::getDirectiveAssociation(item->id) ==
                  llvm::omp::Association::Loop;
  if (loopLeaf) {
    symTable.pushScope();
    if (genOMPCompositeDispatch(converter, symTable, semaCtx, eval, loc, queue,
                                item)) {
      symTable.popScope();
      return;
    }
  }

  switch (llvm::omp::Directive dir = item->id) {
  case llvm::omp::Directive::OMPD_barrier:
    genBarrierOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_distribute:
    genStandaloneDistribute(converter, symTable, semaCtx, eval, loc, queue,
                            item);
    break;
  case llvm::omp::Directive::OMPD_do:
    genStandaloneDo(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_loop:
    genLoopOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_masked:
    genMaskedOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_master:
    genMasterOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_ordered:
    // Block-associated "ordered" construct.
    genOrderedRegionOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_parallel:
    genStandaloneParallel(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_scan:
    genScanOp(converter, symTable, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_section:
    llvm_unreachable("genOMPDispatch: OMPD_section");
    // Lowered in the enclosing genSectionsOp.
    break;
  case llvm::omp::Directive::OMPD_sections:
    // Called directly from genOMP([...], OpenMPSectionsConstruct) because it
    // has a different prototype.
    // This code path is still taken when iterating through the construct queue
    // in genBodyOfOp
    break;
  case llvm::omp::Directive::OMPD_simd:
    genStandaloneSimd(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_scope:
    genScopeOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_single:
    genSingleOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target:
    genTargetOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_data:
    genTargetDataOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    genTargetEnterExitUpdateDataOp<mlir::omp::TargetEnterDataOp>(
        converter, symTable, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_exit_data:
    genTargetEnterExitUpdateDataOp<mlir::omp::TargetExitDataOp>(
        converter, symTable, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_update:
    genTargetEnterExitUpdateDataOp<mlir::omp::TargetUpdateOp>(
        converter, symTable, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_task:
    genTaskOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskgroup:
    genTaskgroupOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskloop:
    genStandaloneTaskloop(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    genTaskwaitOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    genTaskyieldOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_teams:
    genTeamsOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_tile:
  case llvm::omp::Directive::OMPD_unroll:
    TODO(loc, "Unhandled loop directive (" +
                  llvm::omp::getOpenMPDirectiveName(dir) + ")");
  // case llvm::omp::Directive::OMPD_workdistribute:
  case llvm::omp::Directive::OMPD_workshare:
    genWorkshareOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  default:
    // Combined and composite constructs should have been split into a sequence
    // of leaf constructs when building the construct queue.
    assert(!llvm::omp::isLeafConstruct(dir) &&
           "Unexpected compound construct.");
    break;
  }

  if (loopLeaf)
    symTable.popScope();
}

//===----------------------------------------------------------------------===//
// OpenMPDeclarativeConstruct visitors
//===----------------------------------------------------------------------===//
static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPUtilityConstruct &);

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPDeclarativeAllocate &declarativeAllocate) {
  TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
}

static void genOMP(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareReductionConstruct &declareReductionConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPDeclareReductionConstruct");
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPDeclareSimdConstruct &declareSimdConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPDeclareSimdConstruct");
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPDeclareMapperConstruct &declareMapperConstruct) {
  mlir::Location loc = converter.genLocation(declareMapperConstruct.source);
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  lower::StatementContext stmtCtx;
  const auto &spec =
      std::get<parser::OmpDeclareMapperSpecifier>(declareMapperConstruct.t);
  const auto &mapperName{std::get<std::optional<parser::Name>>(spec.t)};
  const auto &varType{std::get<parser::TypeSpec>(spec.t)};
  const auto &varName{std::get<parser::Name>(spec.t)};
  assert(varType.declTypeSpec->category() ==
             semantics::DeclTypeSpec::Category::TypeDerived &&
         "Expected derived type");

  std::string mapperNameStr;
  if (mapperName.has_value())
    mapperNameStr = mapperName->ToString();
  else
    mapperNameStr =
        "default_" + varType.declTypeSpec->derivedTypeSpec().name().ToString();
  auto mlirType = converter.genType(varType.declTypeSpec->derivedTypeSpec());
  auto declMapperOp = firOpBuilder.create<mlir::omp::DeclareMapperOp>(
      loc, mapperNameStr, mlirType);
  auto &region = declMapperOp.getRegion();

  // Save insert point just after the DeclMapperOp.
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.createBlock(&region);
  auto varVal =
      firOpBuilder.createTemporaryAlloc(loc, mlirType, varName.ToString());
  converter.bindSymbol(*varName.symbol, varVal);

  // Insert dummy instruction to remember the insertion position. The
  // marker will be deleted by clean up passes since there are no uses.
  // Remembering the position for further insertion is important since
  // there are hlfir.declares inserted above while setting block arguments
  // and new code from the body should be inserted after that.
  mlir::Value undefMarker =
      firOpBuilder.create<fir::UndefOp>(loc, firOpBuilder.getIndexType());

  // Create blocks for unstructured regions. This has to be done since
  // blocks are initially allocated with the function as the parent region.
  if (eval.lowerAsUnstructured()) {
    lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp, mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  firOpBuilder.create<mlir::omp::TerminatorOp>(loc);

  // Set the insertion point after the marker.
  firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());
  genNestedEvaluations(converter, eval);

  // Populate the declareMapper region with the map information.
  mlir::omp::DeclareMapperInfoOperands clauseOps;
  const auto *clauseList{
      parser::Unwrap<parser::OmpClauseList>(declareMapperConstruct.t)};
  List<Clause> clauses = makeClauses(*clauseList, semaCtx);
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processMap(loc, stmtCtx, clauseOps);

  firOpBuilder.create<mlir::omp::DeclareMapperInfoOp>(loc, clauseOps.mapVars);

  // Restore the insert point to just after the DeclareMapperOp.
  firOpBuilder.restoreInsertionPoint(insPt);
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct) {
  mlir::omp::DeclareTargetOperands clauseOps;
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  getDeclareTargetInfo(converter, semaCtx, eval, declareTargetConstruct,
                       clauseOps, symbolAndClause);

  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(
        converter.mangleName(std::get<const semantics::Symbol &>(symClause)));

    // Some symbols are deferred until later in the module, these are handled
    // upon finalization of the module for OpenMP inside of Bridge, so we simply
    // skip for now.
    if (!op)
      continue;

    markDeclareTarget(
        op, converter,
        std::get<mlir::omp::DeclareTargetCaptureClause>(symClause),
        clauseOps.deviceType);
  }
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPRequiresConstruct &requiresConstruct) {
  // Requires directives are gathered and processed in semantics and
  // then combined in the lowering bridge before triggering codegen
  // just once. Hence, there is no need to lower each individual
  // occurrence here.
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPThreadprivate &threadprivate) {
  // The directive is lowered when instantiating the variable to
  // support the case of threadprivate variable declared in module.
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpMetadirectiveDirective &meta) {
  TODO(converter.getCurrentLocation(), "METADIRECTIVE");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPDeclarativeConstruct &ompDeclConstruct) {
  Fortran::common::visit(
      [&](auto &&s) { return genOMP(converter, symTable, semaCtx, eval, s); },
      ompDeclConstruct.u);
}

//===----------------------------------------------------------------------===//
// OpenMPStandaloneConstruct visitors
//===----------------------------------------------------------------------===//

static void genOMP(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    const parser::OpenMPSimpleStandaloneConstruct &simpleStandaloneConstruct) {
  const auto &directive = std::get<parser::OmpSimpleStandaloneDirective>(
      simpleStandaloneConstruct.t);
  List<Clause> clauses = makeClauses(
      std::get<parser::OmpClauseList>(simpleStandaloneConstruct.t), semaCtx);
  mlir::Location currentLocation = converter.genLocation(directive.source);

  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, directive.source, directive.v, clauses)};
  if (directive.v == llvm::omp::Directive::OMPD_ordered) {
    // Standalone "ordered" directive.
    genOrderedOp(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
  } else {
    // Dispatch handles the "block-associated" variant of "ordered".
    genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                   queue.begin());
  }
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPFlushConstruct &flushConstruct) {
  const auto &verbatim = std::get<parser::Verbatim>(flushConstruct.t);
  const auto &objectList =
      std::get<std::optional<parser::OmpObjectList>>(flushConstruct.t);
  const auto &clauseList =
      std::get<std::optional<std::list<parser::OmpMemoryOrderClause>>>(
          flushConstruct.t);
  ObjectList objects =
      objectList ? makeObjects(*objectList, semaCtx) : ObjectList{};
  List<Clause> clauses =
      clauseList ? makeList(*clauseList,
                            [&](auto &&s) { return makeClause(s.v, semaCtx); })
                 : List<Clause>{};
  mlir::Location currentLocation = converter.genLocation(verbatim.source);

  ConstructQueue queue{buildConstructQueue(
      converter.getFirOpBuilder().getModule(), semaCtx, eval, verbatim.source,
      llvm::omp::Directive::OMPD_flush, clauses)};
  genFlushOp(converter, symTable, semaCtx, eval, currentLocation, objects,
             queue, queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPCancelConstruct &cancelConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPCancellationPointConstruct
                       &cancellationPointConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPDepobjConstruct &construct) {
  // These values will be ignored until the construct itself is implemented,
  // but run them anyway for the sake of testing (via a Todo test).
  auto &ompObj = std::get<parser::OmpObject>(construct.t);
  const Object &depObj = makeObject(ompObj, semaCtx);
  Clause clause = makeClause(std::get<parser::OmpClause>(construct.t), semaCtx);
  (void)depObj;
  (void)clause;

  TODO(converter.getCurrentLocation(), "OpenMPDepobjConstruct");
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  Fortran::common::visit(
      [&](auto &&s) { return genOMP(converter, symTable, semaCtx, eval, s); },
      standaloneConstruct.u);
}

//===----------------------------------------------------------------------===//
// OpenMPConstruct visitors
//===----------------------------------------------------------------------===//

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPAllocatorsConstruct &allocsConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPAllocatorsConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPAtomicConstruct &atomicConstruct) {
  Fortran::common::visit(
      common::visitors{
          [&](const parser::OmpAtomicRead &atomicRead) {
            mlir::Location loc = converter.genLocation(atomicRead.source);
            lower::genOmpAccAtomicRead<parser::OmpAtomicRead,
                                       parser::OmpAtomicClauseList>(
                converter, atomicRead, loc);
          },
          [&](const parser::OmpAtomicWrite &atomicWrite) {
            mlir::Location loc = converter.genLocation(atomicWrite.source);
            lower::genOmpAccAtomicWrite<parser::OmpAtomicWrite,
                                        parser::OmpAtomicClauseList>(
                converter, atomicWrite, loc);
          },
          [&](const parser::OmpAtomic &atomicConstruct) {
            mlir::Location loc = converter.genLocation(atomicConstruct.source);
            lower::genOmpAtomic<parser::OmpAtomic, parser::OmpAtomicClauseList>(
                converter, atomicConstruct, loc);
          },
          [&](const parser::OmpAtomicUpdate &atomicUpdate) {
            mlir::Location loc = converter.genLocation(atomicUpdate.source);
            lower::genOmpAccAtomicUpdate<parser::OmpAtomicUpdate,
                                         parser::OmpAtomicClauseList>(
                converter, atomicUpdate, loc);
          },
          [&](const parser::OmpAtomicCapture &atomicCapture) {
            mlir::Location loc = converter.genLocation(atomicCapture.source);
            lower::genOmpAccAtomicCapture<parser::OmpAtomicCapture,
                                          parser::OmpAtomicClauseList>(
                converter, atomicCapture, loc);
          },
          [&](const parser::OmpAtomicCompare &atomicCompare) {
            mlir::Location loc = converter.genLocation(atomicCompare.source);
            TODO(loc, "OpenMP atomic compare");
          },
      },
      atomicConstruct.u);
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &endBlockDirective =
      std::get<parser::OmpEndBlockDirective>(blockConstruct.t);
  mlir::Location currentLocation =
      converter.genLocation(beginBlockDirective.source);
  const auto origDirective =
      std::get<parser::OmpBlockDirective>(beginBlockDirective.t).v;
  List<Clause> clauses = makeClauses(
      std::get<parser::OmpClauseList>(beginBlockDirective.t), semaCtx);
  clauses.append(makeClauses(
      std::get<parser::OmpClauseList>(endBlockDirective.t), semaCtx));

  assert(llvm::omp::blockConstructSet.test(origDirective) &&
         "Expected block construct");
  (void)origDirective;

  for (const Clause &clause : clauses) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (!std::holds_alternative<clause::Affinity>(clause.u) &&
        !std::holds_alternative<clause::Allocate>(clause.u) &&
        !std::holds_alternative<clause::Copyin>(clause.u) &&
        !std::holds_alternative<clause::Copyprivate>(clause.u) &&
        !std::holds_alternative<clause::Default>(clause.u) &&
        !std::holds_alternative<clause::Depend>(clause.u) &&
        !std::holds_alternative<clause::Filter>(clause.u) &&
        !std::holds_alternative<clause::Final>(clause.u) &&
        !std::holds_alternative<clause::Firstprivate>(clause.u) &&
        !std::holds_alternative<clause::HasDeviceAddr>(clause.u) &&
        !std::holds_alternative<clause::If>(clause.u) &&
        !std::holds_alternative<clause::IsDevicePtr>(clause.u) &&
        !std::holds_alternative<clause::Map>(clause.u) &&
        !std::holds_alternative<clause::Nowait>(clause.u) &&
        !std::holds_alternative<clause::NumTeams>(clause.u) &&
        !std::holds_alternative<clause::NumThreads>(clause.u) &&
        !std::holds_alternative<clause::OmpxBare>(clause.u) &&
        !std::holds_alternative<clause::Priority>(clause.u) &&
        !std::holds_alternative<clause::Private>(clause.u) &&
        !std::holds_alternative<clause::ProcBind>(clause.u) &&
        !std::holds_alternative<clause::Reduction>(clause.u) &&
        !std::holds_alternative<clause::Shared>(clause.u) &&
        !std::holds_alternative<clause::Simd>(clause.u) &&
        !std::holds_alternative<clause::ThreadLimit>(clause.u) &&
        !std::holds_alternative<clause::Threads>(clause.u) &&
        !std::holds_alternative<clause::UseDeviceAddr>(clause.u) &&
        !std::holds_alternative<clause::UseDevicePtr>(clause.u) &&
        !std::holds_alternative<clause::InReduction>(clause.u) &&
        !std::holds_alternative<clause::Mergeable>(clause.u) &&
        !std::holds_alternative<clause::Untied>(clause.u) &&
        !std::holds_alternative<clause::TaskReduction>(clause.u) &&
        !std::holds_alternative<clause::Detach>(clause.u)) {
      std::string name =
          parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(clause.id));
      TODO(clauseLocation, name + " clause is not implemented yet");
    }
  }

  llvm::omp::Directive directive =
      std::get<parser::OmpBlockDirective>(beginBlockDirective.t).v;
  const parser::CharBlock &source =
      std::get<parser::OmpBlockDirective>(beginBlockDirective.t).source;
  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, source, directive, clauses)};
  genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPCriticalConstruct &criticalConstruct) {
  const auto &cd = std::get<parser::OmpCriticalDirective>(criticalConstruct.t);
  List<Clause> clauses =
      makeClauses(std::get<parser::OmpClauseList>(cd.t), semaCtx);

  ConstructQueue queue{buildConstructQueue(
      converter.getFirOpBuilder().getModule(), semaCtx, eval, cd.source,
      llvm::omp::Directive::OMPD_critical, clauses)};

  const auto &name = std::get<std::optional<parser::Name>>(cd.t);
  mlir::Location currentLocation = converter.getCurrentLocation();
  genCriticalOp(converter, symTable, semaCtx, eval, currentLocation, queue,
                queue.begin(), name);
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPUtilityConstruct &) {
  TODO(converter.getCurrentLocation(), "OpenMPUtilityConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPDispatchConstruct &) {
  TODO(converter.getCurrentLocation(), "OpenMPDispatchConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPExecutableAllocate &execAllocConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPExecutableAllocate");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPLoopConstruct &loopConstruct) {
  const auto &beginLoopDirective =
      std::get<parser::OmpBeginLoopDirective>(loopConstruct.t);
  List<Clause> clauses = makeClauses(
      std::get<parser::OmpClauseList>(beginLoopDirective.t), semaCtx);
  if (auto &endLoopDirective =
          std::get<std::optional<parser::OmpEndLoopDirective>>(
              loopConstruct.t)) {
    clauses.append(makeClauses(
        std::get<parser::OmpClauseList>(endLoopDirective->t), semaCtx));
  }

  mlir::Location currentLocation =
      converter.genLocation(beginLoopDirective.source);

  llvm::omp::Directive directive =
      std::get<parser::OmpLoopDirective>(beginLoopDirective.t).v;
  const parser::CharBlock &source =
      std::get<parser::OmpLoopDirective>(beginLoopDirective.t).source;
  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, source, directive, clauses)};
  genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPSectionConstruct &sectionConstruct) {
  // Do nothing here. SECTION is lowered inside of the lowering for Sections
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPSectionsConstruct &sectionsConstruct) {
  const auto &beginSectionsDirective =
      std::get<parser::OmpBeginSectionsDirective>(sectionsConstruct.t);
  List<Clause> clauses = makeClauses(
      std::get<parser::OmpClauseList>(beginSectionsDirective.t), semaCtx);
  const auto &endSectionsDirective =
      std::get<parser::OmpEndSectionsDirective>(sectionsConstruct.t);
  const auto &sectionBlocks =
      std::get<parser::OmpSectionBlocks>(sectionsConstruct.t);
  clauses.append(makeClauses(
      std::get<parser::OmpClauseList>(endSectionsDirective.t), semaCtx));
  mlir::Location currentLocation = converter.getCurrentLocation();

  llvm::omp::Directive directive =
      std::get<parser::OmpSectionsDirective>(beginSectionsDirective.t).v;
  const parser::CharBlock &source =
      std::get<parser::OmpSectionsDirective>(beginSectionsDirective.t).source;
  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, source, directive, clauses)};
  ConstructQueue::iterator next = queue.begin();
  // Generate constructs that come first e.g. Parallel
  while (next != queue.end() &&
         next->id != llvm::omp::Directive::OMPD_sections) {
    genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                   next);
    next = std::next(next);
  }

  // call genSectionsOp directly (not via genOMPDispatch) so that we can add the
  // sectionBlocks argument
  assert(next != queue.end());
  assert(next->id == llvm::omp::Directive::OMPD_sections);
  genSectionsOp(converter, symTable, semaCtx, eval, currentLocation, queue,
                next, sectionBlocks);
  assert(std::next(next) == queue.end());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPConstruct &ompConstruct) {
  Fortran::common::visit(
      [&](auto &&s) { return genOMP(converter, symTable, semaCtx, eval, s); },
      ompConstruct.u);
}

//===----------------------------------------------------------------------===//
// Public functions
//===----------------------------------------------------------------------===//

mlir::Operation *Fortran::lower::genOpenMPTerminator(fir::FirOpBuilder &builder,
                                                     mlir::Operation *op,
                                                     mlir::Location loc) {
  if (mlir::isa<mlir::omp::AtomicUpdateOp, mlir::omp::DeclareReductionOp,
                mlir::omp::LoopNestOp>(op))
    return builder.create<mlir::omp::YieldOp>(loc);
  return builder.create<mlir::omp::TerminatorOp>(loc);
}

void Fortran::lower::genOpenMPConstruct(lower::AbstractConverter &converter,
                                        lower::SymMap &symTable,
                                        semantics::SemanticsContext &semaCtx,
                                        lower::pft::Evaluation &eval,
                                        const parser::OpenMPConstruct &omp) {
  lower::SymMapScope scope(symTable);
  genOMP(converter, symTable, semaCtx, eval, omp);
}

void Fortran::lower::genOpenMPDeclarativeConstruct(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    const parser::OpenMPDeclarativeConstruct &omp) {
  genOMP(converter, symTable, semaCtx, eval, omp);
  genNestedEvaluations(converter, eval);
}

void Fortran::lower::genOpenMPSymbolProperties(
    lower::AbstractConverter &converter, const lower::pft::Variable &var) {
  assert(var.hasSymbol() && "Expecting Symbol");
  const semantics::Symbol &sym = var.getSymbol();

  if (sym.test(semantics::Symbol::Flag::OmpThreadprivate))
    lower::genThreadprivateOp(converter, var);

  if (sym.test(semantics::Symbol::Flag::OmpDeclareTarget))
    lower::genDeclareTargetIntGlobal(converter, var);
}

int64_t
Fortran::lower::getCollapseValue(const parser::OmpClauseList &clauseList) {
  for (const parser::OmpClause &clause : clauseList.v) {
    if (const auto &collapseClause =
            std::get_if<parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = semantics::GetExpr(collapseClause->v);
      return evaluate::ToInt64(*expr).value();
    }
  }
  return 1;
}

void Fortran::lower::genThreadprivateOp(lower::AbstractConverter &converter,
                                        const lower::pft::Variable &var) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  const semantics::Symbol &sym = var.getSymbol();
  mlir::Value symThreadprivateValue;
  if (const semantics::Symbol *common =
          semantics::FindCommonBlockContaining(sym.GetUltimate())) {
    mlir::Value commonValue = converter.getSymbolAddress(*common);
    if (mlir::isa<mlir::omp::ThreadprivateOp>(commonValue.getDefiningOp())) {
      // Generate ThreadprivateOp for a common block instead of its members and
      // only do it once for a common block.
      return;
    }
    // Generate ThreadprivateOp and rebind the common block.
    mlir::Value commonThreadprivateValue =
        firOpBuilder.create<mlir::omp::ThreadprivateOp>(
            currentLocation, commonValue.getType(), commonValue);
    converter.bindSymbol(*common, commonThreadprivateValue);
    // Generate the threadprivate value for the common block member.
    symThreadprivateValue = genCommonBlockMember(converter, currentLocation,
                                                 sym, commonThreadprivateValue);
  } else if (!var.isGlobal()) {
    // Non-global variable which can be in threadprivate directive must be one
    // variable in main program, and it has implicit SAVE attribute. Take it as
    // with SAVE attribute, so to create GlobalOp for it to simplify the
    // translation to LLVM IR.
    // Avoids performing multiple globalInitializations.
    fir::GlobalOp global;
    auto module = converter.getModuleOp();
    std::string globalName = converter.mangleName(sym);
    if (module.lookupSymbol<fir::GlobalOp>(globalName))
      global = module.lookupSymbol<fir::GlobalOp>(globalName);
    else
      global = globalInitialization(converter, firOpBuilder, sym, var,
                                    currentLocation);

    mlir::Value symValue = firOpBuilder.create<fir::AddrOfOp>(
        currentLocation, global.resultType(), global.getSymbol());
    symThreadprivateValue = firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  } else {
    mlir::Value symValue = converter.getSymbolAddress(sym);

    // The symbol may be use-associated multiple times, and nothing needs to be
    // done after the original symbol is mapped to the threadprivatized value
    // for the first time. Use the threadprivatized value directly.
    mlir::Operation *op;
    if (auto declOp = symValue.getDefiningOp<hlfir::DeclareOp>())
      op = declOp.getMemref().getDefiningOp();
    else
      op = symValue.getDefiningOp();
    if (mlir::isa<mlir::omp::ThreadprivateOp>(op))
      return;

    symThreadprivateValue = firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  }

  fir::ExtendedValue sexv = converter.getSymbolExtendedValue(sym);
  fir::ExtendedValue symThreadprivateExv =
      getExtendedValue(sexv, symThreadprivateValue);
  converter.bindSymbol(sym, symThreadprivateExv);
}

// This function replicates threadprivate's behaviour of generating
// an internal fir.GlobalOp for non-global variables in the main program
// that have the implicit SAVE attribute, to simplifiy LLVM-IR and MLIR
// generation.
void Fortran::lower::genDeclareTargetIntGlobal(
    lower::AbstractConverter &converter, const lower::pft::Variable &var) {
  if (!var.isGlobal()) {
    // A non-global variable which can be in a declare target directive must
    // be a variable in the main program, and it has the implicit SAVE
    // attribute. We create a GlobalOp for it to simplify the translation to
    // LLVM IR.
    globalInitialization(converter, converter.getFirOpBuilder(),
                         var.getSymbol(), var, converter.getCurrentLocation());
  }
}

bool Fortran::lower::isOpenMPTargetConstruct(
    const parser::OpenMPConstruct &omp) {
  llvm::omp::Directive dir = llvm::omp::Directive::OMPD_unknown;
  if (const auto *block = std::get_if<parser::OpenMPBlockConstruct>(&omp.u)) {
    const auto &begin = std::get<parser::OmpBeginBlockDirective>(block->t);
    dir = std::get<parser::OmpBlockDirective>(begin.t).v;
  } else if (const auto *loop =
                 std::get_if<parser::OpenMPLoopConstruct>(&omp.u)) {
    const auto &begin = std::get<parser::OmpBeginLoopDirective>(loop->t);
    dir = std::get<parser::OmpLoopDirective>(begin.t).v;
  }
  return llvm::omp::allTargetSet.test(dir);
}

void Fortran::lower::gatherOpenMPDeferredDeclareTargets(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclarativeConstruct &ompDecl,
    llvm::SmallVectorImpl<OMPDeferredDeclareTargetInfo>
        &deferredDeclareTarget) {
  Fortran::common::visit(
      common::visitors{
          [&](const parser::OpenMPDeclareTargetConstruct &ompReq) {
            collectDeferredDeclareTargets(converter, semaCtx, eval, ompReq,
                                          deferredDeclareTarget);
          },
          [&](const auto &) {},
      },
      ompDecl.u);
}

bool Fortran::lower::isOpenMPDeviceDeclareTarget(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclarativeConstruct &ompDecl) {
  return Fortran::common::visit(
      common::visitors{
          [&](const parser::OpenMPDeclareTargetConstruct &ompReq) {
            mlir::omp::DeclareTargetDeviceType targetType =
                getDeclareTargetFunctionDevice(converter, semaCtx, eval, ompReq)
                    .value_or(mlir::omp::DeclareTargetDeviceType::host);
            return targetType != mlir::omp::DeclareTargetDeviceType::host;
          },
          [&](const auto &) { return false; },
      },
      ompDecl.u);
}

// In certain cases such as subroutine or function interfaces which declare
// but do not define or directly call the subroutine or function in the same
// module, their lowering is delayed until after the declare target construct
// itself is processed, so there symbol is not within the table.
//
// This function will also return true if we encounter any device declare
// target cases, to satisfy checking if we require the requires attributes
// on the module.
bool Fortran::lower::markOpenMPDeferredDeclareTargetFunctions(
    mlir::Operation *mod,
    llvm::SmallVectorImpl<OMPDeferredDeclareTargetInfo> &deferredDeclareTargets,
    AbstractConverter &converter) {
  bool deviceCodeFound = false;
  auto modOp = llvm::cast<mlir::ModuleOp>(mod);
  for (auto declTar : deferredDeclareTargets) {
    mlir::Operation *op = modOp.lookupSymbol(converter.mangleName(declTar.sym));

    // Due to interfaces being optionally emitted on usage in a module,
    // not finding an operation at this point cannot be a hard error, we
    // simply ignore it for now.
    // TODO: Add semantic checks for detecting cases where an erronous
    // (undefined) symbol has been supplied to a declare target clause
    if (!op)
      continue;

    auto devType = declTar.declareTargetDeviceType;
    if (!deviceCodeFound && devType != mlir::omp::DeclareTargetDeviceType::host)
      deviceCodeFound = true;

    markDeclareTarget(op, converter, declTar.declareTargetCaptureClause,
                      devType);
  }

  return deviceCodeFound;
}

void Fortran::lower::genOpenMPRequires(mlir::Operation *mod,
                                       const semantics::Symbol *symbol) {
  using MlirRequires = mlir::omp::ClauseRequires;
  using SemaRequires = semantics::WithOmpDeclarative::RequiresFlag;

  if (auto offloadMod =
          llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(mod)) {
    semantics::WithOmpDeclarative::RequiresFlags semaFlags;
    if (symbol) {
      common::visit(
          [&](const auto &details) {
            if constexpr (std::is_base_of_v<semantics::WithOmpDeclarative,
                                            std::decay_t<decltype(details)>>) {
              if (details.has_ompRequires())
                semaFlags = *details.ompRequires();
            }
          },
          symbol->details());
    }

    // Use pre-populated omp.requires module attribute if it was set, so that
    // the "-fopenmp-force-usm" compiler option is honored.
    MlirRequires mlirFlags = offloadMod.getRequires();
    if (semaFlags.test(SemaRequires::ReverseOffload))
      mlirFlags = mlirFlags | MlirRequires::reverse_offload;
    if (semaFlags.test(SemaRequires::UnifiedAddress))
      mlirFlags = mlirFlags | MlirRequires::unified_address;
    if (semaFlags.test(SemaRequires::UnifiedSharedMemory))
      mlirFlags = mlirFlags | MlirRequires::unified_shared_memory;
    if (semaFlags.test(SemaRequires::DynamicAllocators))
      mlirFlags = mlirFlags | MlirRequires::dynamic_allocators;

    offloadMod.setRequires(mlirFlags);
  }
}
