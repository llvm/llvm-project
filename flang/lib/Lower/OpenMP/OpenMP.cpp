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

#include "Atomic.h"
#include "ClauseProcessor.h"
#include "DataSharingProcessor.h"
#include "Decomposer.h"
#include "Utils.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/DirectivesCommon.h"
#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Utils.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/tools.h"
#include "flang/Support/Flags.h"
#include "flang/Support/OpenMP-utils.h"
#include "flang/Utils/OpenMP.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Support/StateStack.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace Fortran::lower::omp;
using namespace Fortran::common::openmp;
using namespace Fortran::utils::openmp;

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
class HostEvalInfoStackFrame
    : public mlir::StateStackFrameBase<HostEvalInfoStackFrame> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HostEvalInfoStackFrame)

  HostEvalInfo info;
};

static HostEvalInfo *
getHostEvalInfoStackTop(lower::AbstractConverter &converter) {
  HostEvalInfoStackFrame *frame =
      converter.getStateStack().getStackTop<HostEvalInfoStackFrame>();
  return frame ? &frame->info : nullptr;
}

/// Stack frame for storing the OpenMPSectionsConstruct currently being
/// processed so that it can be referred to when lowering the construct.
class SectionsConstructStackFrame
    : public mlir::StateStackFrameBase<SectionsConstructStackFrame> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SectionsConstructStackFrame)

  explicit SectionsConstructStackFrame(
      const parser::OpenMPSectionsConstruct &sectionsConstruct)
      : sectionsConstruct{sectionsConstruct} {}

  const parser::OpenMPSectionsConstruct &sectionsConstruct;
};

static const parser::OpenMPSectionsConstruct *
getSectionsConstructStackTop(lower::AbstractConverter &converter) {
  SectionsConstructStackFrame *frame =
      converter.getStateStack().getStackTop<SectionsConstructStackFrame>();
  return frame ? &frame->sectionsConstruct : nullptr;
}

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

  auto bindSingleMapLike = [&converter](const semantics::Symbol &sym,
                                        const mlir::BlockArgument &arg) {
    fir::ExtendedValue extVal = converter.getSymbolExtendedValue(sym);
    auto refType = mlir::dyn_cast<fir::ReferenceType>(arg.getType());
    if (refType && fir::isa_builtin_cptr_type(refType.getElementType())) {
      converter.bindSymbol(sym, arg);
    } else {
      extVal.match(
          [&](const fir::BoxValue &v) {
            converter.bindSymbol(sym, fir::BoxValue(arg, v.getLBounds(),
                                                    v.getExplicitParameters(),
                                                    v.getExplicitExtents()));
          },
          [&](const fir::MutableBoxValue &v) {
            converter.bindSymbol(
                sym, fir::MutableBoxValue(arg, v.getLBounds(),
                                          v.getMutableProperties()));
          },
          [&](const fir::ArrayBoxValue &v) {
            converter.bindSymbol(sym, fir::ArrayBoxValue(arg, v.getExtents(),
                                                         v.getLBounds(),
                                                         v.getSourceBox()));
          },
          [&](const fir::CharArrayBoxValue &v) {
            converter.bindSymbol(sym, fir::CharArrayBoxValue(arg, v.getLen(),
                                                             v.getExtents(),
                                                             v.getLBounds()));
          },
          [&](const fir::CharBoxValue &v) {
            converter.bindSymbol(sym, fir::CharBoxValue(arg, v.getLen()));
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
  bindMapLike(args.hasDeviceAddr.syms, op.getHasDeviceAddrBlockArgs());
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
            [&](const parser::OmpBlockConstruct &ompConstruct) {
              beginClauseList = &ompConstruct.BeginDir().Clauses();
              if (auto &endSpec = ompConstruct.EndDir())
                endClauseList = &endSpec->Clauses();
            },
            [&](const parser::OpenMPLoopConstruct &ompConstruct) {
              const auto &beginDirective =
                  std::get<parser::OmpBeginLoopDirective>(ompConstruct.t);
              beginClauseList =
                  &std::get<parser::OmpClauseList>(beginDirective.t);

              if (auto &endDirective =
                      std::get<std::optional<parser::OmpEndLoopDirective>>(
                          ompConstruct.t)) {
                endClauseList =
                    &std::get<parser::OmpClauseList>(endDirective->t);
              }
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
      dir = parser::omp::GetOmpDirectiveName(*ompEval).v;
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

    HostEvalInfo *hostInfo = getHostEvalInfoStackTop(converter);
    assert(hostInfo && "expected HOST_EVAL info structure");

    switch (parser::omp::GetOmpDirectiveName(*ompEval).v) {
    case OMPD_teams_distribute_parallel_do:
    case OMPD_teams_distribute_parallel_do_simd:
      cp.processThreadLimit(stmtCtx, hostInfo->ops);
      [[fallthrough]];
    case OMPD_target_teams_distribute_parallel_do:
    case OMPD_target_teams_distribute_parallel_do_simd:
      cp.processNumTeams(stmtCtx, hostInfo->ops);
      [[fallthrough]];
    case OMPD_distribute_parallel_do:
    case OMPD_distribute_parallel_do_simd:
      cp.processNumThreads(stmtCtx, hostInfo->ops);
      [[fallthrough]];
    case OMPD_distribute:
    case OMPD_distribute_simd:
      cp.processCollapse(loc, eval, hostInfo->ops, hostInfo->ops, hostInfo->iv);
      break;

    case OMPD_teams:
      cp.processThreadLimit(stmtCtx, hostInfo->ops);
      [[fallthrough]];
    case OMPD_target_teams:
      cp.processNumTeams(stmtCtx, hostInfo->ops);
      processSingleNestedIf([](Directive nestedDir) {
        return topDistributeSet.test(nestedDir) || topLoopSet.test(nestedDir);
      });
      break;

    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
      cp.processThreadLimit(stmtCtx, hostInfo->ops);
      [[fallthrough]];
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
      cp.processCollapse(loc, eval, hostInfo->ops, hostInfo->ops, hostInfo->iv);
      cp.processNumTeams(stmtCtx, hostInfo->ops);
      break;

    case OMPD_teams_loop:
      cp.processThreadLimit(stmtCtx, hostInfo->ops);
      [[fallthrough]];
    case OMPD_target_teams_loop:
      cp.processNumTeams(stmtCtx, hostInfo->ops);
      [[fallthrough]];
    case OMPD_loop:
      cp.processCollapse(loc, eval, hostInfo->ops, hostInfo->ops, hostInfo->iv);
      break;

    case OMPD_teams_workdistribute:
      cp.processThreadLimit(stmtCtx, hostInfo->ops);
      [[fallthrough]];
    case OMPD_target_teams_workdistribute:
      cp.processNumTeams(stmtCtx, hostInfo->ops);
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

  const auto *ompEval = eval.getIf<parser::OpenMPConstruct>();
  assert(ompEval &&
         llvm::omp::allTargetSet.test(
             parser::omp::GetOmpDirectiveName(*ompEval).v) &&
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
  std::string globalName = converter.mangleName(sym);
  mlir::StringAttr linkage = firOpBuilder.createInternalLinkage();
  return Fortran::lower::defineGlobal(converter, var, globalName, linkage);
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
    return mlir::omp::ThreadprivateOp::create(firOpBuilder, currentLocation,
                                              symValue.getType(), symValue);
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
          converter, currentLocation, sym->GetUltimate(),
          commonThreadprivateValue, common->size());
    } else {
      symThreadprivateValue = genThreadprivateOp(*sym);
    }

    fir::ExtendedValue sexv = converter.getSymbolExtendedValue(*sym);
    fir::ExtendedValue symThreadprivateExv =
        getExtendedValue(sexv, symThreadprivateValue);
    converter.bindSymbol(*sym, symThreadprivateExv);
  }
}

static mlir::Operation *setLoopVar(lower::AbstractConverter &converter,
                                   mlir::Location loc, mlir::Value indexVal,
                                   const semantics::Symbol *sym) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());
  mlir::Type tempTy = converter.genType(*sym);
  firOpBuilder.restoreInsertionPoint(insPt);

  mlir::Value cvtVal = firOpBuilder.createConvert(loc, tempTy, indexVal);
  hlfir::Entity lhs{converter.getSymbolAddress(*sym)};

  lhs = hlfir::derefPointersAndAllocatables(loc, firOpBuilder, lhs);

  mlir::Operation *storeOp =
      hlfir::AssignOp::create(firOpBuilder, loc, cvtVal, lhs);
  return storeOp;
}

static mlir::Operation *
createAndSetPrivatizedLoopVar(lower::AbstractConverter &converter,
                              mlir::Location loc, mlir::Value indexVal,
                              const semantics::Symbol *sym) {
  assert(converter.isPresentShallowLookup(*sym) &&
         "Expected symbol to be in symbol table.");
  return setLoopVar(converter, loc, indexVal, sym);
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
    llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &symbolAndClause) {
  const auto &spec =
      std::get<parser::OmpDeclareTargetSpecifier>(declareTargetConstruct.t);
  if (const auto *objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
    ObjectList objects{makeObjects(*objectList, semaCtx)};
    // Case: declare target(func, var1, var2)
    gatherFuncAndVarSyms(objects, mlir::omp::DeclareTargetCaptureClause::to,
                         symbolAndClause, /*automap=*/false);
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
  llvm::SmallVector<DeclareTargetCaptureInfo> symbolAndClause;
  getDeclareTargetInfo(converter, semaCtx, eval, declareTargetConstruct,
                       clauseOps, symbolAndClause);
  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();

  for (const DeclareTargetCaptureInfo &symClause : symbolAndClause) {
    mlir::Operation *op =
        mod.lookupSymbol(converter.mangleName(symClause.symbol));

    if (!op) {
      deferredDeclareTarget.push_back({symClause.clause, clauseOps.deviceType,
                                       symClause.automap, symClause.symbol});
    }
  }
}

static std::optional<mlir::omp::DeclareTargetDeviceType>
getDeclareTargetFunctionDevice(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct) {
  mlir::omp::DeclareTargetOperands clauseOps;
  llvm::SmallVector<DeclareTargetCaptureInfo> symbolAndClause;
  getDeclareTargetInfo(converter, semaCtx, eval, declareTargetConstruct,
                       clauseOps, symbolAndClause);

  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  for (const DeclareTargetCaptureInfo &symClause : symbolAndClause) {
    mlir::Operation *op =
        mod.lookupSymbol(converter.mangleName(symClause.symbol));

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

static clause::Defaultmap::ImplicitBehavior
getDefaultmapIfPresent(const DefaultMapsTy &defaultMaps, mlir::Type varType) {
  using DefMap = clause::Defaultmap;

  if (defaultMaps.empty())
    return DefMap::ImplicitBehavior::Default;

  if (llvm::is_contained(defaultMaps, DefMap::VariableCategory::All))
    return defaultMaps.at(DefMap::VariableCategory::All);

  // NOTE: Unsure if complex and/or vector falls into a scalar type
  // or aggregate, but the current default implicit behaviour is to
  // treat them as such (c_ptr has its own behaviour, so perhaps
  // being lumped in as a scalar isn't the right thing).
  if ((fir::isa_trivial(varType) || fir::isa_char(varType) ||
       fir::isa_builtin_cptr_type(varType)) &&
      llvm::is_contained(defaultMaps, DefMap::VariableCategory::Scalar))
    return defaultMaps.at(DefMap::VariableCategory::Scalar);

  if (fir::isPointerType(varType) &&
      llvm::is_contained(defaultMaps, DefMap::VariableCategory::Pointer))
    return defaultMaps.at(DefMap::VariableCategory::Pointer);

  if (fir::isAllocatableType(varType) &&
      llvm::is_contained(defaultMaps, DefMap::VariableCategory::Allocatable))
    return defaultMaps.at(DefMap::VariableCategory::Allocatable);

  if (fir::isa_aggregate(varType) &&
      llvm::is_contained(defaultMaps, DefMap::VariableCategory::Aggregate))
    return defaultMaps.at(DefMap::VariableCategory::Aggregate);

  return DefMap::ImplicitBehavior::Default;
}

static std::pair<llvm::omp::OpenMPOffloadMappingFlags,
                 mlir::omp::VariableCaptureKind>
getImplicitMapTypeAndKind(fir::FirOpBuilder &firOpBuilder,
                          lower::AbstractConverter &converter,
                          const DefaultMapsTy &defaultMaps, mlir::Type varType,
                          mlir::Location loc, const semantics::Symbol &sym) {
  using DefMap = clause::Defaultmap;
  // Check if a value of type `type` can be passed to the kernel by value.
  // All kernel parameters are of pointer type, so if the value can be
  // represented inside of a pointer, then it can be passed by value.
  auto isLiteralType = [&](mlir::Type type) {
    const mlir::DataLayout &dl = firOpBuilder.getDataLayout();
    mlir::Type ptrTy =
        mlir::LLVM::LLVMPointerType::get(&converter.getMLIRContext());
    uint64_t ptrSize = dl.getTypeSize(ptrTy);
    uint64_t ptrAlign = dl.getTypePreferredAlignment(ptrTy);

    auto [size, align] = fir::getTypeSizeAndAlignmentOrCrash(
        loc, type, dl, converter.getKindMap());
    return size <= ptrSize && align <= ptrAlign;
  };

  llvm::omp::OpenMPOffloadMappingFlags mapFlag =
      llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;

  auto implicitBehaviour = getDefaultmapIfPresent(defaultMaps, varType);
  if (implicitBehaviour == DefMap::ImplicitBehavior::Default) {
    mlir::omp::VariableCaptureKind captureKind =
        mlir::omp::VariableCaptureKind::ByRef;

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
    } else if (fir::isa_trivial(varType) || fir::isa_char(varType)) {
      // Scalars behave as if they were "firstprivate".
      // TODO: Handle objects that are shared/lastprivate or were listed
      // in an in_reduction clause.
      if (isLiteralType(varType)) {
        captureKind = mlir::omp::VariableCaptureKind::ByCopy;
      } else {
        mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
      }
    } else if (!fir::isa_builtin_cptr_type(varType)) {
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
    }
    return std::make_pair(mapFlag, captureKind);
  }

  switch (implicitBehaviour) {
  case DefMap::ImplicitBehavior::Alloc:
    return std::make_pair(llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE,
                          mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::Firstprivate:
  case DefMap::ImplicitBehavior::None:
    TODO(loc, "Firstprivate and None are currently unsupported defaultmap "
              "behaviour");
    break;
  case DefMap::ImplicitBehavior::From:
    return std::make_pair(mapFlag |=
                          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM,
                          mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::Present:
    return std::make_pair(mapFlag |=
                          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PRESENT,
                          mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::To:
    return std::make_pair(mapFlag |=
                          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO,
                          (fir::isa_trivial(varType) || fir::isa_char(varType))
                              ? mlir::omp::VariableCaptureKind::ByCopy
                              : mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::Tofrom:
    return std::make_pair(mapFlag |=
                          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM |
                          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO,
                          mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::Default:
    llvm_unreachable(
        "Implicit None Behaviour Should Have Been Handled Earlier");
    break;
  }

  return std::make_pair(mapFlag |=
                        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM |
                        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO,
                        mlir::omp::VariableCaptureKind::ByRef);
}

static void
markDeclareTarget(mlir::Operation *op, lower::AbstractConverter &converter,
                  mlir::omp::DeclareTargetCaptureClause captureClause,
                  mlir::omp::DeclareTargetDeviceType deviceType, bool automap) {
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
                                       captureClause, automap);
    return;
  }

  declareTargetOp.setDeclareTarget(deviceType, captureClause, automap);
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

  OpWithBodyGenInfo &setEntryBlockArgs(const EntryBlockArgs *value) {
    blockArgs = value;
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

  OpWithBodyGenInfo &setPrivatize(bool value) {
    privatize = value;
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
  /// [in] if provided, it is used to create the op's region entry block. It is
  /// overriden when a \see genRegionEntryCB is provided. This is only valid for
  /// operations implementing the \see mlir::omp::BlockArgOpenMPOpInterface.
  const EntryBlockArgs *blockArgs = nullptr;
  /// [in] if provided, it overrides the default op's region entry block
  /// creation.
  GenOMPRegionEntryCBFn genRegionEntryCB = nullptr;
  /// [in] if set to `true`, skip generating nested evaluations and dispatching
  /// any further leaf constructs.
  bool genSkeletonOnly = false;
  /// [in] enables handling of privatized variable unless set to `false`.
  bool privatize = true;
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
    mlir::Value undef = fir::UndefOp::create(builder, builder.getUnknownLoc(),
                                             builder.getIndexType());
    return undef.getDefiningOp();
  };

  // Create the entry block for the region and collect its arguments for use
  // within the region. The entry block will be created as follows:
  //   - By default, it will be empty and have no arguments.
  //   - Operations implementing the omp::BlockArgOpenMPOpInterface can set the
  //     `info.blockArgs` pointer so that block arguments will be those
  //     corresponding to entry block argument-generating clauses. Binding of
  //     Fortran symbols to the new MLIR values is done automatically.
  //   - If the `info.genRegionEntryCB` callback is set, it takes precedence and
  //     allows callers to manually create the entry block with its intended
  //     list of arguments and to bind these arguments to their corresponding
  //     Fortran symbols. This is used for e.g. loop induction variables.
  auto regionArgs = [&]() -> llvm::SmallVector<const semantics::Symbol *> {
    if (info.genRegionEntryCB)
      return info.genRegionEntryCB(&op);

    if (info.blockArgs) {
      genEntryBlock(firOpBuilder, *info.blockArgs, op.getRegion(0));
      bindEntryBlockArgs(info.converter,
                         llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(op),
                         *info.blockArgs);
      return llvm::to_vector(info.blockArgs->getSyms());
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
  bool privatize = info.clauses && info.privatize;

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
      mlir::cf::BranchOp::create(firOpBuilder, info.loc, exit);
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
      // nested loop as a unit, so we need to pass the bottom level wrapper (if
      // present). Otherwise, these operations will be inserted within a
      // wrapper region.
      mlir::Operation *privatizationBottomLevelOp = &op;
      if (auto loopNest = llvm::dyn_cast<mlir::omp::LoopNestOp>(op)) {
        llvm::SmallVector<mlir::omp::LoopWrapperInterface> wrappers;
        loopNest.gatherWrappers(wrappers);
        if (!wrappers.empty())
          privatizationBottomLevelOp = &*wrappers.front();
      }

      if (!info.dsp) {
        assert(tempDsp.has_value());
        tempDsp->processStep2(privatizationBottomLevelOp, isLoop);
      } else {
        if (isLoop && regionArgs.size() > 0) {
          for (const auto &regionArg : regionArgs) {
            info.dsp->pushLoopIV(info.converter.getSymbolAddress(*regionArg));
          }
        }
        info.dsp->processStep2(privatizationBottomLevelOp, isLoop);
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
  mlir::Value undefMarker = fir::UndefOp::create(firOpBuilder, dataOp.getLoc(),
                                                 firOpBuilder.getIndexType());

  // Create blocks for unstructured regions. This has to be done since
  // blocks are initially allocated with the function as the parent region.
  if (eval.lowerAsUnstructured()) {
    lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp, mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  mlir::omp::TerminatorOp::create(firOpBuilder, currentLocation);

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
          converter, currentLocation, *obj, mapArg, mapSym->size());
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
  genEntryBlock(firOpBuilder, args, region);
  bindEntryBlockArgs(converter, targetOp, args);
  if (HostEvalInfo *hostEvalInfo = getHostEvalInfoStackTop(converter))
    hostEvalInfo->bindOperands(argIface.getHostEvalBlockArgs());

  // Check if cloning the bounds introduced any dependency on the outer region.
  // If so, then either clone them as well if they are MemoryEffectFree, or else
  // copy them to a new temporary and add them to the map and block_argument
  // lists and replace their uses with the new temporary.
  cloneOrMapRegionOutsiders(firOpBuilder, targetOp);

  // Insert dummy instruction to remember the insertion position. The
  // marker will be deleted since there are not uses.
  // In the HLFIR flow there are hlfir.declares inserted above while
  // setting block arguments.
  mlir::Value undefMarker = fir::UndefOp::create(
      firOpBuilder, targetOp.getLoc(), firOpBuilder.getIndexType());

  // Create blocks for unstructured regions. This has to be done since
  // blocks are initially allocated with the function as the parent region.
  if (lower::omp::isLastItemInQueue(item, queue) &&
      eval.lowerAsUnstructured()) {
    lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp, mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  mlir::omp::TerminatorOp::create(firOpBuilder, currentLocation);

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
  auto op = OpTy::create(info.converter.getFirOpBuilder(), info.loc,
                         std::forward<Args>(args)...);
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
  auto op = OpTy::create(firOpBuilder, loc, clauseOps);

  // Create entry block with arguments.
  genEntryBlock(firOpBuilder, args, op.getRegion());

  return op;
}

//===----------------------------------------------------------------------===//
// Code generation functions for clauses
//===----------------------------------------------------------------------===//

static void genCancelClauses(lower::AbstractConverter &converter,
                             semantics::SemanticsContext &semaCtx,
                             const List<Clause> &clauses, mlir::Location loc,
                             mlir::omp::CancelOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processCancelDirectiveName(clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_cancel, clauseOps);
}

static void
genCancellationPointClauses(lower::AbstractConverter &converter,
                            semantics::SemanticsContext &semaCtx,
                            const List<Clause> &clauses, mlir::Location loc,
                            mlir::omp::CancellationPointOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processCancelDirectiveName(clauseOps);
}

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

  HostEvalInfo *hostEvalInfo = getHostEvalInfoStackTop(converter);
  if (!hostEvalInfo || !hostEvalInfo->apply(clauseOps, iv))
    cp.processCollapse(loc, eval, clauseOps, clauseOps, iv);

  clauseOps.loopInclusive = converter.getFirOpBuilder().getUnitAttr();
  cp.processTileSizes(eval, clauseOps);
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

  HostEvalInfo *hostEvalInfo = getHostEvalInfoStackTop(converter);
  if (!hostEvalInfo || !hostEvalInfo->apply(clauseOps))
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
    lower::SymMap &symTable, lower::StatementContext &stmtCtx,
    lower::pft::Evaluation &eval, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::TargetOperands &clauseOps,
    DefaultMapsTy &defaultMaps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &hasDeviceAddrSyms,
    llvm::SmallVectorImpl<const semantics::Symbol *> &isDevicePtrSyms,
    llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processBare(clauseOps);
  cp.processDefaultMap(stmtCtx, defaultMaps);
  cp.processDepend(symTable, stmtCtx, clauseOps);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processHasDeviceAddr(stmtCtx, clauseOps, hasDeviceAddrSyms);
  if (HostEvalInfo *hostEvalInfo = getHostEvalInfoStackTop(converter)) {
    // Only process host_eval if compiling for the host device.
    processHostEvalClauses(converter, semaCtx, stmtCtx, eval, loc);
    hostEvalInfo->collectValues(clauseOps.hostEvalVars);
  }
  cp.processIf(llvm::omp::Directive::OMPD_target, clauseOps);
  cp.processIsDevicePtr(clauseOps, isDevicePtrSyms);
  cp.processMap(loc, stmtCtx, clauseOps, llvm::omp::Directive::OMPD_unknown,
                &mapSyms);
  cp.processNowait(clauseOps);
  cp.processThreadLimit(stmtCtx, clauseOps);

  cp.processTODO<clause::Allocate, clause::InReduction, clause::UsesAllocators>(
      loc, llvm::omp::Directive::OMPD_target);

  // `target private(..)` is only supported in delayed privatization mode.
  if (!enableDelayedPrivatizationStaging)
    cp.processTODO<clause::Firstprivate, clause::Private>(
        loc, llvm::omp::Directive::OMPD_target);
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
    lower::SymMap &symTable, lower::StatementContext &stmtCtx,
    const List<Clause> &clauses, mlir::Location loc,
    llvm::omp::Directive directive,
    mlir::omp::TargetEnterExitUpdateDataOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDepend(symTable, stmtCtx, clauseOps);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processIf(directive, clauseOps);

  if (directive == llvm::omp::Directive::OMPD_target_update)
    cp.processMotionClauses(stmtCtx, clauseOps);
  else
    cp.processMap(loc, stmtCtx, clauseOps, directive);

  cp.processNowait(clauseOps);
}

static void genTaskClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::SymMap &symTable, lower::StatementContext &stmtCtx,
    const List<Clause> &clauses, mlir::Location loc,
    mlir::omp::TaskOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &inReductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processDepend(symTable, stmtCtx, clauseOps);
  cp.processFinal(stmtCtx, clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_task, clauseOps);
  cp.processInReduction(loc, clauseOps, inReductionSyms);
  cp.processMergeable(clauseOps);
  cp.processPriority(stmtCtx, clauseOps);
  cp.processUntied(clauseOps);
  cp.processDetach(clauseOps);

  cp.processTODO<clause::Affinity>(loc, llvm::omp::Directive::OMPD_task);
}

static void genTaskgroupClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    const List<Clause> &clauses, mlir::Location loc,
    mlir::omp::TaskgroupOperands &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &taskReductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processTaskReduction(loc, clauseOps, taskReductionSyms);
}

static void genTaskloopClauses(lower::AbstractConverter &converter,
                               semantics::SemanticsContext &semaCtx,
                               lower::StatementContext &stmtCtx,
                               const List<Clause> &clauses, mlir::Location loc,
                               mlir::omp::TaskloopOperands &clauseOps) {

  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processGrainsize(stmtCtx, clauseOps);
  cp.processNumTasks(stmtCtx, clauseOps);

  cp.processTODO<clause::Allocate, clause::Collapse, clause::Default,
                 clause::Final, clause::If, clause::InReduction,
                 clause::Lastprivate, clause::Mergeable, clause::Nogroup,
                 clause::Priority, clause::Reduction, clause::Shared,
                 clause::Untied>(loc, llvm::omp::Directive::OMPD_taskloop);
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

  HostEvalInfo *hostEvalInfo = getHostEvalInfoStackTop(converter);
  if (!hostEvalInfo || !hostEvalInfo->apply(clauseOps)) {
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
  return mlir::omp::BarrierOp::create(converter.getFirOpBuilder(), loc);
}

static mlir::omp::CancelOp genCancelOp(lower::AbstractConverter &converter,
                                       semantics::SemanticsContext &semaCtx,
                                       lower::pft::Evaluation &eval,
                                       mlir::Location loc,
                                       const ConstructQueue &queue,
                                       ConstructQueue::const_iterator item) {
  mlir::omp::CancelOperands clauseOps;
  genCancelClauses(converter, semaCtx, item->clauses, loc, clauseOps);

  return mlir::omp::CancelOp::create(converter.getFirOpBuilder(), loc,
                                     clauseOps);
}

static mlir::omp::CancellationPointOp genCancellationPointOp(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  mlir::omp::CancellationPointOperands clauseOps;
  genCancellationPointClauses(converter, semaCtx, item->clauses, loc,
                              clauseOps);

  return mlir::omp::CancellationPointOp::create(converter.getFirOpBuilder(),
                                                loc, clauseOps);
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
      global = mlir::omp::CriticalDeclareOp::create(modBuilder, loc, clauseOps);
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

  return mlir::omp::FlushOp::create(converter.getFirOpBuilder(),
                                    converter.getCurrentLocation(),
                                    operandRange);
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

  uint64_t nestValue = getCollapseValue(item->clauses);
  nestValue = nestValue < iv.size() ? iv.size() : nestValue;
  auto *nestedEval = getCollapsedLoopEval(eval, nestValue);
  return genOpWithBody<mlir::omp::LoopNestOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, *nestedEval,
                        directive)
          .setClauses(&item->clauses)
          .setDataSharingProcessor(&dsp)
          .setGenRegionEntryCb(ivCallback),
      queue, item, clauseOps);
}

static mlir::omp::LoopOp
genLoopOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
          semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
          mlir::Location loc, const ConstructQueue &queue,
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
  return loopOp;
}

static mlir::omp::CanonicalLoopOp
genCanonicalLoopOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval, mlir::Location loc,
                   const ConstructQueue &queue,
                   ConstructQueue::const_iterator item,
                   llvm::ArrayRef<const semantics::Symbol *> ivs,
                   llvm::omp::Directive directive) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  assert(ivs.size() == 1 && "Nested loops not yet implemented");
  const semantics::Symbol *iv = ivs[0];

  auto &nestedEval = eval.getFirstNestedEvaluation();
  if (nestedEval.getIf<parser::DoConstruct>()->IsDoConcurrent()) {
    // OpenMP specifies DO CONCURRENT only with the `!omp loop` construct. Will
    // need to add special cases for this combination.
    TODO(loc, "DO CONCURRENT as canonical loop not supported");
  }

  // Get the loop bounds (and increment)
  auto &doLoopEval = nestedEval.getFirstNestedEvaluation();
  auto *doStmt = doLoopEval.getIf<parser::NonLabelDoStmt>();
  assert(doStmt && "Expected do loop to be in the nested evaluation");
  auto &loopControl = std::get<std::optional<parser::LoopControl>>(doStmt->t);
  assert(loopControl.has_value());
  auto *bounds = std::get_if<parser::LoopControl::Bounds>(&loopControl->u);
  assert(bounds && "Expected bounds for canonical loop");
  lower::StatementContext stmtCtx;
  mlir::Value loopLBVar = fir::getBase(
      converter.genExprValue(*semantics::GetExpr(bounds->lower), stmtCtx));
  mlir::Value loopUBVar = fir::getBase(
      converter.genExprValue(*semantics::GetExpr(bounds->upper), stmtCtx));
  mlir::Value loopStepVar = [&]() {
    if (bounds->step) {
      return fir::getBase(
          converter.genExprValue(*semantics::GetExpr(bounds->step), stmtCtx));
    }

    // If `step` is not present, assume it is `1`.
    return firOpBuilder.createIntegerConstant(loc, firOpBuilder.getI32Type(),
                                              1);
  }();

  // Get the integer kind for the loop variable and cast the loop bounds
  size_t loopVarTypeSize = bounds->name.thing.symbol->GetUltimate().size();
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  loopLBVar = firOpBuilder.createConvert(loc, loopVarType, loopLBVar);
  loopUBVar = firOpBuilder.createConvert(loc, loopVarType, loopUBVar);
  loopStepVar = firOpBuilder.createConvert(loc, loopVarType, loopStepVar);

  // Start lowering
  mlir::Value zero = firOpBuilder.createIntegerConstant(loc, loopVarType, 0);
  mlir::Value one = firOpBuilder.createIntegerConstant(loc, loopVarType, 1);
  mlir::Value isDownwards = mlir::arith::CmpIOp::create(
      firOpBuilder, loc, mlir::arith::CmpIPredicate::slt, loopStepVar, zero);

  // Ensure we are counting upwards. If not, negate step and swap lb and ub.
  mlir::Value negStep =
      mlir::arith::SubIOp::create(firOpBuilder, loc, zero, loopStepVar);
  mlir::Value incr = mlir::arith::SelectOp::create(
      firOpBuilder, loc, isDownwards, negStep, loopStepVar);
  mlir::Value lb = mlir::arith::SelectOp::create(firOpBuilder, loc, isDownwards,
                                                 loopUBVar, loopLBVar);
  mlir::Value ub = mlir::arith::SelectOp::create(firOpBuilder, loc, isDownwards,
                                                 loopLBVar, loopUBVar);

  // Compute the trip count assuming lb <= ub. This guarantees that the result
  // is non-negative and we can use unsigned arithmetic.
  mlir::Value span = mlir::arith::SubIOp::create(
      firOpBuilder, loc, ub, lb, ::mlir::arith::IntegerOverflowFlags::nuw);
  mlir::Value tcMinusOne =
      mlir::arith::DivUIOp::create(firOpBuilder, loc, span, incr);
  mlir::Value tcIfLooping =
      mlir::arith::AddIOp::create(firOpBuilder, loc, tcMinusOne, one,
                                  ::mlir::arith::IntegerOverflowFlags::nuw);

  // Fall back to 0 if lb > ub
  mlir::Value isZeroTC = mlir::arith::CmpIOp::create(
      firOpBuilder, loc, mlir::arith::CmpIPredicate::slt, ub, lb);
  mlir::Value tripcount = mlir::arith::SelectOp::create(
      firOpBuilder, loc, isZeroTC, zero, tcIfLooping);

  // Create the CLI handle.
  auto newcli = mlir::omp::NewCliOp::create(firOpBuilder, loc);
  mlir::Value cli = newcli.getResult();

  auto ivCallback = [&](mlir::Operation *op)
      -> llvm::SmallVector<const Fortran::semantics::Symbol *> {
    mlir::Region &region = op->getRegion(0);

    // Create the op's region skeleton (BB taking the iv as argument)
    firOpBuilder.createBlock(&region, {}, {loopVarType}, {loc});

    // Compute the value of the loop variable from the logical iteration number.
    mlir::Value natIterNum = fir::getBase(region.front().getArgument(0));
    mlir::Value scaled =
        mlir::arith::MulIOp::create(firOpBuilder, loc, natIterNum, loopStepVar);
    mlir::Value userVal =
        mlir::arith::AddIOp::create(firOpBuilder, loc, loopLBVar, scaled);

    // Write loop value to loop variable
    mlir::Operation *storeOp = setLoopVar(converter, loc, userVal, iv);

    firOpBuilder.setInsertionPointAfter(storeOp);
    return {iv};
  };

  // Create the omp.canonical_loop operation
  auto canonLoop = genOpWithBody<mlir::omp::CanonicalLoopOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, nestedEval,
                        directive)
          .setClauses(&item->clauses)
          .setPrivatize(false)
          .setGenRegionEntryCb(ivCallback),
      queue, item, tripcount, cli);

  firOpBuilder.setInsertionPointAfter(canonLoop);
  return canonLoop;
}

static void genUnrollOp(Fortran::lower::AbstractConverter &converter,
                        Fortran::lower::SymMap &symTable,
                        lower::StatementContext &stmtCtx,
                        Fortran::semantics::SemanticsContext &semaCtx,
                        Fortran::lower::pft::Evaluation &eval,
                        mlir::Location loc, const ConstructQueue &queue,
                        ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  mlir::omp::LoopRelatedClauseOps loopInfo;
  llvm::SmallVector<const semantics::Symbol *> iv;
  collectLoopRelatedInfo(converter, loc, eval, item->clauses, loopInfo, iv);

  // Clauses for unrolling not yet implemnted
  ClauseProcessor cp(converter, semaCtx, item->clauses);
  cp.processTODO<clause::Partial, clause::Full>(
      loc, llvm::omp::Directive::OMPD_unroll);

  // Emit the associated loop
  auto canonLoop =
      genCanonicalLoopOp(converter, symTable, semaCtx, eval, loc, queue, item,
                         iv, llvm::omp::Directive::OMPD_unroll);

  // Apply unrolling to it
  auto cli = canonLoop.getCli();
  mlir::omp::UnrollHeuristicOp::create(firOpBuilder, loc, cli);
}

static mlir::omp::MaskedOp
genMaskedOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            lower::StatementContext &stmtCtx,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::const_iterator item) {
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
  if (!semaCtx.langOptions().OpenMPSimd)
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
  assert((!enableDelayedPrivatization || dsp) &&
         "expected valid DataSharingProcessor");

  OpWithBodyGenInfo genInfo =
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_parallel)
          .setClauses(&item->clauses)
          .setEntryBlockArgs(&args)
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
  return mlir::omp::ScanOp::create(converter.getFirOpBuilder(),
                                   converter.getCurrentLocation(), clauseOps);
}

static mlir::omp::SectionsOp
genSectionsOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue,
              ConstructQueue::const_iterator item) {
  const parser::OpenMPSectionsConstruct *sectionsConstruct =
      getSectionsConstructStackTop(converter);
  assert(sectionsConstruct && "Missing additional parsing information");

  const auto &sectionBlocks =
      std::get<std::list<parser::OpenMPConstruct>>(sectionsConstruct->t);
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
  auto sectionsOp = mlir::omp::SectionsOp::create(builder, loc, clauseOps);

  // Create entry block with reduction variables as arguments.
  EntryBlockArgs args;
  // TODO: Add private syms and vars.
  args.reduction.syms = reductionSyms;
  args.reduction.vars = clauseOps.reductionVars;

  genEntryBlock(builder, args, sectionsOp.getRegion());
  mlir::Operation *terminator =
      lower::genOpenMPTerminator(builder, sectionsOp, loc);

  // Generate nested SECTION constructs.
  // This is done here rather than in genOMP([...], OpenMPSectionConstruct )
  // because we need to run genReductionVars on each omp.section so that the
  // reduction variable gets mapped to the private version
  for (auto [construct, nestedEval] :
       llvm::zip(sectionBlocks, eval.getNestedEvaluations())) {
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
            .setDataSharingProcessor(&dsp)
            .setEntryBlockArgs(&args),
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
    mlir::omp::BarrierOp::create(builder, loc);

  return sectionsOp;
}

static mlir::Operation *
genScopeOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
           semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
           mlir::Location loc, const ConstructQueue &queue,
           ConstructQueue::const_iterator item) {
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(loc, "Scope construct");
  return nullptr;
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

static bool isDuplicateMappedSymbol(
    const semantics::Symbol &sym,
    const llvm::SetVector<const semantics::Symbol *> &privatizedSyms,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &hasDevSyms,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &mappedSyms) {
  llvm::SmallVector<const semantics::Symbol *> concatSyms;
  concatSyms.reserve(privatizedSyms.size() + hasDevSyms.size() +
                     mappedSyms.size());
  concatSyms.append(privatizedSyms.begin(), privatizedSyms.end());
  concatSyms.append(hasDevSyms.begin(), hasDevSyms.end());
  concatSyms.append(mappedSyms.begin(), mappedSyms.end());

  auto checkSymbol = [&](const semantics::Symbol &checkSym) {
    return std::any_of(concatSyms.begin(), concatSyms.end(),
                       [&](auto v) { return v->GetUltimate() == checkSym; });
  };

  if (checkSymbol(sym))
    return true;

  const auto *hostAssoc{sym.detailsIf<semantics::HostAssocDetails>()};
  if (hostAssoc && checkSymbol(hostAssoc->symbol()))
    return true;

  return checkSymbol(sym.GetUltimate());
}

static mlir::omp::TargetOp
genTargetOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            lower::StatementContext &stmtCtx,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  bool isTargetDevice =
      llvm::cast<mlir::omp::OffloadModuleInterface>(*converter.getModuleOp())
          .getIsTargetDevice();

  // Introduce a new host_eval information structure for this target region.
  if (!isTargetDevice)
    converter.getStateStack().stackPush<HostEvalInfoStackFrame>();

  mlir::omp::TargetOperands clauseOps;
  DefaultMapsTy defaultMaps;
  llvm::SmallVector<const semantics::Symbol *> mapSyms, isDevicePtrSyms,
      hasDeviceAddrSyms;
  genTargetClauses(converter, semaCtx, symTable, stmtCtx, eval, item->clauses,
                   loc, clauseOps, defaultMaps, hasDeviceAddrSyms,
                   isDevicePtrSyms, mapSyms);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/
                           lower::omp::isLastItemInQueue(item, queue),
                           /*useDelayedPrivatization=*/true, symTable,
                           /*isTargetPrivitization=*/true);
  dsp.processStep1(&clauseOps);

  // 5.8.1 Implicit Data-Mapping Attribute Rules
  // The following code follows the implicit data-mapping rules to map all the
  // symbols used inside the region that do not have explicit data-environment
  // attribute clauses (neither data-sharing; e.g. `private`, nor `map`
  // clauses).
  auto captureImplicitMap = [&](const semantics::Symbol &sym) {
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

    // Skip parameters/constants as they do not need to be mapped.
    if (semantics::IsNamedConstant(sym))
      return;

    if (!isDuplicateMappedSymbol(sym, dsp.getAllSymbolsToPrivatize(),
                                 hasDeviceAddrSyms, mapSyms)) {
      if (const auto *details =
              sym.template detailsIf<semantics::HostAssocDetails>())
        converter.copySymbolBinding(details->symbol(), sym);
      std::stringstream name;
      fir::ExtendedValue dataExv = converter.getSymbolExtendedValue(sym);
      name << sym.name().ToString();

      mlir::FlatSymbolRefAttr mapperId;
      if (sym.GetType()->category() == semantics::DeclTypeSpec::TypeDerived) {
        auto &typeSpec = sym.GetType()->derivedTypeSpec();
        std::string mapperIdName =
            typeSpec.name().ToString() + llvm::omp::OmpDefaultMapperName;
        if (auto *sym = converter.getCurrentScope().FindSymbol(mapperIdName))
          mapperIdName = converter.mangleName(mapperIdName, sym->owner());
        if (converter.getModuleOp().lookupSymbol(mapperIdName))
          mapperId = mlir::FlatSymbolRefAttr::get(&converter.getMLIRContext(),
                                                  mapperIdName);
      }

      fir::factory::AddrAndBoundsInfo info =
          Fortran::lower::getDataOperandBaseAddr(
              converter, firOpBuilder, sym.GetUltimate(),
              converter.getCurrentLocation());
      llvm::SmallVector<mlir::Value> bounds = flangomp::genBoundsOps(
          firOpBuilder, info.rawInput,
          semantics::IsAssumedSizeArray(sym.GetUltimate()),
          semantics::IsOptional(sym.GetUltimate()));
      mlir::Value baseOp = info.rawInput;
      mlir::Type eleType = baseOp.getType();
      if (auto refType = mlir::dyn_cast<fir::ReferenceType>(baseOp.getType()))
        eleType = refType.getElementType();

      std::pair<llvm::omp::OpenMPOffloadMappingFlags,
                mlir::omp::VariableCaptureKind>
          mapFlagAndKind = getImplicitMapTypeAndKind(
              firOpBuilder, converter, defaultMaps, eleType, loc, sym);

      mlir::Value mapOp = createMapInfoOp(
          firOpBuilder, converter.getCurrentLocation(), baseOp,
          /*varPtrPtr=*/mlir::Value{}, name.str(), bounds, /*members=*/{},
          /*membersIndex=*/mlir::ArrayAttr{},
          static_cast<
              std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
              std::get<0>(mapFlagAndKind)),
          std::get<1>(mapFlagAndKind), baseOp.getType(),
          /*partialMap=*/false, mapperId);

      clauseOps.mapVars.push_back(mapOp);
      mapSyms.push_back(&sym);
    }
  };
  lower::pft::visitAllSymbols(eval, captureImplicitMap);

  auto targetOp = mlir::omp::TargetOp::create(firOpBuilder, loc, clauseOps);

  llvm::SmallVector<mlir::Value> hasDeviceAddrBaseValues, mapBaseValues;
  extractMappedBaseValues(clauseOps.hasDeviceAddrVars, hasDeviceAddrBaseValues);
  extractMappedBaseValues(clauseOps.mapVars, mapBaseValues);

  EntryBlockArgs args;
  args.hasDeviceAddr.syms = hasDeviceAddrSyms;
  args.hasDeviceAddr.vars = hasDeviceAddrBaseValues;
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
    converter.getStateStack().stackPop();
  return targetOp;
}

static mlir::omp::TargetDataOp genTargetDataOp(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  mlir::omp::TargetDataOperands clauseOps;
  llvm::SmallVector<const semantics::Symbol *> useDeviceAddrSyms,
      useDevicePtrSyms;
  genTargetDataClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                       clauseOps, useDeviceAddrSyms, useDevicePtrSyms);

  auto targetDataOp = mlir::omp::TargetDataOp::create(
      converter.getFirOpBuilder(), loc, clauseOps);

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
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    mlir::Location loc, const ConstructQueue &queue,
    ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

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
  genTargetEnterExitUpdateDataClauses(converter, semaCtx, symTable, stmtCtx,
                                      item->clauses, loc, directive, clauseOps);

  return OpTy::create(firOpBuilder, loc, clauseOps);
}

static mlir::omp::TaskOp
genTaskOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
          lower::StatementContext &stmtCtx,
          semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
          mlir::Location loc, const ConstructQueue &queue,
          ConstructQueue::const_iterator item) {
  mlir::omp::TaskOperands clauseOps;
  llvm::SmallVector<const semantics::Symbol *> inReductionSyms;
  genTaskClauses(converter, semaCtx, symTable, stmtCtx, item->clauses, loc,
                 clauseOps, inReductionSyms);

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
  taskArgs.inReduction.syms = inReductionSyms;
  taskArgs.inReduction.vars = clauseOps.inReductionVars;

  return genOpWithBody<mlir::omp::TaskOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_task)
          .setClauses(&item->clauses)
          .setDataSharingProcessor(&dsp)
          .setEntryBlockArgs(&taskArgs),
      queue, item, clauseOps);
}

static mlir::omp::TaskgroupOp
genTaskgroupOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
               semantics::SemanticsContext &semaCtx,
               lower::pft::Evaluation &eval, mlir::Location loc,
               const ConstructQueue &queue,
               ConstructQueue::const_iterator item) {
  mlir::omp::TaskgroupOperands clauseOps;
  llvm::SmallVector<const semantics::Symbol *> taskReductionSyms;
  genTaskgroupClauses(converter, semaCtx, item->clauses, loc, clauseOps,
                      taskReductionSyms);

  EntryBlockArgs taskgroupArgs;
  taskgroupArgs.taskReduction.syms = taskReductionSyms;
  taskgroupArgs.taskReduction.vars = clauseOps.taskReductionVars;

  return genOpWithBody<mlir::omp::TaskgroupOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_taskgroup)
          .setClauses(&item->clauses)
          .setEntryBlockArgs(&taskgroupArgs),
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
  return mlir::omp::TaskwaitOp::create(converter.getFirOpBuilder(), loc,
                                       clauseOps);
}

static mlir::omp::TaskyieldOp
genTaskyieldOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
               semantics::SemanticsContext &semaCtx,
               lower::pft::Evaluation &eval, mlir::Location loc,
               const ConstructQueue &queue,
               ConstructQueue::const_iterator item) {
  return mlir::omp::TaskyieldOp::create(converter.getFirOpBuilder(), loc);
}

static mlir::omp::WorkshareOp genWorkshareOp(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
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
           lower::StatementContext &stmtCtx,
           semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
           mlir::Location loc, const ConstructQueue &queue,
           ConstructQueue::const_iterator item) {
  lower::SymMapScope scope(symTable);
  mlir::omp::TeamsOperands clauseOps;
  llvm::SmallVector<const semantics::Symbol *> reductionSyms;
  genTeamsClauses(converter, semaCtx, stmtCtx, item->clauses, loc, clauseOps,
                  reductionSyms);

  EntryBlockArgs args;
  // TODO: Add private syms and vars.
  args.reduction.syms = reductionSyms;
  args.reduction.vars = clauseOps.reductionVars;

  return genOpWithBody<mlir::omp::TeamsOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_teams)
          .setClauses(&item->clauses)
          .setEntryBlockArgs(&args),
      queue, item, clauseOps);
}

static mlir::omp::WorkdistributeOp genWorkdistributeOp(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::Location loc, const ConstructQueue &queue,
    ConstructQueue::const_iterator item) {
  return genOpWithBody<mlir::omp::WorkdistributeOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_workdistribute),
      queue, item);
}

//===----------------------------------------------------------------------===//
// Code generation functions for the standalone version of constructs that can
// also be a leaf of a composite construct
//===----------------------------------------------------------------------===//

static mlir::omp::DistributeOp genStandaloneDistribute(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  mlir::omp::DistributeOperands distributeClauseOps;
  genDistributeClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                       distributeClauseOps);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           enableDelayedPrivatization, symTable);
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
  return distributeOp;
}

static mlir::omp::WsloopOp genStandaloneDo(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
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
  return wsloopOp;
}

static mlir::omp::ParallelOp genStandaloneParallel(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  lower::SymMapScope scope(symTable);
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
  return genParallelOp(converter, symTable, semaCtx, eval, loc, queue, item,
                       parallelClauseOps, parallelArgs,
                       enableDelayedPrivatization ? &dsp.value() : nullptr);
}

static mlir::omp::SimdOp
genStandaloneSimd(lower::AbstractConverter &converter, lower::SymMap &symTable,
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
  return simdOp;
}

static mlir::omp::TaskloopOp genStandaloneTaskloop(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  mlir::omp::TaskloopOperands taskloopClauseOps;
  genTaskloopClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                     taskloopClauseOps);
  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           enableDelayedPrivatization, symTable);
  dsp.processStep1(&taskloopClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);

  EntryBlockArgs taskloopArgs;
  taskloopArgs.priv.syms = dsp.getDelayedPrivSymbols();
  taskloopArgs.priv.vars = taskloopClauseOps.privateVars;

  auto taskLoopOp = genWrapperOp<mlir::omp::TaskloopOp>(
      converter, loc, taskloopClauseOps, taskloopArgs);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, item,
                loopNestClauseOps, iv, {{taskLoopOp, taskloopArgs}},
                llvm::omp::Directive::OMPD_taskloop, dsp);
  return taskLoopOp;
}

//===----------------------------------------------------------------------===//
// Code generation functions for composite constructs
//===----------------------------------------------------------------------===//

static mlir::omp::DistributeOp genCompositeDistributeParallelDo(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
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
  return distributeOp;
}

static mlir::omp::DistributeOp genCompositeDistributeParallelDoSimd(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
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

  DataSharingProcessor parallelItemDSP(
      converter, semaCtx, parallelItem->clauses, eval,
      /*shouldCollectPreDeterminedSymbols=*/false,
      /*useDelayedPrivatization=*/true, symTable);
  parallelItemDSP.processStep1(&parallelClauseOps);

  EntryBlockArgs parallelArgs;
  parallelArgs.priv.syms = parallelItemDSP.getDelayedPrivSymbols();
  parallelArgs.priv.vars = parallelClauseOps.privateVars;
  parallelArgs.reduction.syms = parallelReductionSyms;
  parallelArgs.reduction.vars = parallelClauseOps.reductionVars;
  genParallelOp(converter, symTable, semaCtx, eval, loc, queue, parallelItem,
                parallelClauseOps, parallelArgs, &parallelItemDSP,
                /*isComposite=*/true);

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

  DataSharingProcessor simdItemDSP(converter, semaCtx, simdItem->clauses, eval,
                                   /*shouldCollectPreDeterminedSymbols=*/true,
                                   /*useDelayedPrivatization=*/true, symTable);
  simdItemDSP.processStep1(&simdClauseOps);

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
  simdArgs.priv.syms = simdItemDSP.getDelayedPrivSymbols();
  simdArgs.priv.vars = simdClauseOps.privateVars;
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
                llvm::omp::Directive::OMPD_distribute_parallel_do_simd,
                simdItemDSP);
  return distributeOp;
}

static mlir::omp::DistributeOp genCompositeDistributeSimd(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
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

  DataSharingProcessor distributeItemDSP(
      converter, semaCtx, distributeItem->clauses, eval,
      /*shouldCollectPreDeterminedSymbols=*/false,
      /*useDelayedPrivatization=*/true, symTable);
  distributeItemDSP.processStep1(&distributeClauseOps);

  DataSharingProcessor simdItemDSP(converter, semaCtx, simdItem->clauses, eval,
                                   /*shouldCollectPreDeterminedSymbols=*/true,
                                   /*useDelayedPrivatization=*/true, symTable);
  simdItemDSP.processStep1(&simdClauseOps);

  // Pass the innermost leaf construct's clauses because that's where COLLAPSE
  // is placed by construct decomposition.
  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, simdItem->clauses, loc,
                     loopNestClauseOps, iv);

  // Operation creation.
  EntryBlockArgs distributeArgs;
  distributeArgs.priv.syms = distributeItemDSP.getDelayedPrivSymbols();
  distributeArgs.priv.vars = distributeClauseOps.privateVars;
  auto distributeOp = genWrapperOp<mlir::omp::DistributeOp>(
      converter, loc, distributeClauseOps, distributeArgs);
  distributeOp.setComposite(/*val=*/true);

  EntryBlockArgs simdArgs;
  simdArgs.priv.syms = simdItemDSP.getDelayedPrivSymbols();
  simdArgs.priv.vars = simdClauseOps.privateVars;
  simdArgs.reduction.syms = simdReductionSyms;
  simdArgs.reduction.vars = simdClauseOps.reductionVars;
  auto simdOp =
      genWrapperOp<mlir::omp::SimdOp>(converter, loc, simdClauseOps, simdArgs);
  simdOp.setComposite(/*val=*/true);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, simdItem,
                loopNestClauseOps, iv,
                {{distributeOp, distributeArgs}, {simdOp, simdArgs}},
                llvm::omp::Directive::OMPD_distribute_simd, simdItemDSP);
  return distributeOp;
}

static mlir::omp::WsloopOp genCompositeDoSimd(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
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

  DataSharingProcessor wsloopItemDSP(
      converter, semaCtx, doItem->clauses, eval,
      /*shouldCollectPreDeterminedSymbols=*/false,
      /*useDelayedPrivatization=*/true, symTable);
  wsloopItemDSP.processStep1(&wsloopClauseOps);

  DataSharingProcessor simdItemDSP(converter, semaCtx, simdItem->clauses, eval,
                                   /*shouldCollectPreDeterminedSymbols=*/true,
                                   /*useDelayedPrivatization=*/true, symTable);
  simdItemDSP.processStep1(&simdClauseOps, simdItem->id);

  // Pass the innermost leaf construct's clauses because that's where COLLAPSE
  // is placed by construct decomposition.
  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, simdItem->clauses, loc,
                     loopNestClauseOps, iv);

  // Operation creation.
  EntryBlockArgs wsloopArgs;
  wsloopArgs.priv.syms = wsloopItemDSP.getDelayedPrivSymbols();
  wsloopArgs.priv.vars = wsloopClauseOps.privateVars;
  wsloopArgs.reduction.syms = wsloopReductionSyms;
  wsloopArgs.reduction.vars = wsloopClauseOps.reductionVars;
  auto wsloopOp = genWrapperOp<mlir::omp::WsloopOp>(
      converter, loc, wsloopClauseOps, wsloopArgs);
  wsloopOp.setComposite(/*val=*/true);

  EntryBlockArgs simdArgs;
  simdArgs.priv.syms = simdItemDSP.getDelayedPrivSymbols();
  simdArgs.priv.vars = simdClauseOps.privateVars;
  simdArgs.reduction.syms = simdReductionSyms;
  simdArgs.reduction.vars = simdClauseOps.reductionVars;
  auto simdOp =
      genWrapperOp<mlir::omp::SimdOp>(converter, loc, simdClauseOps, simdArgs);
  simdOp.setComposite(/*val=*/true);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, simdItem,
                loopNestClauseOps, iv,
                {{wsloopOp, wsloopArgs}, {simdOp, simdArgs}},
                llvm::omp::Directive::OMPD_do_simd, simdItemDSP);
  return wsloopOp;
}

static mlir::omp::TaskloopOp genCompositeTaskloopSimd(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  assert(std::distance(item, queue.end()) == 2 && "Invalid leaf constructs");
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(loc, "Composite TASKLOOP SIMD");
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

static bool genOMPCompositeDispatch(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item,
    mlir::Operation *&newOp) {
  using llvm::omp::Directive;
  using lower::omp::matchLeafSequence;

  // TODO: Privatization for composite constructs is currently only done based
  // on the clauses for their last leaf construct, which may not always be
  // correct. Consider per-leaf privatization of composite constructs once
  // delayed privatization is supported by all participating ops.
  if (matchLeafSequence(item, queue, Directive::OMPD_distribute_parallel_do))
    newOp = genCompositeDistributeParallelDo(converter, symTable, stmtCtx,
                                             semaCtx, eval, loc, queue, item);
  else if (matchLeafSequence(item, queue,
                             Directive::OMPD_distribute_parallel_do_simd))
    newOp = genCompositeDistributeParallelDoSimd(
        converter, symTable, stmtCtx, semaCtx, eval, loc, queue, item);
  else if (matchLeafSequence(item, queue, Directive::OMPD_distribute_simd))
    newOp = genCompositeDistributeSimd(converter, symTable, stmtCtx, semaCtx,
                                       eval, loc, queue, item);
  else if (matchLeafSequence(item, queue, Directive::OMPD_do_simd))
    newOp = genCompositeDoSimd(converter, symTable, stmtCtx, semaCtx, eval, loc,
                               queue, item);
  else if (matchLeafSequence(item, queue, Directive::OMPD_taskloop_simd))
    newOp = genCompositeTaskloopSimd(converter, symTable, stmtCtx, semaCtx,
                                     eval, loc, queue, item);
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

  lower::StatementContext stmtCtx;
  mlir::Operation *newOp = nullptr;

  // Generate cleanup code for the stmtCtx after newOp
  auto finalizeStmtCtx = [&]() {
    if (newOp) {
      fir::FirOpBuilder &builder = converter.getFirOpBuilder();
      fir::FirOpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(newOp);
      stmtCtx.finalizeAndPop();
    }
  };

  bool loopLeaf = llvm::omp::getDirectiveAssociation(item->id) ==
                  llvm::omp::Association::Loop;
  if (loopLeaf) {
    symTable.pushScope();
    if (genOMPCompositeDispatch(converter, symTable, stmtCtx, semaCtx, eval,
                                loc, queue, item, newOp)) {
      symTable.popScope();
      finalizeStmtCtx();
      return;
    }
  }

  switch (llvm::omp::Directive dir = item->id) {
  case llvm::omp::Directive::OMPD_barrier:
    newOp = genBarrierOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_distribute:
    newOp = genStandaloneDistribute(converter, symTable, stmtCtx, semaCtx, eval,
                                    loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_do:
    newOp = genStandaloneDo(converter, symTable, stmtCtx, semaCtx, eval, loc,
                            queue, item);
    break;
  case llvm::omp::Directive::OMPD_loop:
    newOp = genLoopOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_masked:
    newOp = genMaskedOp(converter, symTable, stmtCtx, semaCtx, eval, loc, queue,
                        item);
    break;
  case llvm::omp::Directive::OMPD_master:
    newOp = genMasterOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_ordered:
    // Block-associated "ordered" construct.
    newOp = genOrderedRegionOp(converter, symTable, semaCtx, eval, loc, queue,
                               item);
    break;
  case llvm::omp::Directive::OMPD_parallel:
    newOp = genStandaloneParallel(converter, symTable, stmtCtx, semaCtx, eval,
                                  loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_scan:
    newOp = genScanOp(converter, symTable, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_section:
    llvm_unreachable("genOMPDispatch: OMPD_section");
    // Lowered in the enclosing genSectionsOp.
    break;
  case llvm::omp::Directive::OMPD_sections:
    genSectionsOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_simd:
    newOp =
        genStandaloneSimd(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_scope:
    newOp = genScopeOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_single:
    newOp = genSingleOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target:
    newOp = genTargetOp(converter, symTable, stmtCtx, semaCtx, eval, loc, queue,
                        item);
    break;
  case llvm::omp::Directive::OMPD_target_data:
    newOp = genTargetDataOp(converter, symTable, stmtCtx, semaCtx, eval, loc,
                            queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    newOp = genTargetEnterExitUpdateDataOp<mlir::omp::TargetEnterDataOp>(
        converter, symTable, stmtCtx, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_exit_data:
    newOp = genTargetEnterExitUpdateDataOp<mlir::omp::TargetExitDataOp>(
        converter, symTable, stmtCtx, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_update:
    newOp = genTargetEnterExitUpdateDataOp<mlir::omp::TargetUpdateOp>(
        converter, symTable, stmtCtx, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_task:
    newOp = genTaskOp(converter, symTable, stmtCtx, semaCtx, eval, loc, queue,
                      item);
    break;
  case llvm::omp::Directive::OMPD_taskgroup:
    newOp =
        genTaskgroupOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskloop:
    newOp = genStandaloneTaskloop(converter, symTable, stmtCtx, semaCtx, eval,
                                  loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    newOp = genTaskwaitOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    newOp =
        genTaskyieldOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_teams:
    newOp = genTeamsOp(converter, symTable, stmtCtx, semaCtx, eval, loc, queue,
                       item);
    break;
  case llvm::omp::Directive::OMPD_tile: {
    unsigned version = semaCtx.langOptions().OpenMPVersion;
    if (!semaCtx.langOptions().OpenMPSimd)
      TODO(loc, "Unhandled loop directive (" +
                    llvm::omp::getOpenMPDirectiveName(dir, version) + ")");
    break;
  }
  case llvm::omp::Directive::OMPD_unroll:
    genUnrollOp(converter, symTable, stmtCtx, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_workdistribute:
    newOp = genWorkdistributeOp(converter, symTable, semaCtx, eval, loc, queue,
                                item);
    break;
  case llvm::omp::Directive::OMPD_workshare:
    newOp = genWorkshareOp(converter, symTable, stmtCtx, semaCtx, eval, loc,
                           queue, item);
    break;
  default:
    // Combined and composite constructs should have been split into a sequence
    // of leaf constructs when building the construct queue.
    assert(!llvm::omp::isLeafConstruct(dir) &&
           "Unexpected compound construct.");
    break;
  }

  finalizeStmtCtx();
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
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPDeclarativeAssumes &assumesConstruct) {
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(converter.getCurrentLocation(), "OpenMP ASSUMES declaration");
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OmpDeclareVariantDirective &declareVariantDirective) {
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(converter.getCurrentLocation(), "OmpDeclareVariantDirective");
}

static void genOMP(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareReductionConstruct &declareReductionConstruct) {
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(converter.getCurrentLocation(), "OpenMPDeclareReductionConstruct");
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPDeclareSimdConstruct &declareSimdConstruct) {
  if (!semaCtx.langOptions().OpenMPSimd)
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
      std::get<parser::OmpMapperSpecifier>(declareMapperConstruct.t);
  const auto &mapperName{std::get<std::string>(spec.t)};
  const auto &varType{std::get<parser::TypeSpec>(spec.t)};
  const auto &varName{std::get<parser::Name>(spec.t)};
  assert(varType.declTypeSpec->category() ==
             semantics::DeclTypeSpec::Category::TypeDerived &&
         "Expected derived type");

  std::string mapperNameStr = mapperName;
  if (auto *sym = converter.getCurrentScope().FindSymbol(mapperNameStr))
    mapperNameStr = converter.mangleName(mapperNameStr, sym->owner());

  // Save current insertion point before moving to the module scope to create
  // the DeclareMapperOp
  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);

  firOpBuilder.setInsertionPointToStart(converter.getModuleOp().getBody());
  auto mlirType = converter.genType(varType.declTypeSpec->derivedTypeSpec());
  auto declMapperOp = mlir::omp::DeclareMapperOp::create(
      firOpBuilder, loc, mapperNameStr, mlirType);
  auto &region = declMapperOp.getRegion();
  firOpBuilder.createBlock(&region);
  auto varVal = region.addArgument(firOpBuilder.getRefType(mlirType), loc);
  converter.bindSymbol(*varName.symbol, varVal);

  // Populate the declareMapper region with the map information.
  mlir::omp::DeclareMapperInfoOperands clauseOps;
  const auto *clauseList{
      parser::Unwrap<parser::OmpClauseList>(declareMapperConstruct.t)};
  List<Clause> clauses = makeClauses(*clauseList, semaCtx);
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processMap(loc, stmtCtx, clauseOps);
  mlir::omp::DeclareMapperInfoOp::create(firOpBuilder, loc, clauseOps.mapVars);
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct) {
  mlir::omp::DeclareTargetOperands clauseOps;
  llvm::SmallVector<DeclareTargetCaptureInfo> symbolAndClause;
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  getDeclareTargetInfo(converter, semaCtx, eval, declareTargetConstruct,
                       clauseOps, symbolAndClause);

  for (const DeclareTargetCaptureInfo &symClause : symbolAndClause) {
    mlir::Operation *op =
        mod.lookupSymbol(converter.mangleName(symClause.symbol));

    // Some symbols are deferred until later in the module, these are handled
    // upon finalization of the module for OpenMP inside of Bridge, so we simply
    // skip for now.
    if (!op)
      continue;

    markDeclareTarget(op, converter, symClause.clause, clauseOps.deviceType,
                      symClause.automap);
  }
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPGroupprivate &directive) {
  TODO(converter.getCurrentLocation(), "GROUPPRIVATE");
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

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPSimpleStandaloneConstruct &construct) {
  const auto &directive = std::get<parser::OmpDirectiveName>(construct.v.t);
  List<Clause> clauses = makeClauses(construct.v.Clauses(), semaCtx);
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
                   const parser::OpenMPFlushConstruct &construct) {
  const auto &argumentList = construct.v.Arguments();
  const auto &clauseList = construct.v.Clauses();
  ObjectList objects = makeObjects(argumentList, semaCtx);
  List<Clause> clauses =
      makeList(clauseList.v, [&](auto &&s) { return makeClause(s, semaCtx); });
  mlir::Location currentLocation = converter.genLocation(construct.source);

  ConstructQueue queue{buildConstructQueue(
      converter.getFirOpBuilder().getModule(), semaCtx, eval, construct.source,
      llvm::omp::Directive::OMPD_flush, clauses)};
  genFlushOp(converter, symTable, semaCtx, eval, currentLocation, objects,
             queue, queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPCancelConstruct &cancelConstruct) {
  List<Clause> clauses = makeList(cancelConstruct.v.Clauses().v, [&](auto &&s) {
    return makeClause(s, semaCtx);
  });
  mlir::Location loc = converter.genLocation(cancelConstruct.source);

  ConstructQueue queue{buildConstructQueue(
      converter.getFirOpBuilder().getModule(), semaCtx, eval,
      cancelConstruct.source, llvm::omp::Directive::OMPD_cancel, clauses)};
  genCancelOp(converter, semaCtx, eval, loc, queue, queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPCancellationPointConstruct
                       &cancellationPointConstruct) {
  List<Clause> clauses =
      makeList(cancellationPointConstruct.v.Clauses().v,
               [&](auto &&s) { return makeClause(s, semaCtx); });
  mlir::Location loc = converter.genLocation(cancellationPointConstruct.source);

  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, cancellationPointConstruct.source,
                          llvm::omp::Directive::OMPD_cancel, clauses)};
  genCancellationPointOp(converter, semaCtx, eval, loc, queue, queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPDepobjConstruct &construct) {
  // These values will be ignored until the construct itself is implemented,
  // but run them anyway for the sake of testing (via a Todo test).
  ObjectList objects = makeObjects(construct.v.Arguments(), semaCtx);
  assert(objects.size() == 1);
  List<Clause> clauses = makeClauses(construct.v.Clauses(), semaCtx);
  assert(clauses.size() == 1);
  (void)objects;
  (void)clauses;

  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(converter.getCurrentLocation(), "OpenMPDepobjConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPInteropConstruct &interopConstruct) {
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(converter.getCurrentLocation(), "OpenMPInteropConstruct");
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  Fortran::common::visit(
      [&](auto &&s) { return genOMP(converter, symTable, semaCtx, eval, s); },
      standaloneConstruct.u);
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPAllocatorsConstruct &allocsConstruct) {
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(converter.getCurrentLocation(), "OpenMPAllocatorsConstruct");
}

//===----------------------------------------------------------------------===//
// OpenMPConstruct visitors
//===----------------------------------------------------------------------===//

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPAtomicConstruct &construct) {
  lowerAtomic(converter, symTable, semaCtx, eval, construct);
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpBlockConstruct &blockConstruct) {
  const parser::OmpDirectiveSpecification &beginSpec =
      blockConstruct.BeginDir();
  List<Clause> clauses = makeClauses(beginSpec.Clauses(), semaCtx);
  if (auto &endSpec = blockConstruct.EndDir())
    clauses.append(makeClauses(endSpec->Clauses(), semaCtx));

  llvm::omp::Directive directive = beginSpec.DirId();
  assert(llvm::omp::blockConstructSet.test(directive) &&
         "Expected block construct");
  mlir::Location currentLocation = converter.genLocation(beginSpec.source);

  for (const Clause &clause : clauses) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (!std::holds_alternative<clause::Affinity>(clause.u) &&
        !std::holds_alternative<clause::Allocate>(clause.u) &&
        !std::holds_alternative<clause::Copyin>(clause.u) &&
        !std::holds_alternative<clause::Copyprivate>(clause.u) &&
        !std::holds_alternative<clause::Default>(clause.u) &&
        !std::holds_alternative<clause::Defaultmap>(clause.u) &&
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
      if (!semaCtx.langOptions().OpenMPSimd)
        TODO(clauseLocation, name + " clause is not implemented yet");
    }
  }

  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, beginSpec.source, directive, clauses)};
  genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPAssumeConstruct &assumeConstruct) {
  mlir::Location clauseLocation = converter.genLocation(assumeConstruct.source);
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(clauseLocation, "OpenMP ASSUME construct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPCriticalConstruct &criticalConstruct) {
  const parser::OmpDirectiveSpecification &beginSpec =
      criticalConstruct.BeginDir();
  List<Clause> clauses = makeClauses(beginSpec.Clauses(), semaCtx);

  ConstructQueue queue{buildConstructQueue(
      converter.getFirOpBuilder().getModule(), semaCtx, eval, beginSpec.source,
      llvm::omp::Directive::OMPD_critical, clauses)};

  std::optional<parser::Name> critName;
  const parser::OmpArgumentList &args = beginSpec.Arguments();
  if (!args.v.empty()) {
    // All of these things should be guaranteed to exist after semantic checks.
    auto *object = parser::Unwrap<parser::OmpObject>(args.v.front());
    assert(object && "Expecting object as argument");
    auto *designator = semantics::omp::GetDesignatorFromObj(*object);
    assert(designator && "Expecting desginator in argument");
    auto *name = semantics::getDesignatorNameIfDataRef(*designator);
    assert(name && "Expecting dataref in designator");
    critName = *name;
  }
  mlir::Location currentLocation = converter.getCurrentLocation();
  genCriticalOp(converter, symTable, semaCtx, eval, currentLocation, queue,
                queue.begin(), critName);
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPUtilityConstruct &) {
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(converter.getCurrentLocation(), "OpenMPUtilityConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPDispatchConstruct &) {
  if (!semaCtx.langOptions().OpenMPSimd)
    TODO(converter.getCurrentLocation(), "OpenMPDispatchConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPExecutableAllocate &execAllocConstruct) {
  if (!semaCtx.langOptions().OpenMPSimd)
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

  auto &optLoopCons =
      std::get<std::optional<parser::NestedConstruct>>(loopConstruct.t);
  if (optLoopCons.has_value()) {
    if (auto *ompNestedLoopCons{
            std::get_if<common::Indirection<parser::OpenMPLoopConstruct>>(
                &*optLoopCons)}) {
      llvm::omp::Directive nestedDirective =
          parser::omp::GetOmpDirectiveName(*ompNestedLoopCons).v;
      switch (nestedDirective) {
      case llvm::omp::Directive::OMPD_tile:
        // Skip OMPD_tile since the tile sizes will be retrieved when
        // generating the omp.loop_nest op.
        break;
      default: {
        unsigned version = semaCtx.langOptions().OpenMPVersion;
        TODO(currentLocation,
             "Applying a loop-associated on the loop generated by the " +
                 llvm::omp::getOpenMPDirectiveName(nestedDirective, version) +
                 " construct");
      }
      }
    }
  }

  llvm::omp::Directive directive =
      parser::omp::GetOmpDirectiveName(beginLoopDirective).v;
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
      std::get<std::optional<parser::OmpEndSectionsDirective>>(
          sectionsConstruct.t);
  assert(endSectionsDirective &&
         "Missing end section directive should have been handled in semantics");
  clauses.append(makeClauses(
      std::get<parser::OmpClauseList>(endSectionsDirective->t), semaCtx));
  mlir::Location currentLocation = converter.getCurrentLocation();

  llvm::omp::Directive directive =
      std::get<parser::OmpSectionsDirective>(beginSectionsDirective.t).v;
  const parser::CharBlock &source =
      std::get<parser::OmpSectionsDirective>(beginSectionsDirective.t).source;
  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, source, directive, clauses)};

  mlir::SaveStateStack<SectionsConstructStackFrame> saveStateStack{
      converter.getStateStack(), sectionsConstruct};
  genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
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
    return mlir::omp::YieldOp::create(builder, loc);
  return mlir::omp::TerminatorOp::create(builder, loc);
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
    mlir::Value commonThreadprivateValue = mlir::omp::ThreadprivateOp::create(
        firOpBuilder, currentLocation, commonValue.getType(), commonValue);
    converter.bindSymbol(*common, commonThreadprivateValue);
    // Generate the threadprivate value for the common block member.
    symThreadprivateValue =
        genCommonBlockMember(converter, currentLocation, sym,
                             commonThreadprivateValue, common->size());
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

    mlir::Value symValue = fir::AddrOfOp::create(
        firOpBuilder, currentLocation, global.resultType(), global.getSymbol());
    symThreadprivateValue = mlir::omp::ThreadprivateOp::create(
        firOpBuilder, currentLocation, symValue.getType(), symValue);
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

    symThreadprivateValue = mlir::omp::ThreadprivateOp::create(
        firOpBuilder, currentLocation, symValue.getType(), symValue);
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
  if (const auto *block = std::get_if<parser::OmpBlockConstruct>(&omp.u)) {
    dir = block->BeginDir().DirId();
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
                      devType, declTar.automap);
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
