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
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/DirectivesCommon.h"
#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/PrivateReductionUtils.h"
#include "flang/Lower/Support/ReductionProcessor.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/tools.h"
#include "flang/Support/Flags.h"
#include "flang/Support/OpenMP-utils.h"
#include "flang/Utils/OpenMP.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/StateStack.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

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

/// Return the directive that is immediately nested inside of the given
/// \c parent evaluation, if it is its only non-end-statement nested evaluation
/// and it represents an OpenMP construct.
static lower::pft::Evaluation *
extractOnlyOmpNestedEval(lower::pft::Evaluation &parent) {
  if (!parent.hasNestedEvaluations())
    return nullptr;

  auto &nested = parent.getFirstNestedEvaluation();
  if (!nested.isA<parser::OpenMPConstruct>())
    return nullptr;

  for (auto &sibling : parent.getNestedEvaluations())
    if (&sibling != &nested && !sibling.isEndStmt())
      return nullptr;

  return &nested;
}

static llvm::SmallVector<Object>
makeObjects(llvm::ArrayRef<const semantics::Symbol *> syms) {
  llvm::SmallVector<Object> objects;
  objects.reserve(syms.size());
  llvm::transform(
      syms, std::back_inserter(objects), [](const semantics::Symbol *sym) {
        return Object{const_cast<semantics::Symbol *>(sym), std::nullopt};
      });
  return objects;
}

/// Structure holding the information needed to create and bind entry block
/// arguments associated to a single clause during OpenMP lowering.
struct ObjectEntryBlockArgsEntry {
  llvm::SmallVector<Object> objects;
  llvm::ArrayRef<mlir::Value> vars;

  bool isValid() const { return objects.size() <= vars.size(); }

  llvm::SmallVector<const semantics::Symbol *> getSyms() const {
    llvm::SmallVector<const semantics::Symbol *> syms;
    syms.reserve(objects.size());
    llvm::transform(objects, std::back_inserter(syms),
                    [](const Object &object) { return object.sym(); });
    return syms;
  }
};

struct ObjectEntryBlockArgs {
  ObjectEntryBlockArgsEntry hasDeviceAddr;
  llvm::ArrayRef<mlir::Value> hostEvalVars;
  ObjectEntryBlockArgsEntry inReduction;
  ObjectEntryBlockArgsEntry map;
  ObjectEntryBlockArgsEntry priv;
  ObjectEntryBlockArgsEntry reduction;
  ObjectEntryBlockArgsEntry taskReduction;
  ObjectEntryBlockArgsEntry useDeviceAddr;
  ObjectEntryBlockArgsEntry useDevicePtr;

  bool isValid() const {
    return hasDeviceAddr.isValid() && inReduction.isValid() && map.isValid() &&
           priv.isValid() && reduction.isValid() && taskReduction.isValid() &&
           useDeviceAddr.isValid() && useDevicePtr.isValid();
  }

  llvm::SmallVector<const semantics::Symbol *> getSyms() const {
    llvm::SmallVector<const semantics::Symbol *> syms;
    auto appendSyms = [&syms](const ObjectEntryBlockArgsEntry &entry) {
      syms.reserve(syms.size() + entry.objects.size());
      llvm::transform(entry.objects, std::back_inserter(syms),
                      [](const Object &object) { return object.sym(); });
    };
    appendSyms(hasDeviceAddr);
    appendSyms(inReduction);
    appendSyms(map);
    appendSyms(priv);
    appendSyms(reduction);
    appendSyms(taskReduction);
    appendSyms(useDeviceAddr);
    appendSyms(useDevicePtr);
    return syms;
  }

  auto getVars() const {
    return llvm::concat<const mlir::Value>(
        hasDeviceAddr.vars, hostEvalVars, inReduction.vars, map.vars, priv.vars,
        reduction.vars, taskReduction.vars, useDeviceAddr.vars,
        useDevicePtr.vars);
  }

  Fortran::common::openmp::EntryBlockArgs asEntryBlockArgs() const {
    Fortran::common::openmp::EntryBlockArgs args;
    args.hasDeviceAddrVars = hasDeviceAddr.vars;
    args.hostEvalVars = hostEvalVars;
    args.inReductionVars = inReduction.vars;
    args.mapVars = map.vars;
    args.privVars = priv.vars;
    args.reductionVars = reduction.vars;
    args.taskReductionVars = taskReduction.vars;
    args.useDeviceAddrVars = useDeviceAddr.vars;
    args.useDevicePtrVars = useDevicePtr.vars;
    return args;
  }
};

namespace {
/// Structure holding information that is needed to pass host-evaluated
/// information to later lowering stages.
class HostEvalInfo {
public:
  friend class HostEvalVisitor;

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

    for (auto numTeamsUpper : ops.numTeamsUpperVars)
      vars.push_back(numTeamsUpper);

    for (auto numThreads : ops.numThreadsVars)
      vars.push_back(numThreads);

    for (mlir::Value val : ops.threadLimitVars)
      vars.push_back(val);
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
                   ops.numTeamsUpperVars.size() + ops.numThreadsVars.size() +
                   ops.threadLimitVars.size() &&
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

    for (size_t i = 0; i < ops.numTeamsUpperVars.size(); ++i)
      ops.numTeamsUpperVars[i] = args[argIndex++];

    for (size_t i = 0; i < ops.numThreadsVars.size(); ++i)
      ops.numThreadsVars[i] = args[argIndex++];

    for (size_t i = 0; i < ops.threadLimitVars.size(); ++i)
      ops.threadLimitVars[i] = args[argIndex++];
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
    clauseOps.collapseNumLoops = ops.collapseNumLoops;
    ivOut.append(iv);
    return true;
  }

  /// Update \p clauseOps with the corresponding host-evaluated values if they
  /// have already been initialized but not yet applied.
  ///
  /// \returns whether an update was performed. If not, these clauses were not
  ///          evaluated in the host device.
  bool apply(mlir::omp::ParallelOperands &clauseOps) {
    if (ops.numThreadsVars.empty() || parallelApplied) {
      parallelApplied = true;
      return false;
    }

    parallelApplied = true;
    clauseOps.numThreadsVars = ops.numThreadsVars;
    return true;
  }

  /// Update \p clauseOps with the corresponding host-evaluated values if they
  /// have already been initialized.
  ///
  /// \returns whether an update was performed. If not, these clauses were not
  ///          evaluated in the host device.
  bool apply(mlir::omp::TeamsOperands &clauseOps) {
    if (!ops.numTeamsLower && ops.numTeamsUpperVars.empty() &&
        ops.threadLimitVars.empty())
      return false;

    clauseOps.numTeamsLower = ops.numTeamsLower;
    clauseOps.numTeamsUpperVars = ops.numTeamsUpperVars;
    clauseOps.threadLimitVars = ops.threadLimitVars;
    return true;
  }

private:
  mlir::omp::HostEvaluatedOperands ops;
  llvm::SmallVector<const semantics::Symbol *> iv;
  bool loopNestApplied = false, parallelApplied = false;
};

/// A base class to help iterate over OpenMP constructs based on an expected
/// sequence.
///
/// The main entry point visit() will call visitDirective() for the OpenMP
/// directive associated to the initial given evaluation based on whether it is
/// part of the initial set of directives of interest. A nested OpenMP
/// evaluation might optionally be also visited by the pattern recursively if it
/// meets all of the following conditions:
///   - It is the only nested evaluation, apart from an optional END statement
///     associated to the same directive.
///   - The OpenMP directive is part of the directive set returned by the
///     `visitDirective` call for the parent.
///
/// Subclasses define the expected pattern by implementing the initialize() and
/// visitDirective() methods, and users are expected to use visit() to trigger
/// the complete pattern visit.
class DirectivePatternVisitor {
public:
  DirectivePatternVisitor(semantics::SemanticsContext &semaCtx)
      : semaCtx{semaCtx} {}
  virtual ~DirectivePatternVisitor() = default;

  /// Run the pattern from the given evaluation.
  void visit(lower::pft::Evaluation &eval) {
    directivesOfInterest = initialize();
    visitEval(eval);
  }

protected:
  /// Initializes the visitor and returns the set of initial directives of
  /// interest to be matched the beginning of the pattern.
  virtual OmpDirectiveSet initialize() = 0;

  /// Visits a single directive and, based on it, returns the set of other
  /// directives of interest that would be part of the pattern if nested inside.
  virtual OmpDirectiveSet visitDirective(lower::pft::Evaluation &eval,
                                         llvm::omp::Directive dir) = 0;

  /// Obtain the list of clauses of the given OpenMP block or loop construct
  /// evaluation. If it's not an OpenMP construct, no modifications are made to
  /// the \c clauses output argument.
  void extractClauses(lower::pft::Evaluation &eval, List<Clause> &clauses) {
    const auto *ompEval{eval.getIf<parser::OpenMPConstruct>()};
    if (!ompEval)
      return;

    const parser::OmpClauseList *beginClauseList{nullptr};
    const parser::OmpClauseList *endClauseList{nullptr};
    common::visit(
        [&](const auto &construct) {
          using Type = llvm::remove_cvref_t<decltype(construct)>;
          if constexpr (std::is_same_v<Type, parser::OmpBlockConstruct> ||
                        std::is_same_v<Type, parser::OpenMPLoopConstruct>) {
            beginClauseList = &construct.BeginDir().Clauses();
            if (auto &endSpec{construct.EndDir()})
              endClauseList = &endSpec->Clauses();
          }
        },
        ompEval->u);

    assert(beginClauseList && "expected begin directive");
    clauses.append(makeClauses(*beginClauseList, semaCtx));

    if (endClauseList)
      clauses.append(makeClauses(*endClauseList, semaCtx));
  }

private:
  /// Decide whether an evaluation must be visited as part of the pattern.
  ///
  /// This is the case whenever it's an OpenMP construct and the associated
  /// directive is part of the current set of directives of interest.
  bool shouldVisitEval(lower::pft::Evaluation &eval) const {
    const auto *ompEval{eval.getIf<parser::OpenMPConstruct>()};
    if (!ompEval)
      return false;

    return directivesOfInterest.test(
        parser::omp::GetOmpDirectiveName(*ompEval).v);
  }

  /// Visits an evaluation and, potentially, recursively visits a single
  /// nested evaluation.
  ///
  /// For a nested evaluation to be recursively visited, it must be an OpenMP
  /// construct, have no sibling evaluations and match one of the
  /// next-directives of interest set returned by a call to visitDirective()
  /// on the parent evaluation.
  void visitEval(lower::pft::Evaluation &eval) {
    if (!shouldVisitEval(eval))
      return;

    const auto &ompEval{eval.get<parser::OpenMPConstruct>()};
    OmpDirectiveSet visitNested{
        visitDirective(eval, parser::omp::GetOmpDirectiveName(ompEval).v)};

    if (visitNested.empty())
      return;

    if (lower::pft::Evaluation *nestedEval = extractOnlyOmpNestedEval(eval)) {
      OmpDirectiveSet prevDirs{directivesOfInterest};
      directivesOfInterest = visitNested;
      visitEval(*nestedEval);
      directivesOfInterest = prevDirs;
    }
  }

protected:
  semantics::SemanticsContext &semaCtx;

private:
  OmpDirectiveSet directivesOfInterest;
};

/// Helper pattern to navigate target SPMD.
class TargetSPMDVisitor : public DirectivePatternVisitor {
public:
  using DirectivePatternVisitor::DirectivePatternVisitor;
  virtual ~TargetSPMDVisitor() = default;

protected:
  virtual OmpDirectiveSet initialize() override {
    teamsVisited = false;
    return llvm::omp::allTargetSet;
  }

  virtual OmpDirectiveSet visitDirective(lower::pft::Evaluation &eval,
                                         llvm::omp::Directive dir) override {
    using namespace llvm::omp;

    // The default implementation does nothing, except it returns the allowed
    // single nested directives for an SPMD kernel. If called by subclasses, it
    // helps navigate SPMD patterns.
    //
    // Patterns considered SPMD:
    //   - target teams distribute parallel do [simd]
    //   - target teams loop
    //   - target parallel do [simd]
    //   - target parallel loop
    switch (dir) {
    case OMPD_target:
      return topTeamsSet | topParallelSet;
    case OMPD_target_teams:
      // The 'bare' kernel type prevents the SPMD pattern from matching.
      if (hasOmpxBareClause(eval))
        return {};
      [[fallthrough]];
    case OMPD_teams:
      teamsVisited = true;
      return topDistributeSet | topLoopSet;
    case OMPD_target_parallel:
    case OMPD_parallel:
      return topLoopSet | topDoSet;
    default:
      return {};
    }
  }

  bool hasOmpxBareClause(lower::pft::Evaluation &eval) {
    List<lower::omp::Clause> clauses;
    extractClauses(eval, clauses);

    return llvm::find_if(clauses, [](const Clause &clause) {
             return std::holds_alternative<clause::OmpxBare>(clause.u);
           }) != clauses.end();
  }

protected:
  /// Whether a `teams` construct has been visited by visitDirective().
  bool teamsVisited;
};

/// Populates the given HostEvalInfo structure after processing clauses for
/// the given \p eval OpenMP target construct, or nested constructs, if these
/// must be evaluated outside of the target region per the spec.
///
/// In particular, this will ensure that in <tt>target teams</tt> and equivalent
/// nested constructs, the \c thread_limit, \c num_teams and \c num_threads
/// clauses will be evaluated in the host. Additionally, loop bounds and steps
/// will also be evaluated in the host if a <tt>target teams distribute</tt> or
/// target SPMD construct is detected (i.e. <tt>target teams distribute parallel
/// do [simd]</tt>, <tt>target parallel do [simd]</tt> or equivalent nesting).
///
/// The resulting updated HostEvalInfo structure is intended to be used to
/// populate the \c host_eval operands of the associated \c omp.target
/// operation, and also to be checked and used by later lowering steps to
/// populate the corresponding operands of the \c omp.teams, \c omp.parallel or
/// \c omp.loop_nest operations.
class HostEvalVisitor : public TargetSPMDVisitor {
public:
  HostEvalVisitor(lower::AbstractConverter &converter,
                  semantics::SemanticsContext &semaCtx,
                  lower::StatementContext &stmtCtx, mlir::Location loc,
                  HostEvalInfo &hostEvalInfo)
      : TargetSPMDVisitor{semaCtx}, converter{converter}, stmtCtx{stmtCtx},
        loc{loc}, hostEvalInfo{hostEvalInfo} {}
  virtual ~HostEvalVisitor() = default;

protected:
  virtual OmpDirectiveSet visitDirective(lower::pft::Evaluation &eval,
                                         llvm::omp::Directive dir) override {
    using namespace llvm::omp;

    List<lower::omp::Clause> clauses;
    extractClauses(eval, clauses);
    ClauseProcessor cp{converter, semaCtx, clauses};

    // Currently, we deal differently with e.g. `target parallel workshare` to
    // `target parallel` with a single nested `workshare`. The first case would
    // result in no clauses being evaluated in the host, as there's not a case
    // for it in the below switch statement. The second case would evaluate
    // `num_threads` clauses in the host, because `target parallel` could be
    // followed by a `do` construct, which would make this an SPMD target
    // region.
    //
    // TODO: We don't probably want to have such divergent behavior when dealing
    // with combined directives. We need to revisit this logic without listing
    // every possible combined directive containing a clause we'd otherwise
    // evaluate in the host if the directive was split into its leafs.
    switch (dir) {
    case OMPD_teams_distribute_parallel_do:
    case OMPD_teams_distribute_parallel_do_simd:
      cp.processThreadLimit(stmtCtx, hostEvalInfo.ops);
      [[fallthrough]];
    case OMPD_target_teams_distribute_parallel_do:
    case OMPD_target_teams_distribute_parallel_do_simd:
      cp.processNumTeams(stmtCtx, hostEvalInfo.ops);
      [[fallthrough]];
    case OMPD_distribute_parallel_do:
    case OMPD_distribute_parallel_do_simd:
    case OMPD_target_parallel_do:
    case OMPD_target_parallel_do_simd:
    case OMPD_target_parallel_loop:
    case OMPD_parallel_do:
    case OMPD_parallel_do_simd:
    case OMPD_parallel_loop:
      cp.processNumThreads(stmtCtx, hostEvalInfo.ops);
      [[fallthrough]];
    case OMPD_distribute:
    case OMPD_distribute_simd:
    case OMPD_do:
    case OMPD_do_simd:
      cp.processCollapse(loc, eval, hostEvalInfo.ops, hostEvalInfo.ops,
                         hostEvalInfo.iv);
      return {};

    case OMPD_teams:
      cp.processThreadLimit(stmtCtx, hostEvalInfo.ops);
      [[fallthrough]];
    case OMPD_target_teams:
      cp.processNumTeams(stmtCtx, hostEvalInfo.ops);
      break;

    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
      cp.processThreadLimit(stmtCtx, hostEvalInfo.ops);
      [[fallthrough]];
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
      cp.processCollapse(loc, eval, hostEvalInfo.ops, hostEvalInfo.ops,
                         hostEvalInfo.iv);
      cp.processNumTeams(stmtCtx, hostEvalInfo.ops);
      return {};

    case OMPD_teams_loop:
      cp.processThreadLimit(stmtCtx, hostEvalInfo.ops);
      [[fallthrough]];
    case OMPD_target_teams_loop:
      cp.processNumTeams(stmtCtx, hostEvalInfo.ops);
      [[fallthrough]];
    case OMPD_loop:
      cp.processCollapse(loc, eval, hostEvalInfo.ops, hostEvalInfo.ops,
                         hostEvalInfo.iv);
      return {};

    case OMPD_teams_workdistribute:
      cp.processThreadLimit(stmtCtx, hostEvalInfo.ops);
      [[fallthrough]];
    case OMPD_target_teams_workdistribute:
      cp.processNumTeams(stmtCtx, hostEvalInfo.ops);
      break;

    case OMPD_target_parallel:
    case OMPD_parallel:
      cp.processNumThreads(stmtCtx, hostEvalInfo.ops);
      break;

    case OMPD_target:
      break;

    default:
      return {};
    }

    // Visit nested directives as per the SPMD pattern.
    return TargetSPMDVisitor::visitDirective(eval, dir);
  }

private:
  lower::AbstractConverter &converter;
  lower::StatementContext &stmtCtx;
  mlir::Location loc;
  HostEvalInfo &hostEvalInfo;
};

/// Checks target regions and, based on the directives and clauses encountered,
/// determines its associated kernel type.
class KernelTypeVisitor : protected TargetSPMDVisitor {
public:
  KernelTypeVisitor(semantics::SemanticsContext &semaCtx,
                    mlir::ModuleOp moduleOp)
      : TargetSPMDVisitor{semaCtx}, moduleOp{moduleOp} {}
  virtual ~KernelTypeVisitor() = default;

  /// Executes the pattern and returns the kernel type of the given target
  /// region, or \c mlir::omp::TargetExecMode::generic by default for non-target
  /// evaluations.
  mlir::omp::TargetExecMode getKernelType(lower::pft::Evaluation &eval) {
    execMode = mlir::omp::TargetExecMode::generic;
    visit(eval);
    return execMode;
  }

protected:
  virtual OmpDirectiveSet visitDirective(lower::pft::Evaluation &eval,
                                         llvm::omp::Directive dir) override {
    using namespace llvm::omp;

    // We know this to be the case because any changes to the exec mode are made
    // only when we know for sure what it is, so pattern matching is always
    // stopped at these points.
    assert(execMode == mlir::omp::TargetExecMode::generic &&
           "unexpected non-default exec mode during pattern match");

    switch (dir) {
    case OMPD_target:
    case OMPD_target_parallel:
    case OMPD_parallel:
    case OMPD_teams:
      break;
    case OMPD_target_teams:
      if (hasOmpxBareClause(eval)) {
        execMode = mlir::omp::TargetExecMode::bare;
        return {};
      }
      break;
    case OMPD_target_teams_distribute_parallel_do:
    case OMPD_target_teams_distribute_parallel_do_simd:
    case OMPD_target_teams_loop:
    case OMPD_teams_distribute_parallel_do:
    case OMPD_teams_distribute_parallel_do_simd:
    case OMPD_teams_loop:
    case OMPD_distribute_parallel_do:
    case OMPD_distribute_parallel_do_simd:
      execMode = canPromoteSPMDToNoLoop(eval)
                     ? mlir::omp::TargetExecMode::spmd_no_loop
                     : mlir::omp::TargetExecMode::spmd;
      return {};
    case OMPD_target_parallel_do:
    case OMPD_target_parallel_do_simd:
    case OMPD_target_parallel_loop:
    case OMPD_parallel_do:
    case OMPD_parallel_do_simd:
    case OMPD_parallel_loop:
    case OMPD_do:
    case OMPD_do_simd:
      // SPMD kernels without a `teams` construct cannot be promoted to no-loop
      // mode.
      execMode = mlir::omp::TargetExecMode::spmd;
      return {};
    case OMPD_loop:
      // Prevent `target parallel loop` or equivalent nests to be promoted to
      // no-loop mode.
      execMode = teamsVisited && canPromoteSPMDToNoLoop(eval)
                     ? mlir::omp::TargetExecMode::spmd_no_loop
                     : mlir::omp::TargetExecMode::spmd;
      return {};
    default:
      return {};
    }

    // Visit nested directives as per the SPMD pattern.
    return TargetSPMDVisitor::visitDirective(eval, dir);
  }

private:
  bool canPromoteSPMDToNoLoop(lower::pft::Evaluation &eval) {
    List<lower::omp::Clause> clauses;
    extractClauses(eval, clauses);

    // First make sure the proper module attributes are present in order to
    // perform this optimization.
    auto ompFlags{
        llvm::cast<mlir::omp::OffloadModuleInterface>(*moduleOp).getFlags()};
    if (!ompFlags || !ompFlags.getAssumeTeamsOversubscription() ||
        !ompFlags.getAssumeThreadsOversubscription())
      return false;

    // The num_teams clause can break no-loop assumptions, and reductions are
    // slower in no-loop mode.
    return llvm::find_if(clauses, [](const Clause &clause) {
             return std::holds_alternative<clause::NumTeams>(clause.u) ||
                    std::holds_alternative<clause::Reduction>(clause.u);
           }) == clauses.end();
  }

private:
  mlir::ModuleOp moduleOp;
  mlir::omp::TargetExecMode execMode;
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

/// Bind objects to their corresponding entry block arguments.
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
                               const ObjectEntryBlockArgs &args) {
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
      [&bindSingleMapLike](llvm::ArrayRef<Object> objects,
                           llvm::ArrayRef<mlir::BlockArgument> args) {
        // Structure component symbols don't have bindings, and can only be
        // explicitly mapped individually. If a member is captured implicitly
        // we map the entirety of the derived type when we find its symbol.
        llvm::SmallVector<const semantics::Symbol *> processedSyms;
        for (const Object &object : objects) {
          const semantics::Symbol *sym = object.sym();
          if (!sym->owner().IsDerivedType())
            processedSyms.push_back(sym);
        }

        for (auto [sym, arg] : llvm::zip_equal(processedSyms, args))
          bindSingleMapLike(*sym, arg);
      };

  auto bindPrivateLike = [&converter, &firOpBuilder](
                             llvm::ArrayRef<Object> objects,
                             llvm::ArrayRef<mlir::Value> vars,
                             llvm::ArrayRef<mlir::BlockArgument> args) {
    llvm::SmallVector<const semantics::Symbol *> processedSyms;
    llvm::SmallVector<const Object *> processedObjects;
    for (const Object &object : objects) {
      const semantics::Symbol *sym = object.sym();
      if (const auto *commonDet =
              sym->detailsIf<semantics::CommonBlockDetails>()) {
        for (auto &mem : commonDet->objects()) {
          processedSyms.push_back(&*mem);
          processedObjects.push_back(&object);
        }
      } else {
        processedSyms.push_back(sym);
        processedObjects.push_back(&object);
      }
    }

    assert(processedSyms.size() == processedObjects.size());
    for (auto [sym, var, arg, object] :
         llvm::zip_equal(processedSyms, vars, args, processedObjects)) {
      bool skipBind =
          ReductionProcessor::isExpressionLoweredAsReductionObject(object) ||
          (object && sym->Rank() > 0 &&
           !fir::unwrapUntilSeqType(arg.getType()));
      if (skipBind)
        continue;

      converter.bindSymbol(
          *sym,
          hlfir::translateToExtendedValue(
              var.getLoc(), firOpBuilder, hlfir::Entity{arg},
              /*contiguousHint=*/
              evaluate::IsSimplyContiguous(*sym, converter.getFoldingContext()))
              .first);
    }
  };

  // Process in clause name alphabetical order to match block arguments order.
  // Do not bind host_eval variables because they cannot be used inside of the
  // corresponding region, except for very specific cases handled separately.
  bindMapLike(args.hasDeviceAddr.objects, op.getHasDeviceAddrBlockArgs());
  bindPrivateLike(args.inReduction.objects, args.inReduction.vars,
                  op.getInReductionBlockArgs());
  bindMapLike(args.map.objects, op.getMapBlockArgs());
  bindPrivateLike(args.priv.objects, args.priv.vars, op.getPrivateBlockArgs());
  bindPrivateLike(args.reduction.objects, args.reduction.vars,
                  op.getReductionBlockArgs());
  bindPrivateLike(args.taskReduction.objects, args.taskReduction.vars,
                  op.getTaskReductionBlockArgs());
  bindMapLike(args.useDeviceAddr.objects, op.getUseDeviceAddrBlockArgs());
  bindMapLike(args.useDevicePtr.objects, op.getUseDevicePtrBlockArgs());
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

static lower::pft::Evaluation *
getCollapsedLoopEval(lower::pft::Evaluation &eval, int collapseValue) {
  // Return the Evaluation of the innermost collapsed loop, or the current one
  // if there was no COLLAPSE.
  if (collapseValue == 0)
    return &eval;

  lower::pft::Evaluation *curEval = &eval;
  for (int i = 0; i < collapseValue; i++)
    curEval = getNestedDoConstruct(*curEval);
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

// Translate a semantics-layer device_type to the MLIR enum used by
// omp.groupprivate.
static mlir::omp::DeclareTargetDeviceType
toMLIRDeclareTargetDeviceType(Fortran::common::OmpDeviceType deviceType) {
  switch (deviceType) {
  case Fortran::common::OmpDeviceType::Any:
    return mlir::omp::DeclareTargetDeviceType::any;
  case Fortran::common::OmpDeviceType::Host:
    return mlir::omp::DeclareTargetDeviceType::host;
  case Fortran::common::OmpDeviceType::Nohost:
    return mlir::omp::DeclareTargetDeviceType::nohost;
  }
  llvm_unreachable("invalid OmpDeviceType");
}

static void groupprivatizeVars(lower::AbstractConverter &converter,
                               lower::pft::Evaluation &eval) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  auto module = converter.getModuleOp();

  // Create a groupprivate operation for the symbol.
  auto genGroupprivateOp = [&](const semantics::Symbol &sym) -> mlir::Value {
    std::string globalName = converter.mangleName(sym);
    fir::GlobalOp global = module.lookupSymbol<fir::GlobalOp>(globalName);
    if (!global)
      return mlir::Value();

    // The device_type modifier was recorded on the symbol during semantic
    // analysis.
    mlir::omp::DeclareTargetDeviceType deviceTypeEnum =
        mlir::omp::DeclareTargetDeviceType::any;
    Fortran::common::visit(
        [&](auto &&details) {
          using TypeD = llvm::remove_cvref_t<decltype(details)>;
          if constexpr (std::is_base_of_v<semantics::WithOmpDeclarative,
                                          TypeD>) {
            if (auto dt = details.ompGroupprivateDeviceType())
              deviceTypeEnum = toMLIRDeclareTargetDeviceType(*dt);
          }
        },
        sym.GetUltimate().details());
    mlir::omp::DeclareTargetDeviceTypeAttr deviceTypeAttr =
        mlir::omp::DeclareTargetDeviceTypeAttr::get(firOpBuilder.getContext(),
                                                    deviceTypeEnum);

    // omp.groupprivate takes a flat symbol reference and returns
    // the address of the per-contention group copy of the global variable.
    return mlir::omp::GroupprivateOp::create(
        firOpBuilder, currentLocation, global.resultType(), global.getSymbol(),
        deviceTypeAttr);
  };

  llvm::SetVector<const semantics::Symbol *> groupprivateSyms;
  converter.collectSymbolSet(eval, groupprivateSyms,
                             semantics::Symbol::Flag::OmpGroupPrivate,
                             /*collectSymbols=*/true,
                             /*collectHostAssociatedSymbols=*/true);
  llvm::SmallSet<semantics::SourceName, 8> groupprivateSymNames;

  // For a COMMON block, the GroupprivateOp is generated for the block itself
  // instead of its members.
  llvm::SmallPtrSet<const semantics::Symbol *, 8> commonSyms;

  for (const semantics::Symbol *sym : groupprivateSyms) {
    mlir::Value symGroupprivateValue;
    // The variable may be used more than once, and each reference has one
    // symbol with the same name. Only do once for references of one variable.
    if (!groupprivateSymNames.insert(sym->name()).second)
      continue;

    if (const semantics::Symbol *common =
            semantics::FindCommonBlockContaining(sym->GetUltimate())) {
      // Handle common block members: create groupprivate op for the entire
      // common block, then compute member offset.
      mlir::Value commonGroupprivateValue;
      if (commonSyms.contains(common)) {
        commonGroupprivateValue = converter.getSymbolAddress(*common);
      } else {
        commonGroupprivateValue = genGroupprivateOp(*common);
        if (!commonGroupprivateValue)
          continue;
        converter.bindSymbol(*common, commonGroupprivateValue);
        commonSyms.insert(common);
      }
      symGroupprivateValue = lower::genCommonBlockMember(
          converter, currentLocation, sym->GetUltimate(),
          commonGroupprivateValue, common->size());
    } else {
      symGroupprivateValue = genGroupprivateOp(*sym);
    }

    if (!symGroupprivateValue)
      continue;

    fir::ExtendedValue sexv = converter.getSymbolExtendedValue(*sym);
    fir::ExtendedValue symGroupprivateExv =
        getExtendedValue(sexv, symGroupprivateValue);
    converter.bindSymbol(*sym, symGroupprivateExv);
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
  // The handling of linear symbols is deferred to the OpenMP IRBuilder,
  // which is responsible for all its aspects, including privatization.
  assert((converter.isPresentShallowLookup(*sym) ||
          sym->test(semantics::Symbol::Flag::OmpLinear)) &&
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
    llvm::SmallVectorImpl<Object> &useDeviceAddrObjects,
    llvm::SmallVectorImpl<mlir::Value> &useDevicePtrVars,
    llvm::SmallVectorImpl<Object> &useDevicePtrObjects) {
  // Iterate over our use_device_ptr list and shift all non-cptr arguments into
  // use_device_addr.
  auto *varIt = useDevicePtrVars.begin();
  auto *objectIt = useDevicePtrObjects.begin();
  while (varIt != useDevicePtrVars.end()) {
    if (fir::isa_builtin_cptr_type(fir::unwrapRefType(varIt->getType()))) {
      ++varIt;
      ++objectIt;
      continue;
    }

    useDeviceAddrVars.push_back(*varIt);
    useDeviceAddrObjects.push_back(*objectIt);

    varIt = useDevicePtrVars.erase(varIt);
    objectIt = useDevicePtrObjects.erase(objectIt);
  }
}

/// Extract the list of function and variable symbols affected by the given
/// 'declare target' directive and return the intended device type for them.
static void getDeclareTargetInfo(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OmpDeclareTargetDirective &construct,
    mlir::omp::DeclareTargetOperands &clauseOps,
    llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &symbolAndClause) {

  if (!construct.v.Arguments().v.empty()) {
    ObjectList objects{makeObjects(construct.v.Arguments(), semaCtx)};
    // Case: declare target(func, var1, var2)
    gatherFuncAndVarSyms(objects, mlir::omp::DeclareTargetCaptureClause::to,
                         symbolAndClause, /*automap=*/false);
  } else {
    List<Clause> clauses = makeClauses(construct.v.Clauses(), semaCtx);
    if (clauses.empty()) {
      Fortran::lower::pft::FunctionLikeUnit *owningProc =
          eval.getOwningProcedure();
      // Main programs are never device routines. Skip them so that a bare
      // '!$omp declare target' inside an interface body that lives in a named
      // main program does not incorrectly mark _QQmain as a device function.
      if (owningProc && !owningProc->isMainProgram()) {
        // Case: declare target, implicit capture of enclosing
        // function/subroutine.
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
    const parser::OmpDeclareTargetDirective &declareTargetConstruct,
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
    const parser::OmpDeclareTargetDirective &declareTargetConstruct) {
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
static void
genLoopVars(mlir::Operation *op, lower::AbstractConverter &converter,
            mlir::Location &loc, llvm::ArrayRef<const semantics::Symbol *> args,
            llvm::ArrayRef<std::pair<mlir::omp::BlockArgOpenMPOpInterface,
                                     const ObjectEntryBlockArgs &>>
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
  llvm::SmallVector<std::pair<Object, mlir::Value>> mappedReductionObjects;
  auto mapEquivalentReductionObjects =
      [&](const ObjectEntryBlockArgsEntry &entry) {
        for (auto [object, var] : llvm::zip(entry.objects, entry.vars)) {
          for (auto [mappedObject, mappedValue] :
               llvm::reverse(mappedReductionObjects)) {
            if (object.id() == mappedObject.id()) {
              mapper.map(var, mappedValue);
              break;
            }
          }
        }
      };
  auto rememberReductionObjects =
      [&](const ObjectEntryBlockArgsEntry &entry,
          llvm::ArrayRef<mlir::BlockArgument> args) {
        for (auto [object, arg] : llvm::zip(entry.objects, args))
          mappedReductionObjects.emplace_back(object, arg);
      };

  for (auto [argGeneratingOp, blockArgs] : wrapperArgs) {
    mapEquivalentReductionObjects(blockArgs.inReduction);
    mapEquivalentReductionObjects(blockArgs.reduction);
    mapEquivalentReductionObjects(blockArgs.taskReduction);

    for (mlir::OpOperand &operand : argGeneratingOp->getOpOperands())
      operand.set(mapper.lookupOrDefault(operand.get()));

    for (const auto [arg, var] : llvm::zip_equal(
             argGeneratingOp->getRegion(0).getArguments(), blockArgs.getVars()))
      mapper.map(var, arg);

    rememberReductionObjects(blockArgs.inReduction,
                             argGeneratingOp.getInReductionBlockArgs());
    rememberReductionObjects(blockArgs.reduction,
                             argGeneratingOp.getReductionBlockArgs());
    rememberReductionObjects(blockArgs.taskReduction,
                             argGeneratingOp.getTaskReductionBlockArgs());
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

static std::pair<mlir::omp::ClauseMapFlags, mlir::omp::VariableCaptureKind>
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

  mlir::omp::ClauseMapFlags mapFlag = mlir::omp::ClauseMapFlags::implicit;

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

    // Double check it's not part of a common block, and that the common block
    // isn't marked declare target.
    if (!declareTargetOp) {
      if (const semantics::Symbol *common =
              semantics::FindCommonBlockContaining(sym.GetUltimate())) {
        mlir::Operation *commonOp =
            mod.lookupSymbol(converter.mangleName(*common));
        declareTargetOp =
            llvm::dyn_cast_if_present<mlir::omp::DeclareTargetInterface>(
                commonOp);
      }
    }

    if (declareTargetOp && declareTargetOp.isDeclareTarget()) {
      if (declareTargetOp.getDeclareTargetCaptureClause() ==
              mlir::omp::DeclareTargetCaptureClause::link &&
          declareTargetOp.getDeclareTargetDeviceType() !=
              mlir::omp::DeclareTargetDeviceType::nohost) {
        mapFlag |= mlir::omp::ClauseMapFlags::to;
        mapFlag |= mlir::omp::ClauseMapFlags::from;
      }
    } else if (fir::isa_trivial(varType) || fir::isa_char(varType)) {
      // Scalars behave as if they were "firstprivate".
      // TODO: Handle objects that are shared/lastprivate or were listed
      // in an in_reduction clause.
      if (isLiteralType(varType)) {
        captureKind = mlir::omp::VariableCaptureKind::ByCopy;
      } else {
        mapFlag |= mlir::omp::ClauseMapFlags::to;
      }
    } else if (semantics::IsNamedConstant(sym)) {
      // Parameter constants should be mapped as read-only (to) since they
      // cannot be modified. Mapping them as tofrom would cause a crash when
      // trying to write back to read-only memory.
      mapFlag |= mlir::omp::ClauseMapFlags::to;
    } else if (!fir::isa_builtin_cptr_type(varType)) {
      mapFlag |= mlir::omp::ClauseMapFlags::to;
      mapFlag |= mlir::omp::ClauseMapFlags::from;
    }
    return std::make_pair(mapFlag, captureKind);
  }

  switch (implicitBehaviour) {
  case DefMap::ImplicitBehavior::Alloc:
    return std::make_pair(mlir::omp::ClauseMapFlags::storage,
                          mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::Firstprivate:
    TODO(loc, "Firstprivate is currently unsupported defaultmap behaviour");
    break;
  case DefMap::ImplicitBehavior::From:
    return std::make_pair(mapFlag |= mlir::omp::ClauseMapFlags::from,
                          mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::Present:
    return std::make_pair(mapFlag |= mlir::omp::ClauseMapFlags::present,
                          mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::To:
    return std::make_pair(mapFlag |= mlir::omp::ClauseMapFlags::to,
                          (fir::isa_trivial(varType) || fir::isa_char(varType))
                              ? mlir::omp::VariableCaptureKind::ByCopy
                              : mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::Tofrom:
    return std::make_pair(mapFlag |= mlir::omp::ClauseMapFlags::from |
                                     mlir::omp::ClauseMapFlags::to,
                          mlir::omp::VariableCaptureKind::ByRef);
    break;
  case DefMap::ImplicitBehavior::Default:
  case DefMap::ImplicitBehavior::None:
    llvm_unreachable(
        "Implicit None and Default behaviour should have been handled earlier");
    break;
  }

  return std::make_pair(mapFlag |= mlir::omp::ClauseMapFlags::from |
                                   mlir::omp::ClauseMapFlags::to,
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

  OpWithBodyGenInfo &setEntryBlockArgs(const ObjectEntryBlockArgs *value) {
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
  const ObjectEntryBlockArgs *blockArgs = nullptr;
  /// [in] if provided, it overrides the default op's region entry block
  /// creation.
  GenOMPRegionEntryCBFn genRegionEntryCB = nullptr;
  /// [in] if set to `true`, skip generating nested evaluations and dispatching
  /// any further leaf constructs.
  bool genSkeletonOnly = false;
  /// [in] enables handling of privatized variable unless set to `false`.
  bool privatize = true;
};

static mlir::Value getReductionOverrideValue(fir::FirOpBuilder &builder,
                                             mlir::Location loc,
                                             const Object *object,
                                             mlir::BlockArgument arg) {
  if (hlfir::isFortranEntityWithAttributes(arg))
    return arg;

  fir::FortranVariableFlagsAttr attributes;
  llvm::SmallVector<mlir::Value> typeParams;
  auto declareOp = hlfir::DeclareOp::create(
      builder, loc, arg, "omp.reduction.element", nullptr, typeParams, nullptr,
      nullptr, 0, attributes);
  return declareOp.getBase();
}

static void
addReductionObjectOverrides(fir::FirOpBuilder &builder, mlir::Location loc,
                            lower::ExprToValueMap &overrides,
                            const ObjectEntryBlockArgsEntry &entry,
                            llvm::ArrayRef<mlir::BlockArgument> blockArgs) {
  if (entry.objects.empty())
    return;

  for (auto pair : llvm::zip_equal(entry.objects, blockArgs)) {
    const Object &object = std::get<0>(pair);
    const mlir::BlockArgument &arg = std::get<1>(pair);
    if (!ReductionProcessor::isExpressionLoweredAsReductionObject(&object))
      continue;
    const SomeExpr *expr = &object.ref().value();

    // Evict any outer-scope entry for the same array element so the
    // innermost scope always wins regardless of DenseMap iteration order.
    llvm::SmallVector<const SomeExpr *> toEvict;
    for (auto [key, value] : overrides) {
      if (Fortran::lower::isEqual(key, expr)) {
        toEvict.push_back(key);
      }
    }
    for (const SomeExpr *key : toEvict) {
      overrides.erase(key);
    }

    overrides[expr] = getReductionOverrideValue(builder, loc, &object, arg);
  }
}

static const semantics::Symbol *getArrayElementSymbol(const SomeExpr &expr) {
  std::optional<Fortran::evaluate::DataRef> dataRef =
      Fortran::evaluate::ExtractDataRef(expr);
  if (!dataRef)
    return nullptr;

  if (const auto *arrayRef =
          std::get_if<Fortran::evaluate::ArrayRef>(&dataRef->u))
    return &arrayRef->GetLastSymbol();

  return nullptr;
}

static void
addSymbolAliases(llvm::SmallVectorImpl<const semantics::Symbol *> &aliases,
                 const semantics::Symbol *symbol) {
  aliases.push_back(symbol);
  aliases.push_back(&symbol->GetUltimate());
  if (const auto *hostAssoc =
          symbol->detailsIf<semantics::HostAssocDetails>()) {
    aliases.push_back(&hostAssoc->symbol());
    aliases.push_back(&hostAssoc->symbol().GetUltimate());
  }
}

struct ArrayElementReductionUseCollector {
  explicit ArrayElementReductionUseCollector(
      const llvm::DenseMap<const semantics::Symbol *, const semantics::Symbol *>
          &aliasToReductionSymbol,
      llvm::DenseMap<const semantics::Symbol *,
                     llvm::SmallVector<const SomeExpr *>>
          &reductionElementExprs)
      : aliasToReductionSymbol(aliasToReductionSymbol),
        reductionElementExprs(reductionElementExprs) {}

  const llvm::DenseMap<const semantics::Symbol *, const semantics::Symbol *>
      &aliasToReductionSymbol;
  llvm::DenseMap<const semantics::Symbol *, llvm::SmallVector<const SomeExpr *>>
      &reductionElementExprs;
  llvm::SmallPtrSet<const semantics::Symbol *, 16> seen;
  llvm::SmallPtrSet<const semantics::Symbol *, 16> uncovered;

  void classifyReductionElementUses(const SomeExpr &expr) {
    llvm::SmallPtrSet<const semantics::Symbol *, 4> exprCandidates;
    auto getReductionSymbol = [this](const semantics::Symbol &symbol) {
      auto it = aliasToReductionSymbol.find(&symbol);
      return it == aliasToReductionSymbol.end() ? nullptr : it->second;
    };
    for (const semantics::Symbol &symbol :
         Fortran::evaluate::CollectSymbols(expr))
      if (const semantics::Symbol *reductionSymbol = getReductionSymbol(symbol))
        exprCandidates.insert(reductionSymbol);
    if (exprCandidates.empty())
      return;

    auto isCoveredReductionUse =
        [this](const semantics::Symbol *reductionSymbol, const SomeExpr &expr) {
          auto it = reductionElementExprs.find(reductionSymbol);
          return it != reductionElementExprs.end() &&
                 llvm::any_of(it->second, [&](const SomeExpr *reductionExpr) {
                   return Fortran::lower::isEqual(&expr, reductionExpr);
                 });
        };
    llvm::SmallPtrSet<const semantics::Symbol *, 4> seenInExpr;
    for (const SomeExpr &designator :
         semantics::omp::GetTopLevelDesignators(expr)) {
      const semantics::Symbol *symbol = getArrayElementSymbol(designator);
      const semantics::Symbol *reductionSymbol =
          symbol ? getReductionSymbol(*symbol) : nullptr;
      if (!reductionSymbol)
        continue;

      if (isCoveredReductionUse(reductionSymbol, designator)) {
        seen.insert(reductionSymbol);
        seenInExpr.insert(reductionSymbol);
      } else {
        uncovered.insert(reductionSymbol);
      }
    }

    for (const semantics::Symbol *symbol : exprCandidates)
      if (!seenInExpr.contains(symbol))
        uncovered.insert(symbol);
  }

  template <typename T>
  bool Pre(const T &node) {
    if constexpr (parser::HasTypedExpr<T>::value) {
      if (const SomeExpr *expr = semantics::GetExpr(nullptr, node)) {
        classifyReductionElementUses(*expr);
        return false;
      }
    }
    return true;
  }

  bool Pre(const parser::Name &name) { return false; }

  template <typename T>
  void Post(const T &) {}
};

static llvm::SmallVector<const semantics::Symbol *>
getSymbolsCoveredByReductionElements(lower::pft::Evaluation &eval,
                                     llvm::ArrayRef<Object> reductionObjects) {
  llvm::DenseMap<const semantics::Symbol *, const semantics::Symbol *>
      aliasToReductionSymbol;
  llvm::DenseMap<const semantics::Symbol *, llvm::SmallVector<const SomeExpr *>>
      reductionElementExprs;
  for (const Object &object : reductionObjects) {
    if (!ReductionProcessor::isExpressionLoweredAsReductionObject(&object))
      continue;
    llvm::SmallVector<const semantics::Symbol *> aliases;
    addSymbolAliases(aliases, object.sym());
    for (const semantics::Symbol *alias : aliases)
      aliasToReductionSymbol[alias] = object.sym();
    reductionElementExprs[object.sym()].push_back(&*object.ref());
  }

  if (reductionElementExprs.empty())
    return {};

  ArrayElementReductionUseCollector collector(aliasToReductionSymbol,
                                              reductionElementExprs);
  eval.visit([&](const auto &node) { parser::Walk(node, collector); });

  llvm::SmallVector<const semantics::Symbol *> suppressList;
  for (auto &[symbol, exprs] : reductionElementExprs)
    if (collector.seen.contains(symbol) &&
        !collector.uncovered.contains(symbol))
      suppressList.push_back(symbol);
  return suppressList;
}

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
      genEntryBlock(firOpBuilder, info.blockArgs->asEntryBlockArgs(),
                    op.getRegion(0));
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
                llvm::omp::Association::LoopNest;
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

  // TODO: groupprivate is currently only materialised for `teams` constructs.
  if (info.dir == llvm::omp::Directive::OMPD_teams)
    groupprivatizeVars(info.converter, info.eval);

  if (!info.genSkeletonOnly) {
    lower::ExprToValueMap local;
    if (auto *old = info.converter.getExprOverrides())
      local.insert(old->begin(), old->end());
    if (info.blockArgs) {
      if (auto ompBlockArgOp =
              mlir::dyn_cast<mlir::omp::BlockArgOpenMPOpInterface>(op)) {
        addReductionObjectOverrides(firOpBuilder, info.loc, local,
                                    info.blockArgs->inReduction,
                                    ompBlockArgOp.getInReductionBlockArgs());
        addReductionObjectOverrides(firOpBuilder, info.loc, local,
                                    info.blockArgs->reduction,
                                    ompBlockArgOp.getReductionBlockArgs());
        addReductionObjectOverrides(firOpBuilder, info.loc, local,
                                    info.blockArgs->taskReduction,
                                    ompBlockArgOp.getTaskReductionBlockArgs());
      }
    }

    auto *old = info.converter.getExprOverrides();
    info.converter.overrideExprValues(local.empty() ? old : &local);

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

    info.converter.overrideExprValues(old);
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
    mlir::omp::TargetDataOp &dataOp, const ObjectEntryBlockArgs &args,
    const mlir::Location &currentLocation, const ConstructQueue &queue,
    ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  genEntryBlock(firOpBuilder, args.asEntryBlockArgs(), dataOp.getRegion());
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
    mlir::omp::TargetOp &targetOp, const ObjectEntryBlockArgs &args,
    const mlir::Location &currentLocation, const ConstructQueue &queue,
    ConstructQueue::const_iterator item, DataSharingProcessor &dsp) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);

  mlir::Region &region = targetOp.getRegion();
  genEntryBlock(firOpBuilder, args.asEntryBlockArgs(), region);
  bindEntryBlockArgs(converter, targetOp, args);
  if (HostEvalInfo *hostEvalInfo = getHostEvalInfoStackTop(converter))
    hostEvalInfo->bindOperands(argIface.getHostEvalBlockArgs());

  // If we map a common block using it's symbol e.g. map(tofrom: /common_block/)
  // and accessing its members within the target region, there is a large
  // chance we will end up with uses external to the region accessing the common
  // resolve these, we do so by generating new common block member accesses
  // within the region, binding them to the member symbol for the scope of the
  // region so that subsequent code generation within the region will utilise
  // our new member accesses we have created.
  genIntermediateCommonBlockAccessors(converter, currentLocation,
                                      argIface.getMapBlockArgs(),
                                      args.map.getSyms());

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
                         const ObjectEntryBlockArgs &args) {
  static_assert(
      OpTy::template hasTrait<mlir::omp::LoopWrapperInterface::Trait>(),
      "expected a loop wrapper");
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Create wrapper.
  auto op = OpTy::create(firOpBuilder, loc, clauseOps);

  // Create entry block with arguments.
  genEntryBlock(firOpBuilder, args.asEntryBlockArgs(), op.getRegion());

  return op;
}

//===----------------------------------------------------------------------===//
// Code generation functions for clauses
//===----------------------------------------------------------------------===//

static void genAllocateClauses(lower::AbstractConverter &converter,
                               semantics::SemanticsContext &semaCtx,
                               lower::StatementContext &stmtCtx,
                               const ObjectList &objects,
                               const List<Clause> &clauses, mlir::Location loc,
                               llvm::SmallVectorImpl<mlir::Value> &operandRange,
                               mlir::omp::AllocateDirOperands &clauseOps) {
  if (!objects.empty())
    genObjectList(objects, converter, operandRange);

  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAlign(clauseOps);
  cp.processAllocator(stmtCtx, clauseOps);
}

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

static void genLoopClauses(lower::AbstractConverter &converter,
                           semantics::SemanticsContext &semaCtx,
                           const List<Clause> &clauses, mlir::Location loc,
                           mlir::omp::LoopOperands &clauseOps,
                           llvm::SmallVectorImpl<Object> &reductionObjects) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processBind(clauseOps);
  cp.processOrder(clauseOps);
  cp.processReduction(loc, clauseOps, reductionObjects);
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
  cp.processSimd(clauseOps);
}

static void genParallelClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::ParallelOperands &clauseOps,
    llvm::SmallVectorImpl<Object> &reductionObjects) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_parallel, clauseOps);

  HostEvalInfo *hostEvalInfo = getHostEvalInfoStackTop(converter);
  if (!hostEvalInfo || !hostEvalInfo->apply(clauseOps))
    cp.processNumThreads(stmtCtx, clauseOps);

  cp.processProcBind(clauseOps);
  cp.processReduction(loc, clauseOps, reductionObjects);
}

static void genScanClauses(lower::AbstractConverter &converter,
                           semantics::SemanticsContext &semaCtx,
                           const List<Clause> &clauses, mlir::Location loc,
                           mlir::omp::ScanOperands &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processInclusive(loc, clauseOps);
  cp.processExclusive(loc, clauseOps);
}

static void
genSectionsClauses(lower::AbstractConverter &converter,
                   semantics::SemanticsContext &semaCtx,
                   const List<Clause> &clauses, mlir::Location loc,
                   mlir::omp::SectionsOperands &clauseOps,
                   llvm::SmallVectorImpl<Object> &reductionObjects) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processNowait(clauseOps);
  cp.processReduction(loc, clauseOps, reductionObjects);
  // TODO Support delayed privatization.
}

static void genSimdClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    const List<Clause> &clauses, mlir::Location loc,
    mlir::omp::SimdOperands &clauseOps,
    llvm::SmallVectorImpl<Object> &reductionObjects,
    llvm::DenseMap<const semantics::Symbol *, mlir::Value> *reductionVarCache =
        nullptr) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAligned(clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_simd, clauseOps);
  cp.processNontemporal(clauseOps);
  cp.processOrder(clauseOps);
  cp.processReduction(loc, clauseOps, reductionObjects, reductionVarCache);
  cp.processSafelen(clauseOps);
  cp.processSimdlen(clauseOps);
  cp.processLinear(clauseOps);
}

// SIMD construct may have implicit
// linear semantics on IV. Process the same here.
static void
genSimdImplicitLinear(lower::AbstractConverter &converter,
                      semantics::SemanticsContext &semaCtx,
                      mlir::omp::SimdOperands &clauseOps,
                      mlir::omp::LoopNestOperands loopNestClauseOps,
                      llvm::SmallVector<const semantics::Symbol *> iv) {

  // If the (standalone/composite) SIMD is enclosed within TARGET,
  // implicit linearization will cause invalid FIR due to
  // target operation `host_eval` argument's illegal use in omp.simd.
  // Hence skip implicit linearization if TARGET encloses the current
  // SIMD.
  auto *currentOp =
      converter.getFirOpBuilder().getInsertionBlock()->getParentOp();
  while (currentOp) {
    if (auto targetOp = mlir::dyn_cast<mlir::omp::TargetOp>(currentOp))
      return;
    currentOp = currentOp->getParentOp();
  }

  std::vector<mlir::Attribute> typeAttrs;
  std::vector<mlir::Attribute> linearModAttrs;
  // If attributes from explicit `linear(...)` clause are present,
  // carry them forward.
  if (clauseOps.linearVarTypes && !clauseOps.linearVarTypes.empty())
    typeAttrs.assign(clauseOps.linearVarTypes.begin(),
                     clauseOps.linearVarTypes.end());
  if (clauseOps.linearModifiers && !clauseOps.linearModifiers.empty())
    linearModAttrs.assign(clauseOps.linearModifiers.begin(),
                          clauseOps.linearModifiers.end());

  for (auto [loopVar, loopStep] : llvm::zip(iv, loopNestClauseOps.loopSteps)) {
    const mlir::Value variable = converter.getSymbolAddress(*loopVar);

    // If the loop variable is already linearized (through an explicit
    // `linear()` clause, skip.
    if (std::find(clauseOps.linearVars.begin(), clauseOps.linearVars.end(),
                  variable) != clauseOps.linearVars.end())
      continue;

    // TODO: Implicit linearization is skipped if iv is a pointer
    // or an allocatable, due to potential mismatch between the linear
    // variable type (example !fir.ref<!fir.box<!fir.heap<i32>>>)
    // and the linear step size (example: i64). Handle this type mismatch
    // gracefully.
    if (loopVar->test(Fortran::semantics::Symbol::Flag::OmpLinear) &&
        !(Fortran::semantics::IsAllocatableOrPointer(*loopVar) ||
          Fortran::semantics::IsAllocatableOrPointer(loopVar->GetUltimate()))) {
      mlir::Type ty = converter.genType(*loopVar);
      typeAttrs.push_back(mlir::TypeAttr::get(ty));
      if (semaCtx.langOptions().OpenMPVersion >= 52)
        linearModAttrs.push_back(mlir::omp::LinearModifierAttr::get(
            &converter.getMLIRContext(), mlir::omp::LinearModifier::val));
      else
        linearModAttrs.push_back(
            mlir::UnitAttr::get(&converter.getMLIRContext()));
      clauseOps.linearVars.push_back(variable);
      clauseOps.linearStepVars.push_back(loopStep);
    }
  }
  if (!typeAttrs.empty()) {
    clauseOps.linearVarTypes =
        mlir::ArrayAttr::get(&converter.getMLIRContext(), typeAttrs);
    clauseOps.linearModifiers =
        mlir::ArrayAttr::get(&converter.getMLIRContext(), linearModAttrs);
  }
}

static void genScopeClauses(lower::AbstractConverter &converter,
                            semantics::SemanticsContext &semaCtx,
                            const List<Clause> &clauses, mlir::Location loc,
                            mlir::omp::ScopeOperands &clauseOps,
                            llvm::SmallVectorImpl<Object> &reductionObjects) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processNowait(clauseOps);
  cp.processReduction(loc, clauseOps, reductionObjects);
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

static void
genTargetClauses(lower::AbstractConverter &converter,
                 semantics::SemanticsContext &semaCtx, lower::SymMap &symTable,
                 lower::StatementContext &stmtCtx, lower::pft::Evaluation &eval,
                 const List<Clause> &clauses, mlir::Location loc,
                 mlir::omp::TargetExtOperands &clauseOps,
                 DefaultMapsTy &defaultMaps,
                 llvm::SmallVectorImpl<Object> &hasDeviceAddrObjects,
                 llvm::SmallVectorImpl<Object> &isDevicePtrObjects,
                 llvm::SmallVectorImpl<Object> &mapObjects) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDefaultMap(stmtCtx, defaultMaps);
  cp.processDepend(symTable, stmtCtx, clauseOps);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processDynGroupprivate(stmtCtx, clauseOps);
  cp.processHasDeviceAddr(stmtCtx, clauseOps, hasDeviceAddrObjects);
  if (HostEvalInfo *hostEvalInfo = getHostEvalInfoStackTop(converter)) {
    // Only process host_eval if compiling for the host device.
    HostEvalVisitor visitor(converter, semaCtx, stmtCtx, loc, *hostEvalInfo);
    visitor.visit(eval);
    hostEvalInfo->collectValues(clauseOps.hostEvalVars);
  }
  cp.processIf(llvm::omp::Directive::OMPD_target, clauseOps);
  cp.processIsDevicePtr(stmtCtx, clauseOps, isDevicePtrObjects);
  cp.processMap(loc, stmtCtx, clauseOps, llvm::omp::Directive::OMPD_unknown,
                &mapObjects);
  cp.processNowait(clauseOps);
  cp.processThreadLimit(stmtCtx, clauseOps);
  cp.processTODO<clause::Allocate, clause::InReduction, clause::UsesAllocators>(
      loc, llvm::omp::Directive::OMPD_target);

  // `target private(..)` is only supported in delayed privatization mode.
  if (!enableDelayedPrivatization)
    cp.processTODO<clause::Firstprivate, clause::Private>(
        loc, llvm::omp::Directive::OMPD_target);
}

static void genTargetDataClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::TargetDataOperands &clauseOps,
    llvm::SmallVectorImpl<Object> &useDeviceAddrObjects,
    llvm::SmallVectorImpl<Object> &useDevicePtrObjects) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_target_data, clauseOps);
  cp.processMap(loc, stmtCtx, clauseOps);
  cp.processUseDeviceAddr(stmtCtx, clauseOps, useDeviceAddrObjects);
  cp.processUseDevicePtr(stmtCtx, clauseOps, useDevicePtrObjects);

  // This function implements the deprecated functionality of use_device_ptr
  // that allows users to provide non-CPTR arguments to it with the caveat
  // that the compiler will treat them as use_device_addr. A lot of legacy
  // code may still depend on this functionality, so we should support it
  // in some manner. We do so currently by simply shifting non-cptr operands
  // from the use_device_ptr lists into the use_device_addr lists.
  // TODO: Perhaps create a user provideable compiler option that will
  // re-introduce a hard-error rather than a warning in these cases.
  promoteNonCPtrUseDevicePtrArgsToUseDeviceAddr(
      clauseOps.useDeviceAddrVars, useDeviceAddrObjects,
      clauseOps.useDevicePtrVars, useDevicePtrObjects);
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

static void genTaskClauses(lower::AbstractConverter &converter,
                           semantics::SemanticsContext &semaCtx,
                           lower::SymMap &symTable,
                           lower::StatementContext &stmtCtx,
                           const List<Clause> &clauses, mlir::Location loc,
                           mlir::omp::TaskOperands &clauseOps,
                           llvm::SmallVectorImpl<Object> &inReductionObjects) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAffinity(clauseOps);
  cp.processAllocate(clauseOps);
  cp.processDepend(symTable, stmtCtx, clauseOps);
  cp.processFinal(stmtCtx, clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_task, clauseOps);
  cp.processInReduction(loc, clauseOps, inReductionObjects);
  cp.processMergeable(clauseOps);
  cp.processPriority(stmtCtx, clauseOps);
  cp.processUntied(clauseOps);
  cp.processDetach(clauseOps);
}

static void
genTaskgroupClauses(lower::AbstractConverter &converter,
                    semantics::SemanticsContext &semaCtx,
                    const List<Clause> &clauses, mlir::Location loc,
                    mlir::omp::TaskgroupOperands &clauseOps,
                    llvm::SmallVectorImpl<Object> &taskReductionObjects) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processTaskReduction(loc, clauseOps, taskReductionObjects);
}

static void genTaskloopClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::TaskloopContextOperands &clauseOps,
    llvm::SmallVectorImpl<Object> &reductionObjects,
    llvm::SmallVectorImpl<Object> &inReductionObjects) {

  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processFinal(stmtCtx, clauseOps);
  cp.processGrainsize(stmtCtx, clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_taskloop, clauseOps);
  cp.processInReduction(loc, clauseOps, inReductionObjects);
  cp.processMergeable(clauseOps);
  cp.processNogroup(clauseOps);
  cp.processNumTasks(stmtCtx, clauseOps);
  cp.processPriority(stmtCtx, clauseOps);
  cp.processReduction(loc, clauseOps, reductionObjects);
  cp.processUntied(clauseOps);
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

static void genTeamsClauses(lower::AbstractConverter &converter,
                            semantics::SemanticsContext &semaCtx,
                            lower::StatementContext &stmtCtx,
                            const List<Clause> &clauses, mlir::Location loc,
                            mlir::omp::TeamsOperands &clauseOps,
                            llvm::SmallVectorImpl<Object> &reductionObjects) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  // TODO: Only evaluate it here if it's not host-evaluated, like num_teams and
  // thread_limit.
  cp.processDynGroupprivate(stmtCtx, clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_teams, clauseOps);

  HostEvalInfo *hostEvalInfo = getHostEvalInfoStackTop(converter);
  if (!hostEvalInfo || !hostEvalInfo->apply(clauseOps)) {
    cp.processNumTeams(stmtCtx, clauseOps);
    cp.processThreadLimit(stmtCtx, clauseOps);
  }

  cp.processReduction(loc, clauseOps, reductionObjects);
  cp.processDynGroupprivate(stmtCtx, clauseOps);
  // TODO Support delayed privatization.
}

static void genWsloopClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::WsloopOperands &clauseOps,
    llvm::SmallVectorImpl<Object> &reductionObjects,
    llvm::DenseMap<const semantics::Symbol *, mlir::Value> *reductionVarCache =
        nullptr) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processNowait(clauseOps);
  cp.processOrder(clauseOps);
  cp.processOrdered(clauseOps);
  cp.processReduction(loc, clauseOps, reductionObjects, reductionVarCache);
  cp.processSchedule(stmtCtx, clauseOps);
  cp.processLinear(clauseOps);
}

//===----------------------------------------------------------------------===//
// Code generation functions for leaf constructs
//===----------------------------------------------------------------------===//
static mlir::omp::AllocateDirOp genAllocateDirOp(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, lower::pft::Evaluation &eval,
    mlir::Location loc, const ObjectList &objects, const ConstructQueue &queue,
    ConstructQueue::const_iterator item) {
  llvm::SmallVector<mlir::Value> operandRange;
  mlir::omp::AllocateDirOperands clauseOps;
  genAllocateClauses(converter, semaCtx, stmtCtx, objects, item->clauses, loc,
                     operandRange, clauseOps);

  auto allocDirOp = mlir::omp::AllocateDirOp::create(
      converter.getFirOpBuilder(), loc, operandRange, clauseOps.align,
      clauseOps.allocator);

  // Register a cleanup at the Fortran scope exit.
  fir::FirOpBuilder *builder = &converter.getFirOpBuilder();
  mlir::Value allocator = clauseOps.allocator;
  converter.getFctCtx().attachCleanup([builder, loc, operandRange,
                                       allocator]() {
    mlir::omp::AllocateFreeOp::create(*builder, loc, operandRange, allocator);
  });

  return allocDirOp;
}

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

static mlir::omp::LoopNestOp
genLoopNestOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue, ConstructQueue::const_iterator item,
              mlir::omp::LoopNestOperands &clauseOps,
              llvm::ArrayRef<const semantics::Symbol *> iv,
              llvm::ArrayRef<std::pair<mlir::omp::BlockArgOpenMPOpInterface,
                                       const ObjectEntryBlockArgs &>>
                  wrapperArgs,
              llvm::omp::Directive directive, DataSharingProcessor &dsp) {
  const lower::ExprToValueMap *oldOverrides = converter.getExprOverrides();
  lower::ExprToValueMap loopNestOverrides;
  auto ivCallback = [&](mlir::Operation *op) {
    genLoopVars(op, converter, loc, iv, wrapperArgs);
    if (oldOverrides)
      loopNestOverrides.insert(oldOverrides->begin(), oldOverrides->end());
    for (auto [argGeneratingOp, blockArgs] : wrapperArgs) {
      addReductionObjectOverrides(converter.getFirOpBuilder(), loc,
                                  loopNestOverrides, blockArgs.inReduction,
                                  argGeneratingOp.getInReductionBlockArgs());
      addReductionObjectOverrides(converter.getFirOpBuilder(), loc,
                                  loopNestOverrides, blockArgs.reduction,
                                  argGeneratingOp.getReductionBlockArgs());
      addReductionObjectOverrides(converter.getFirOpBuilder(), loc,
                                  loopNestOverrides, blockArgs.taskReduction,
                                  argGeneratingOp.getTaskReductionBlockArgs());
    }
    converter.overrideExprValues(
        loopNestOverrides.empty() ? oldOverrides : &loopNestOverrides);
    return llvm::SmallVector<const semantics::Symbol *>(iv);
  };

  uint64_t nestValue = getCollapseValue(item->clauses);
  nestValue = nestValue < iv.size() ? iv.size() : nestValue;
  auto *nestedEval = getCollapsedLoopEval(eval, nestValue);
  auto loopNestOp = genOpWithBody<mlir::omp::LoopNestOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, *nestedEval,
                        directive)
          .setClauses(&item->clauses)
          .setDataSharingProcessor(&dsp)
          .setGenRegionEntryCb(ivCallback),
      queue, item, clauseOps);
  converter.overrideExprValues(oldOverrides);
  return loopNestOp;
}

static mlir::omp::LoopOp
genLoopOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
          semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
          mlir::Location loc, const ConstructQueue &queue,
          ConstructQueue::const_iterator item) {
  mlir::omp::LoopOperands loopClauseOps;
  llvm::SmallVector<Object> loopReductionObjects;
  genLoopClauses(converter, semaCtx, item->clauses, loc, loopClauseOps,
                 loopReductionObjects);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           /*useDelayedPrivatization=*/true, symTable);
  dsp.processStep1(&loopClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);

  ObjectEntryBlockArgs loopArgs;
  loopArgs.priv.objects = makeObjects(dsp.getDelayedPrivSymbols());
  loopArgs.priv.vars = loopClauseOps.privateVars;
  loopArgs.reduction.objects = loopReductionObjects;
  loopArgs.reduction.vars = loopClauseOps.reductionVars;

  auto loopOp =
      genWrapperOp<mlir::omp::LoopOp>(converter, loc, loopClauseOps, loopArgs);
  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, item,
                loopNestClauseOps, iv, {{loopOp, loopArgs}},
                llvm::omp::Directive::OMPD_loop, dsp);
  return loopOp;
}

// ´nestedEval´ is the Evaluation of a children loop of ´eval´.
// In a regular OpenMP Construct Evaluation ´nestedEval´ is the only children.
// Can be retrieved with getNestedDoConstruct(Evaluation).
//   <<OpenMPConstruct>>
//     Loop
//   <<End OpenMPConstruct>>
//
// ´nestedEval´ is most useful in the case that ´eval´ contains a sequence
// of loops. Then this function generates Canonical loop nests for individual
// loops.
//   <<OpenMPConstruct>>
//     Loop 1
//     Loop 2
//   <<End OpenMPConstruct>>
//
static void genCanonicalLoopNest(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    lower::pft::Evaluation *nestedEval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item,
    size_t numLoops, llvm::SmallVectorImpl<mlir::omp::CanonicalLoopOp> &loops) {
  assert(loops.empty() && "Expecting empty list to fill");
  assert(numLoops >= 1 && "Expecting at least one loop");

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  mlir::omp::LoopRelatedClauseOps loopInfo;
  llvm::SmallVector<const semantics::Symbol *, 3> ivs;
  collectLoopRelatedInfo(converter, loc, eval, nestedEval, numLoops, loopInfo,
                         ivs);
  assert(ivs.size() == numLoops &&
         "Expected to parse as many loop variables as there are loops");

  // Steps that follow:
  // 1. Emit all of the loop's prologues (compute the tripcount)
  // 2. Emit omp.canonical_loop nested inside each other (iteratively)
  // 2.1. In the innermost omp.canonical_loop, emit the loop body prologue (in
  // the body callback)
  //
  // Since emitting prologues and body code is split, remember prologue values
  // for use when emitting the same loop's epilogues.
  llvm::SmallVector<mlir::Value> tripcounts;
  llvm::SmallVector<mlir::Value> clis;
  llvm::SmallVector<lower::pft::Evaluation *> evals;
  llvm::SmallVector<mlir::Type> loopVarTypes;
  llvm::SmallVector<mlir::Value> loopStepVars;
  llvm::SmallVector<mlir::Value> loopLBVars;
  llvm::SmallVector<mlir::Value> blockArgs;

  // Step 1: Loop prologues
  // Computing the trip count must happen before entering the outermost loop
  lower::pft::Evaluation *innermostEval = nestedEval;
  for ([[maybe_unused]] auto iv : ivs) {
    if (innermostEval->getIf<parser::DoConstruct>()->IsDoConcurrent()) {
      // OpenMP specifies DO CONCURRENT only with the `!omp loop` construct.
      // Will need to add special cases for this combination.
      TODO(loc, "DO CONCURRENT as canonical loop not supported");
    }

    auto &doLoopEval = innermostEval->getFirstNestedEvaluation();
    evals.push_back(innermostEval);

    // Get the loop bounds (and increment)
    // auto &doLoopEval = nestedEval.getFirstNestedEvaluation();
    auto *doStmt = doLoopEval.getIf<parser::NonLabelDoStmt>();
    assert(doStmt && "Expected do loop to be in the nested evaluation");
    auto &loopControl = std::get<std::optional<parser::LoopControl>>(doStmt->t);
    assert(loopControl.has_value());
    auto *bounds = std::get_if<parser::LoopControl::Bounds>(&loopControl->u);
    assert(bounds && "Expected bounds for canonical loop");
    lower::StatementContext stmtCtx;
    mlir::Value loopLBVar = fir::getBase(
        converter.genExprValue(*semantics::GetExpr(bounds->Lower()), stmtCtx));
    mlir::Value loopUBVar = fir::getBase(
        converter.genExprValue(*semantics::GetExpr(bounds->Upper()), stmtCtx));
    mlir::Value loopStepVar = [&]() {
      if (auto &step = bounds->Step()) {
        return fir::getBase(
            converter.genExprValue(*semantics::GetExpr(step), stmtCtx));
      }

      // If `step` is not present, assume it is `1`.
      auto intTy = firOpBuilder.getI32Type();
      return firOpBuilder.createIntegerConstant(loc, intTy, 1);
    }();

    // Get the integer kind for the loop variable and cast the loop bounds
    size_t loopVarTypeSize = bounds->Name().thing.symbol->GetUltimate().size();
    mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
    loopVarTypes.push_back(loopVarType);
    loopLBVar = firOpBuilder.createConvert(loc, loopVarType, loopLBVar);
    loopUBVar = firOpBuilder.createConvert(loc, loopVarType, loopUBVar);
    loopStepVar = firOpBuilder.createConvert(loc, loopVarType, loopStepVar);
    loopLBVars.push_back(loopLBVar);
    loopStepVars.push_back(loopStepVar);

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
    mlir::Value lb = mlir::arith::SelectOp::create(
        firOpBuilder, loc, isDownwards, loopUBVar, loopLBVar);
    mlir::Value ub = mlir::arith::SelectOp::create(
        firOpBuilder, loc, isDownwards, loopLBVar, loopUBVar);

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
    tripcounts.push_back(tripcount);

    // Create the CLI handle.
    auto newcli = mlir::omp::NewCliOp::create(firOpBuilder, loc);
    mlir::Value cli = newcli.getResult();
    clis.push_back(cli);

    innermostEval = &*std::next(innermostEval->getNestedEvaluations().begin());
  }

  // Step 2: Create nested canoncial loops
  for (auto i : llvm::seq<size_t>(numLoops)) {
    bool isInnermost = (i == numLoops - 1);
    mlir::Type loopVarType = loopVarTypes[i];
    mlir::Value tripcount = tripcounts[i];
    mlir::Value cli = clis[i];
    auto &&eval = evals[i];

    auto ivCallback = [&, i, isInnermost](mlir::Operation *op)
        -> llvm::SmallVector<const Fortran::semantics::Symbol *> {
      mlir::Region &region = op->getRegion(0);

      // Create the op's region skeleton (BB taking the iv as argument)
      firOpBuilder.createBlock(&region, {}, {loopVarType}, {loc});
      blockArgs.push_back(region.front().getArgument(0));

      // Step 2.1: Emit body prologue code
      // Compute the translation from logical iteration number to the value of
      // the loop's iteration variable only in the innermost body. Currently,
      // loop transformations do not allow any instruction between loops, but
      // this will change with
      if (isInnermost) {
        assert(blockArgs.size() == numLoops &&
               "Expecting all block args to have been collected by now");
        for (auto j : llvm::seq<size_t>(numLoops)) {
          mlir::Value natIterNum = fir::getBase(blockArgs[j]);
          mlir::Value scaled = mlir::arith::MulIOp::create(
              firOpBuilder, loc, natIterNum, loopStepVars[j]);
          mlir::Value userVal = mlir::arith::AddIOp::create(
              firOpBuilder, loc, loopLBVars[j], scaled);

          mlir::OpBuilder::InsertPoint insPt =
              firOpBuilder.saveInsertionPoint();
          firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());
          mlir::Type tempTy = converter.genType(*ivs[j]);
          firOpBuilder.restoreInsertionPoint(insPt);

          // Write the loop value into loop variable
          mlir::Value cvtVal = firOpBuilder.createConvert(loc, tempTy, userVal);
          hlfir::Entity lhs{converter.getSymbolAddress(*ivs[j])};
          lhs = hlfir::derefPointersAndAllocatables(loc, firOpBuilder, lhs);
          mlir::Operation *storeOp =
              hlfir::AssignOp::create(firOpBuilder, loc, cvtVal, lhs);
          firOpBuilder.setInsertionPointAfter(storeOp);
        }
      }

      return {ivs[i]};
    };

    // Create the omp.canonical_loop operation
    auto opGenInfo = OpWithBodyGenInfo(converter, symTable, semaCtx, loc, *eval,
                                       llvm::omp::Directive::OMPD_unknown)
                         .setGenSkeletonOnly(!isInnermost)
                         .setClauses(&item->clauses)
                         .setPrivatize(false)
                         .setGenRegionEntryCb(ivCallback);
    auto canonLoop = genOpWithBody<mlir::omp::CanonicalLoopOp>(
        std::move(opGenInfo), queue, item, tripcount, cli);
    loops.push_back(canonLoop);

    // Insert next loop nested inside last loop
    firOpBuilder.setInsertionPoint(
        canonLoop.getRegion().back().getTerminator());
  }

  firOpBuilder.setInsertionPointAfter(loops.front());
}

static void genInterchangeOp(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::SymMap &symTable,
                             lower::StatementContext &stmtCtx,
                             Fortran::semantics::SemanticsContext &semaCtx,
                             Fortran::lower::pft::Evaluation &eval,
                             mlir::Location loc, const ConstructQueue &queue,
                             ConstructQueue::const_iterator item) {
  TODO(converter.getCurrentLocation(), "OpenMP Interchange");
}

static void genTileOp(Fortran::lower::AbstractConverter &converter,
                      Fortran::lower::SymMap &symTable,
                      lower::StatementContext &stmtCtx,
                      Fortran::semantics::SemanticsContext &semaCtx,
                      Fortran::lower::pft::Evaluation &eval, mlir::Location loc,
                      const ConstructQueue &queue,
                      ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  mlir::omp::SizesClauseOps sizesClause;
  ClauseProcessor cp(converter, semaCtx, item->clauses);
  cp.processSizes(stmtCtx, sizesClause);

  size_t numLoops = sizesClause.sizes.size();
  llvm::SmallVector<mlir::omp::CanonicalLoopOp, 3> canonLoops;
  canonLoops.reserve(numLoops);

  genCanonicalLoopNest(converter, symTable, semaCtx, eval,
                       getNestedDoConstruct(eval), loc, queue, item, numLoops,
                       canonLoops);
  assert((canonLoops.size() == numLoops) &&
         "Expecting the predetermined number of loops");

  llvm::SmallVector<mlir::Value, 3> applyees;
  applyees.reserve(numLoops);
  for (mlir::omp::CanonicalLoopOp l : canonLoops)
    applyees.push_back(l.getCli());

  // Emit the associated loops and create a CLI for each affected loop
  llvm::SmallVector<mlir::Value, 3> gridGeneratees;
  llvm::SmallVector<mlir::Value, 3> intratileGeneratees;
  gridGeneratees.reserve(numLoops);
  intratileGeneratees.reserve(numLoops);
  for ([[maybe_unused]] auto i : llvm::seq<int>(0, sizesClause.sizes.size())) {
    auto gridCLI = mlir::omp::NewCliOp::create(firOpBuilder, loc);
    gridGeneratees.push_back(gridCLI.getResult());
    auto intratileCLI = mlir::omp::NewCliOp::create(firOpBuilder, loc);
    intratileGeneratees.push_back(intratileCLI.getResult());
  }

  llvm::SmallVector<mlir::Value, 6> generatees;
  generatees.reserve(2 * numLoops);
  generatees.append(gridGeneratees);
  generatees.append(intratileGeneratees);

  mlir::omp::TileOp::create(firOpBuilder, loc, generatees, applyees,
                            sizesClause.sizes);
}

static void genFuseOp(Fortran::lower::AbstractConverter &converter,
                      Fortran::lower::SymMap &symTable,
                      lower::StatementContext &stmtCtx,
                      Fortran::semantics::SemanticsContext &semaCtx,
                      Fortran::lower::pft::Evaluation &eval, mlir::Location loc,
                      const ConstructQueue &queue,
                      ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  int64_t count = 0;
  mlir::omp::LooprangeClauseOps looprangeClause;
  ClauseProcessor cp(converter, semaCtx, item->clauses);
  bool looprange = cp.processLooprange(stmtCtx, looprangeClause, count);
  cp.processTODO<clause::Depth>(loc, llvm::omp::Directive::OMPD_fuse);

  llvm::SmallVector<mlir::Value> applyees;
  for (auto &child : eval.getNestedEvaluations()) {
    // Skip any Compiler Directive
    if (child.getIf<parser::CompilerDirective>())
      continue;

    // Emit the associated loop
    llvm::SmallVector<mlir::omp::CanonicalLoopOp> canonLoops;
    genCanonicalLoopNest(converter, symTable, semaCtx, eval, &child, loc, queue,
                         item, 1, canonLoops);

    auto cli = llvm::getSingleElement(canonLoops).getCli();
    applyees.push_back(cli);
  }
  // One generated loop + one for each loop not inside the specified looprange
  // if present
  llvm::SmallVector<mlir::Value> generatees;
  int64_t numGeneratees = !looprange ? 1 : applyees.size() - count + 1;
  for (int i = 0; i < numGeneratees; i++) {
    auto fusedCLI = mlir::omp::NewCliOp::create(firOpBuilder, loc);
    generatees.push_back(fusedCLI);
  }

  mlir::omp::FuseOp::create(firOpBuilder, loc, generatees, applyees,
                            looprangeClause.first, looprangeClause.count);
}

static void genUnrollOp(Fortran::lower::AbstractConverter &converter,
                        Fortran::lower::SymMap &symTable,
                        lower::StatementContext &stmtCtx,
                        Fortran::semantics::SemanticsContext &semaCtx,
                        Fortran::lower::pft::Evaluation &eval,
                        mlir::Location loc, const ConstructQueue &queue,
                        ConstructQueue::const_iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Clauses for unrolling not yet implemnted
  ClauseProcessor cp(converter, semaCtx, item->clauses);
  cp.processTODO<clause::Partial, clause::Full>(
      loc, llvm::omp::Directive::OMPD_unroll);

  // Emit the associated loop
  llvm::SmallVector<mlir::omp::CanonicalLoopOp, 1> canonLoops;
  genCanonicalLoopNest(converter, symTable, semaCtx, eval,
                       getNestedDoConstruct(eval), loc, queue, item, 1,
                       canonLoops);

  llvm::SmallVector<mlir::Value, 1> applyees;
  for (auto &&canonLoop : canonLoops)
    applyees.push_back(canonLoop.getCli());

  // Apply unrolling to it
  auto cli = llvm::getSingleElement(canonLoops).getCli();
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
              const ObjectEntryBlockArgs &args, DataSharingProcessor *dsp,
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
  llvm::SmallVector<Object> reductionObjects;
  genSectionsClauses(converter, semaCtx, item->clauses, loc, clauseOps,
                     reductionObjects);

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
  ObjectEntryBlockArgs args;
  // TODO: Add private syms and vars.
  args.reduction.objects = reductionObjects;
  args.reduction.vars = clauseOps.reductionVars;

  genEntryBlock(builder, args.asEntryBlockArgs(), sectionsOp.getRegion());
  mlir::Operation *terminator =
      lower::genOpenMPTerminator(builder, sectionsOp, loc);

  // Generate nested SECTION constructs.
  // This is done here rather than in genOMP([...], OmpSectionDirective )
  // because we need to run genReductionVars on each omp.section so that the
  // reduction variable gets mapped to the private version
  for (auto [construct, nestedEval] :
       llvm::zip(sectionBlocks, eval.getNestedEvaluations())) {
    const auto *sectionConstruct =
        std::get_if<parser::OmpSectionDirective>(&construct.u);
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

static mlir::omp::ScopeOp
genScopeOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
           semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
           mlir::Location loc, const ConstructQueue &queue,
           ConstructQueue::const_iterator item) {
  lower::SymMapScope scope(symTable);
  mlir::omp::ScopeOperands clauseOps;
  llvm::SmallVector<Object> reductionObjects;
  genScopeClauses(converter, semaCtx, item->clauses, loc, clauseOps,
                  reductionObjects);

  std::optional<DataSharingProcessor> dsp;
  if (enableDelayedPrivatization) {
    dsp.emplace(converter, semaCtx, item->clauses, eval,
                lower::omp::isLastItemInQueue(item, queue),
                /*useDelayedPrivatization=*/true, symTable);
    dsp->processStep1(&clauseOps);
  }

  ObjectEntryBlockArgs args;
  if (dsp)
    args.priv.objects = makeObjects(dsp->getDelayedPrivSymbols());
  args.priv.vars = clauseOps.privateVars;
  args.reduction.objects = reductionObjects;
  args.reduction.vars = clauseOps.reductionVars;

  return genOpWithBody<mlir::omp::ScopeOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_scope)
          .setClauses(&item->clauses)
          .setEntryBlockArgs(&args)
          .setDataSharingProcessor(enableDelayedPrivatization ? &dsp.value()
                                                              : nullptr),
      queue, item, clauseOps);
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
    llvm::ArrayRef<Object> hasDevObjects, llvm::ArrayRef<Object> mappedObjects,
    llvm::ArrayRef<Object> isDevicePtrObjects) {
  llvm::SmallVector<const semantics::Symbol *> concatSyms;
  concatSyms.reserve(privatizedSyms.size() + hasDevObjects.size() +
                     mappedObjects.size() + isDevicePtrObjects.size());
  concatSyms.append(privatizedSyms.begin(), privatizedSyms.end());
  llvm::transform(hasDevObjects, std::back_inserter(concatSyms),
                  [](const Object &object) { return object.sym(); });
  llvm::transform(mappedObjects, std::back_inserter(concatSyms),
                  [](const Object &object) { return object.sym(); });
  llvm::transform(isDevicePtrObjects, std::back_inserter(concatSyms),
                  [](const Object &object) { return object.sym(); });

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

// Visitor to collect symbols that have dynamic substring accesses
struct DynamicSubstringVisitor {
  llvm::SmallPtrSet<const semantics::Symbol *, 8> symbolsWithDynamicSubstring;
  semantics::SemanticsContext &semaCtx;

  explicit DynamicSubstringVisitor(semantics::SemanticsContext &ctx)
      : semaCtx(ctx) {}

  template <typename T>
  bool Pre(const T &) {
    return true;
  }
  template <typename T>
  void Post(const T &) {}

  // Check each expression for substring access
  void Post(const parser::Expr &expr) {
    if (const auto *typedExpr = semantics::GetExpr(semaCtx, expr)) {
      // Try to extract a substring from this expression
      if (auto substring = Fortran::evaluate::ExtractSubstring(*typedExpr)) {
        // Check if the substring has non-constant (dynamic) indices
        bool hasDynamicIndex = false;

        // Check if explicit lower bound exists and is non-constant
        if (const auto *lowerExpr = substring->GetLower()) {
          if (!Fortran::evaluate::ToInt64(*lowerExpr))
            hasDynamicIndex = true;
        }

        // Check if explicit upper bound exists and is non-constant
        if (const auto *upperExpr = substring->GetUpper()) {
          if (!Fortran::evaluate::ToInt64(*upperExpr))
            hasDynamicIndex = true;
        }

        // If we have dynamic indices, extract the base symbol
        if (hasDynamicIndex) {
          if (auto dataRef =
                  Fortran::evaluate::ExtractSubstringBase(*substring)) {
            if (const auto *symRef =
                    std::get_if<Fortran::evaluate::SymbolRef>(&dataRef->u)) {
              symbolsWithDynamicSubstring.insert(&symRef->get());
            }
          }
        }
      }
    }
  }
};

// Collect symbols that have dynamic substring accesses in the target region
static void collectSymbolsWithDynamicSubstring(
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    llvm::SmallPtrSet<const semantics::Symbol *, 8>
        &symbolsWithDynamicSubstring) {
  DynamicSubstringVisitor visitor(semaCtx);
  eval.visit([&](const auto &node) { parser::Walk(node, visitor); });
  symbolsWithDynamicSubstring = visitor.symbolsWithDynamicSubstring;
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

  mlir::omp::TargetExtOperands clauseOps;
  DefaultMapsTy defaultMaps;
  llvm::SmallVector<Object> mapObjects, hasDeviceAddrObjects,
      isDevicePtrObjects;
  genTargetClauses(converter, semaCtx, symTable, stmtCtx, eval, item->clauses,
                   loc, clauseOps, defaultMaps, hasDeviceAddrObjects,
                   isDevicePtrObjects, mapObjects);

  KernelTypeVisitor visitor(semaCtx, converter.getModuleOp());
  clauseOps.kernelType = mlir::omp::TargetExecModeAttr::get(
      &converter.getMLIRContext(), visitor.getKernelType(eval));

  if (!isDevicePtrObjects.empty()) {
    // is_device_ptr maps get duplicated so the clause and synthesized
    // has_device_addr entry each own a unique MapInfoOp user, keeping
    // MapInfoFinalization happy while still wiring the symbol into
    // has_device_addr when the user didn’t spell it explicitly.
    auto insertionPt = firOpBuilder.saveInsertionPoint();
    auto alreadyPresent = [&](const semantics::Symbol *sym) {
      return llvm::any_of(hasDeviceAddrObjects, [&](const Object &object) {
        const semantics::Symbol *objectSym = object.sym();
        return objectSym && sym &&
               objectSym->GetUltimate() == sym->GetUltimate();
      });
    };

    for (auto [idx, object] : llvm::enumerate(isDevicePtrObjects)) {
      const semantics::Symbol *sym = object.sym();
      mlir::Value mapVal = clauseOps.isDevicePtrVars[idx];
      assert(sym && "expected symbol for is_device_ptr");
      assert(mapVal && "expected map value for is_device_ptr");
      auto mapInfo = mapVal.getDefiningOp<mlir::omp::MapInfoOp>();
      assert(mapInfo && "expected map info op");

      if (!alreadyPresent(sym)) {
        clauseOps.hasDeviceAddrVars.push_back(mapVal);
        hasDeviceAddrObjects.push_back(object);
      }

      firOpBuilder.setInsertionPointAfter(mapInfo);
      mlir::Operation *clonedOp = firOpBuilder.clone(*mapInfo.getOperation());
      auto clonedMapInfo = mlir::cast<mlir::omp::MapInfoOp>(clonedOp);
      clauseOps.isDevicePtrVars[idx] = clonedMapInfo.getResult();
    }
    firOpBuilder.restoreInsertionPoint(insertionPt);
  }

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/
                           lower::omp::isLastItemInQueue(item, queue),
                           /*useDelayedPrivatization=*/true, symTable,
                           /*isTargetPrivitization=*/true);
  dsp.processStep1(&clauseOps);

  // Collect symbols that have dynamic substring accesses
  llvm::SmallPtrSet<const semantics::Symbol *, 8> symbolsWithDynamicSubstring;
  collectSymbolsWithDynamicSubstring(semaCtx, eval,
                                     symbolsWithDynamicSubstring);

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
      if (llvm::any_of(mapObjects, [=](const Object &object) {
            return object.sym() == common;
          }))
        return;

    // If we come across a symbol without a symbol address, we
    // return as we cannot process it, this is intended as a
    // catch all early exit for symbols that do not have a
    // corresponding extended value. Such as subroutines,
    // interfaces and named blocks.
    if (!converter.getSymbolAddress(sym))
      return;

    // Skip scalar parameters/constants as they do not need to be mapped.
    // However, parameter arrays must be mapped as they may be accessed with
    // dynamic indices on the device (e.g., const_array(runtime_index)).
    // Also, character scalar parameters must be mapped if they have dynamic
    // substring access.
    if (semantics::IsNamedConstant(sym) && sym.Rank() == 0 &&
        !symbolsWithDynamicSubstring.contains(&sym.GetUltimate()))
      return;

    // Skip groupprivate symbols - they don't need to be mapped because
    // groupprivate creates its own storage.
    if (sym.GetUltimate().test(semantics::Symbol::Flag::OmpGroupPrivate))
      return;

    if (!isDuplicateMappedSymbol(sym, dsp.getAllSymbolsToPrivatize(),
                                 hasDeviceAddrObjects, mapObjects,
                                 isDevicePtrObjects)) {
      if (const auto *details =
              sym.template detailsIf<semantics::HostAssocDetails>())
        converter.copySymbolBinding(details->symbol(), sym);
      std::stringstream name;
      fir::ExtendedValue dataExv = converter.getSymbolExtendedValue(sym);
      name << sym.name().ToString();

      fir::factory::AddrAndBoundsInfo info =
          Fortran::lower::getDataOperandBaseAddr(
              converter, firOpBuilder, sym.GetUltimate(),
              converter.getCurrentLocation());
      llvm::SmallVector<mlir::Value> bounds =
          fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
                                             mlir::omp::MapBoundsType>(
              firOpBuilder, info, dataExv,
              semantics::IsAssumedSizeArray(sym.GetUltimate()),
              converter.getCurrentLocation());
      mlir::Value baseOp = info.rawInput;
      mlir::Type eleType = baseOp.getType();
      if (auto refType = mlir::dyn_cast<fir::ReferenceType>(baseOp.getType()))
        eleType = refType.getElementType();

      std::pair<mlir::omp::ClauseMapFlags, mlir::omp::VariableCaptureKind>
          mapFlagAndKind = getImplicitMapTypeAndKind(
              firOpBuilder, converter, defaultMaps, eleType, loc, sym);

      mlir::FlatSymbolRefAttr mapperId;
      auto defaultmapBehaviour = getDefaultmapIfPresent(defaultMaps, eleType);
      if (defaultmapBehaviour ==
          clause::Defaultmap::ImplicitBehavior::Default) {
        const semantics::DerivedTypeSpec *typeSpec =
            sym.GetType() ? sym.GetType()->AsDerived() : nullptr;
        if (typeSpec) {
          std::string mapperIdName =
              typeSpec->name().ToString() + llvm::omp::OmpDefaultMapperName;
          if (auto *mapperSym =
                  converter.getCurrentScope().FindSymbol(mapperIdName))
            mapperIdName = converter.mangleName(
                mapperIdName, mapperSym->GetUltimate().owner());
          else
            mapperIdName =
                converter.mangleName(mapperIdName, *typeSpec->GetScope());

          if (!mapperIdName.empty()) {
            bool isPointer = semantics::IsPointer(sym);
            bool isAllocatable = semantics::IsAllocatable(sym);
            bool hasDefaultMapper =
                converter.getModuleOp().lookupSymbol(mapperIdName);
            // Avoid attaching implicit default mappers to pointer captures.
            // For large pointer-based derived aggregates this can over-map
            // nested payloads and conflict with explicit enter/exit maps.
            if (!isPointer && (hasDefaultMapper || isAllocatable)) {
              if (!hasDefaultMapper) {
                if (auto recordType = mlir::dyn_cast_or_null<fir::RecordType>(
                        converter.genType(*typeSpec)))
                  mapperId = getOrGenImplicitDefaultDeclareMapper(
                      converter.getFirOpBuilder(), loc, recordType,
                      mapperIdName,
                      [&](std::string &mapperIdName,
                          llvm::StringRef memberName) {
                        defaultMangler(converter, mapperIdName, memberName);
                      });
              } else {
                mapperId = mlir::FlatSymbolRefAttr::get(
                    &converter.getMLIRContext(), mapperIdName);
              }
            }
          }
        }
      }

      mlir::Value mapOp = createMapInfoOp(
          firOpBuilder, converter.getCurrentLocation(), baseOp,
          /*varPtrPtr=*/mlir::Value{}, name.str(), bounds, /*members=*/{},
          /*membersIndex=*/mlir::ArrayAttr{}, std::get<0>(mapFlagAndKind),
          std::get<1>(mapFlagAndKind), baseOp.getType(),
          /*partialMap=*/false, mapperId);

      clauseOps.mapVars.push_back(mapOp);
      mapObjects.push_back(
          Object{const_cast<semantics::Symbol *>(&sym), std::nullopt});
    }
  };
  lower::pft::visitAllSymbols(eval, captureImplicitMap);

  auto targetOp = mlir::omp::TargetOp::create(firOpBuilder, loc, clauseOps);

  llvm::SmallVector<mlir::Value> hasDeviceAddrBaseValues, mapBaseValues;
  extractMappedBaseValues(clauseOps.hasDeviceAddrVars, hasDeviceAddrBaseValues);
  extractMappedBaseValues(clauseOps.mapVars, mapBaseValues);

  ObjectEntryBlockArgs args;
  args.hasDeviceAddr.objects = hasDeviceAddrObjects;
  args.hasDeviceAddr.vars = hasDeviceAddrBaseValues;
  args.hostEvalVars = clauseOps.hostEvalVars;
  // TODO: Add in_reduction syms and vars.
  args.map.objects = mapObjects;
  args.map.vars = mapBaseValues;
  args.priv.objects = makeObjects(dsp.getDelayedPrivSymbols());
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
  llvm::SmallVector<Object> useDeviceAddrObjects, useDevicePtrObjects;
  genTargetDataClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                       clauseOps, useDeviceAddrObjects, useDevicePtrObjects);

  auto targetDataOp = mlir::omp::TargetDataOp::create(
      converter.getFirOpBuilder(), loc, clauseOps);

  llvm::SmallVector<mlir::Value> useDeviceAddrBaseValues,
      useDevicePtrBaseValues;
  extractMappedBaseValues(clauseOps.useDeviceAddrVars, useDeviceAddrBaseValues);
  extractMappedBaseValues(clauseOps.useDevicePtrVars, useDevicePtrBaseValues);

  ObjectEntryBlockArgs args;
  args.useDeviceAddr.objects = useDeviceAddrObjects;
  args.useDeviceAddr.vars = useDeviceAddrBaseValues;
  args.useDevicePtr.objects = useDevicePtrObjects;
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
  llvm::SmallVector<Object> inReductionObjects;
  genTaskClauses(converter, semaCtx, symTable, stmtCtx, item->clauses, loc,
                 clauseOps, inReductionObjects);

  if (!enableDelayedPrivatization)
    return genOpWithBody<mlir::omp::TaskOp>(
        OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                          llvm::omp::Directive::OMPD_task)
            .setClauses(&item->clauses),
        queue, item, clauseOps);

  llvm::SmallVector<const semantics::Symbol *>
      symbolsCoveredByReductionElements =
          getSymbolsCoveredByReductionElements(eval, inReductionObjects);
  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           lower::omp::isLastItemInQueue(item, queue),
                           /*useDelayedPrivatization=*/true, symTable,
                           /*isTargetPrivatization=*/false,
                           symbolsCoveredByReductionElements);
  dsp.processStep1(&clauseOps);

  ObjectEntryBlockArgs taskArgs;
  taskArgs.priv.objects = makeObjects(dsp.getDelayedPrivSymbols());
  taskArgs.priv.vars = clauseOps.privateVars;
  taskArgs.inReduction.objects = inReductionObjects;
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
  llvm::SmallVector<Object> taskReductionObjects;
  genTaskgroupClauses(converter, semaCtx, item->clauses, loc, clauseOps,
                      taskReductionObjects);

  ObjectEntryBlockArgs taskgroupArgs;
  taskgroupArgs.taskReduction.objects = taskReductionObjects;
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
  llvm::SmallVector<Object> reductionObjects;
  genTeamsClauses(converter, semaCtx, stmtCtx, item->clauses, loc, clauseOps,
                  reductionObjects);

  ObjectEntryBlockArgs args;
  // TODO: Add private syms and vars.
  args.reduction.objects = reductionObjects;
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
  // Dynamic private arrays cannot safely be allocated in GPU scratch when the
  // descriptor is captured through the distribute callback.
  dsp.setForceHeapAllocationForPrivateDynamicArrays();
  dsp.processStep1(&distributeClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);

  ObjectEntryBlockArgs distributeArgs;
  distributeArgs.priv.objects = makeObjects(dsp.getDelayedPrivSymbols());
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
  llvm::SmallVector<Object> wsloopReductionObjects;
  genWsloopClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                   wsloopClauseOps, wsloopReductionObjects);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           enableDelayedPrivatization, symTable);
  dsp.processStep1(&wsloopClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);

  ObjectEntryBlockArgs wsloopArgs;
  wsloopArgs.priv.objects = makeObjects(dsp.getDelayedPrivSymbols());
  wsloopArgs.priv.vars = wsloopClauseOps.privateVars;
  wsloopArgs.reduction.objects = wsloopReductionObjects;
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
  llvm::SmallVector<Object> parallelReductionObjects;
  genParallelClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                     parallelClauseOps, parallelReductionObjects);

  std::optional<DataSharingProcessor> dsp;
  if (enableDelayedPrivatization) {
    dsp.emplace(converter, semaCtx, item->clauses, eval,
                lower::omp::isLastItemInQueue(item, queue),
                /*useDelayedPrivatization=*/true, symTable);
    dsp->processStep1(&parallelClauseOps);
  }

  ObjectEntryBlockArgs parallelArgs;
  if (dsp)
    parallelArgs.priv.objects = makeObjects(dsp->getDelayedPrivSymbols());
  parallelArgs.priv.vars = parallelClauseOps.privateVars;
  parallelArgs.reduction.objects = parallelReductionObjects;
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
  llvm::SmallVector<Object> simdReductionObjects;
  genSimdClauses(converter, semaCtx, item->clauses, loc, simdClauseOps,
                 simdReductionObjects);

  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           enableDelayedPrivatization, symTable);
  dsp.processStep1(&simdClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);
  genSimdImplicitLinear(converter, semaCtx, simdClauseOps, loopNestClauseOps,
                        iv);

  ObjectEntryBlockArgs simdArgs;
  simdArgs.priv.objects = makeObjects(dsp.getDelayedPrivSymbols());
  simdArgs.priv.vars = simdClauseOps.privateVars;
  simdArgs.reduction.objects = simdReductionObjects;
  simdArgs.reduction.vars = simdClauseOps.reductionVars;
  auto simdOp =
      genWrapperOp<mlir::omp::SimdOp>(converter, loc, simdClauseOps, simdArgs);
  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, item,
                loopNestClauseOps, iv, {{simdOp, simdArgs}},
                llvm::omp::Directive::OMPD_simd, dsp);
  return simdOp;
}

static mlir::omp::TaskloopContextOp genStandaloneTaskloop(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    lower::StatementContext &stmtCtx, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval, mlir::Location loc,
    const ConstructQueue &queue, ConstructQueue::const_iterator item) {
  mlir::omp::TaskloopContextOperands taskloopClauseOps;
  llvm::SmallVector<Object> reductionObjects;
  llvm::SmallVector<Object> inReductionObjects;

  genTaskloopClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                     taskloopClauseOps, reductionObjects, inReductionObjects);
  llvm::SmallVector<Object> allReductionObjects;
  llvm::append_range(allReductionObjects, reductionObjects);
  llvm::append_range(allReductionObjects, inReductionObjects);
  llvm::SmallVector<const semantics::Symbol *>
      symbolsCoveredByReductionElements =
          getSymbolsCoveredByReductionElements(eval, allReductionObjects);
  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           enableDelayedPrivatization, symTable,
                           /*isTargetPrivatization=*/false,
                           symbolsCoveredByReductionElements);
  dsp.processStep1(&taskloopClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopNestClauseOps, iv);

  ObjectEntryBlockArgs taskloopArgs;
  taskloopArgs.priv.objects = makeObjects(dsp.getDelayedPrivSymbols());
  taskloopArgs.priv.vars = taskloopClauseOps.privateVars;
  taskloopArgs.reduction.objects = reductionObjects;
  taskloopArgs.reduction.vars = taskloopClauseOps.reductionVars;
  taskloopArgs.inReduction.objects = inReductionObjects;
  taskloopArgs.inReduction.vars = taskloopClauseOps.inReductionVars;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto taskLoopContextOp = mlir::omp::TaskloopContextOp::create(
      firOpBuilder, loc, taskloopClauseOps);
  // Create entry block with arguments.
  genEntryBlock(firOpBuilder, taskloopArgs.asEntryBlockArgs(),
                taskLoopContextOp.getRegion());

  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
  firOpBuilder.setInsertionPointToStart(&taskLoopContextOp.getRegion().front());
  mlir::omp::TaskloopWrapperOperands wrapperClauseOps;
  ObjectEntryBlockArgs wrapperEntryBlockArgs;
  auto taskLoopWrapperOp = genWrapperOp<mlir::omp::TaskloopWrapperOp>(
      converter, loc, wrapperClauseOps, wrapperEntryBlockArgs);

  genLoopNestOp(converter, symTable, semaCtx, eval, loc, queue, item,
                loopNestClauseOps, iv, {{taskLoopContextOp, taskloopArgs}},
                llvm::omp::Directive::OMPD_taskloop, dsp);

  firOpBuilder.setInsertionPointAfter(taskLoopWrapperOp);
  mlir::omp::TerminatorOp::create(firOpBuilder, loc);
  return taskLoopContextOp;
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
  llvm::SmallVector<Object> parallelReductionObjects;
  genParallelClauses(converter, semaCtx, stmtCtx, parallelItem->clauses, loc,
                     parallelClauseOps, parallelReductionObjects);

  DataSharingProcessor dsp(converter, semaCtx, doItem->clauses, eval,
                           /*shouldCollectPreDeterminedSymbols=*/true,
                           /*useDelayedPrivatization=*/true, symTable);
  dsp.setForceHeapAllocationForPrivateDynamicArrays();
  dsp.processStep1(&parallelClauseOps);

  ObjectEntryBlockArgs parallelArgs;
  parallelArgs.priv.objects = makeObjects(dsp.getDelayedPrivSymbols());
  parallelArgs.priv.vars = parallelClauseOps.privateVars;
  parallelArgs.reduction.objects = parallelReductionObjects;
  parallelArgs.reduction.vars = parallelClauseOps.reductionVars;
  genParallelOp(converter, symTable, semaCtx, eval, loc, queue, parallelItem,
                parallelClauseOps, parallelArgs, &dsp, /*isComposite=*/true);

  // Clause processing.
  mlir::omp::DistributeOperands distributeClauseOps;
  genDistributeClauses(converter, semaCtx, stmtCtx, distributeItem->clauses,
                       loc, distributeClauseOps);

  mlir::omp::WsloopOperands wsloopClauseOps;
  llvm::SmallVector<Object> wsloopReductionObjects;
  genWsloopClauses(converter, semaCtx, stmtCtx, doItem->clauses, loc,
                   wsloopClauseOps, wsloopReductionObjects);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, doItem->clauses, loc,
                     loopNestClauseOps, iv);

  // Operation creation.
  ObjectEntryBlockArgs distributeArgs;
  // TODO: Add private syms and vars.
  auto distributeOp = genWrapperOp<mlir::omp::DistributeOp>(
      converter, loc, distributeClauseOps, distributeArgs);
  distributeOp.setComposite(/*val=*/true);

  ObjectEntryBlockArgs wsloopArgs;
  // TODO: Add private syms and vars.
  wsloopArgs.reduction.objects = wsloopReductionObjects;
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
  llvm::SmallVector<Object> parallelReductionObjects;
  genParallelClauses(converter, semaCtx, stmtCtx, parallelItem->clauses, loc,
                     parallelClauseOps, parallelReductionObjects);

  DataSharingProcessor parallelItemDSP(
      converter, semaCtx, parallelItem->clauses, eval,
      /*shouldCollectPreDeterminedSymbols=*/false,
      /*useDelayedPrivatization=*/true, symTable);
  parallelItemDSP.processStep1(&parallelClauseOps);

  ObjectEntryBlockArgs parallelArgs;
  parallelArgs.priv.objects =
      makeObjects(parallelItemDSP.getDelayedPrivSymbols());
  parallelArgs.priv.vars = parallelClauseOps.privateVars;
  parallelArgs.reduction.objects = parallelReductionObjects;
  parallelArgs.reduction.vars = parallelClauseOps.reductionVars;
  genParallelOp(converter, symTable, semaCtx, eval, loc, queue, parallelItem,
                parallelClauseOps, parallelArgs, &parallelItemDSP,
                /*isComposite=*/true);

  // Clause processing.
  // Use a shared cache so that both wsloop and simd produce the same SSA
  // values for array/box reduction variables. See genCompositeDoSimd.
  llvm::DenseMap<const semantics::Symbol *, mlir::Value> reductionVarCache;

  mlir::omp::DistributeOperands distributeClauseOps;
  genDistributeClauses(converter, semaCtx, stmtCtx, distributeItem->clauses,
                       loc, distributeClauseOps);

  mlir::omp::WsloopOperands wsloopClauseOps;
  llvm::SmallVector<Object> wsloopReductionObjects;
  genWsloopClauses(converter, semaCtx, stmtCtx, doItem->clauses, loc,
                   wsloopClauseOps, wsloopReductionObjects, &reductionVarCache);

  mlir::omp::SimdOperands simdClauseOps;
  llvm::SmallVector<Object> simdReductionObjects;
  genSimdClauses(converter, semaCtx, simdItem->clauses, loc, simdClauseOps,
                 simdReductionObjects, &reductionVarCache);

  // Same as genCompositeDoSimd.
  if (!simdClauseOps.linearVars.empty()) {
    wsloopClauseOps.linearVars = std::move(simdClauseOps.linearVars);
    wsloopClauseOps.linearStepVars = std::move(simdClauseOps.linearStepVars);
    wsloopClauseOps.linearVarTypes = simdClauseOps.linearVarTypes;
    wsloopClauseOps.linearModifiers = simdClauseOps.linearModifiers;
    simdClauseOps.linearVars.clear();
    simdClauseOps.linearStepVars.clear();
    simdClauseOps.linearVarTypes = nullptr;
    simdClauseOps.linearModifiers = nullptr;
  }
  DataSharingProcessor simdItemDSP(converter, semaCtx, simdItem->clauses, eval,
                                   /*shouldCollectPreDeterminedSymbols=*/true,
                                   /*useDelayedPrivatization=*/true, symTable);
  simdItemDSP.processStep1(&simdClauseOps);

  mlir::omp::LoopNestOperands loopNestClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, simdItem->clauses, loc,
                     loopNestClauseOps, iv);
  genSimdImplicitLinear(converter, semaCtx, simdClauseOps, loopNestClauseOps,
                        iv);

  // Operation creation.
  ObjectEntryBlockArgs distributeArgs;
  // TODO: Add private syms and vars.
  auto distributeOp = genWrapperOp<mlir::omp::DistributeOp>(
      converter, loc, distributeClauseOps, distributeArgs);
  distributeOp.setComposite(/*val=*/true);

  ObjectEntryBlockArgs wsloopArgs;
  // TODO: Add private syms and vars.
  wsloopArgs.reduction.objects = wsloopReductionObjects;
  wsloopArgs.reduction.vars = wsloopClauseOps.reductionVars;
  auto wsloopOp = genWrapperOp<mlir::omp::WsloopOp>(
      converter, loc, wsloopClauseOps, wsloopArgs);
  wsloopOp.setComposite(/*val=*/true);

  ObjectEntryBlockArgs simdArgs;
  simdArgs.priv.objects = makeObjects(simdItemDSP.getDelayedPrivSymbols());
  simdArgs.priv.vars = simdClauseOps.privateVars;
  simdArgs.reduction.objects = simdReductionObjects;
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
  llvm::SmallVector<Object> simdReductionObjects;
  genSimdClauses(converter, semaCtx, simdItem->clauses, loc, simdClauseOps,
                 simdReductionObjects);

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
  genSimdImplicitLinear(converter, semaCtx, simdClauseOps, loopNestClauseOps,
                        iv);

  // Operation creation.
  ObjectEntryBlockArgs distributeArgs;
  distributeArgs.priv.objects =
      makeObjects(distributeItemDSP.getDelayedPrivSymbols());
  distributeArgs.priv.vars = distributeClauseOps.privateVars;
  auto distributeOp = genWrapperOp<mlir::omp::DistributeOp>(
      converter, loc, distributeClauseOps, distributeArgs);
  distributeOp.setComposite(/*val=*/true);

  ObjectEntryBlockArgs simdArgs;
  simdArgs.priv.objects = makeObjects(simdItemDSP.getDelayedPrivSymbols());
  simdArgs.priv.vars = simdClauseOps.privateVars;
  simdArgs.reduction.objects = simdReductionObjects;
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
  // Use a shared cache so that both wsloop and simd produce the same SSA
  // values for array/box reduction variables, enabling genLoopVars()'s
  // IRMapping to correctly chain the inner wrapper's operands to the outer
  // wrapper's block arguments.
  llvm::DenseMap<const semantics::Symbol *, mlir::Value> reductionVarCache;

  mlir::omp::WsloopOperands wsloopClauseOps;
  llvm::SmallVector<Object> wsloopReductionObjects;
  genWsloopClauses(converter, semaCtx, stmtCtx, doItem->clauses, loc,
                   wsloopClauseOps, wsloopReductionObjects, &reductionVarCache);

  mlir::omp::SimdOperands simdClauseOps;
  llvm::SmallVector<Object> simdReductionObjects;
  genSimdClauses(converter, semaCtx, simdItem->clauses, loc, simdClauseOps,
                 simdReductionObjects, &reductionVarCache);

  // omp.simd writes back linear vars unconditionally, causing a race when
  // inside a parallel region. Move them to wsloop which has proper last-iter
  // write-back guarded by a barrier.
  if (!simdClauseOps.linearVars.empty()) {
    wsloopClauseOps.linearVars = std::move(simdClauseOps.linearVars);
    wsloopClauseOps.linearStepVars = std::move(simdClauseOps.linearStepVars);
    wsloopClauseOps.linearVarTypes = simdClauseOps.linearVarTypes;
    wsloopClauseOps.linearModifiers = simdClauseOps.linearModifiers;
    simdClauseOps.linearVars.clear();
    simdClauseOps.linearStepVars.clear();
    simdClauseOps.linearVarTypes = nullptr;
    simdClauseOps.linearModifiers = nullptr;
  }
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
  genSimdImplicitLinear(converter, semaCtx, simdClauseOps, loopNestClauseOps,
                        iv);

  // Operation creation.
  ObjectEntryBlockArgs wsloopArgs;
  wsloopArgs.priv.objects = makeObjects(wsloopItemDSP.getDelayedPrivSymbols());
  wsloopArgs.priv.vars = wsloopClauseOps.privateVars;
  wsloopArgs.reduction.objects = wsloopReductionObjects;
  wsloopArgs.reduction.vars = wsloopClauseOps.reductionVars;
  auto wsloopOp = genWrapperOp<mlir::omp::WsloopOp>(
      converter, loc, wsloopClauseOps, wsloopArgs);
  wsloopOp.setComposite(/*val=*/true);

  ObjectEntryBlockArgs simdArgs;
  simdArgs.priv.objects = makeObjects(simdItemDSP.getDelayedPrivSymbols());
  simdArgs.priv.vars = simdClauseOps.privateVars;
  simdArgs.reduction.objects = simdReductionObjects;
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

static mlir::omp::TaskloopWrapperOp genCompositeTaskloopSimd(
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
                  llvm::omp::Association::LoopNest;
  if (loopLeaf) {
    symTable.pushScope();
    if (genOMPCompositeDispatch(converter, symTable, stmtCtx, semaCtx, eval,
                                loc, queue, item, newOp)) {
      symTable.popScope();
      finalizeStmtCtx();
      return;
    }
  }

  llvm::omp::Directive dir = item->id;
  switch (dir) {
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
    newOp = genSectionsOp(converter, symTable, semaCtx, eval, loc, queue, item);
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
  case llvm::omp::Directive::OMPD_interchange:
    genInterchangeOp(converter, symTable, stmtCtx, semaCtx, eval, loc, queue,
                     item);
    break;
  case llvm::omp::Directive::OMPD_tile:
    genTileOp(converter, symTable, stmtCtx, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_fuse:
    genFuseOp(converter, symTable, stmtCtx, semaCtx, eval, loc, queue, item);
    break;
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

  // Add the omp.combined attribute to eligible ops, including non-innermost
  // leafs of a combined construct and immediately nested block-associated
  // combinable constructs. SECTIONS, WORKSHARE and WORKDISTRIBUTE are skipped
  // due to only being able to appear as an innermost combined construct.
  if (!loopLeaf &&
      llvm::isa_and_present<mlir::omp::ComposableOpInterface>(newOp) &&
      !llvm::isa<mlir::omp::SectionsOp, mlir::omp::WorkshareOp,
                 mlir::omp::WorkdistributeOp>(newOp)) {
    bool isCombined = false;
    if (std::next(item) != queue.end()) {
      // Non-innermost leafs of a combined construct must always hold the
      // attribute.
      isCombined = true;
    } else if (lower::pft::Evaluation *nestedEval =
                   extractOnlyOmpNestedEval(eval)) {
      // Combinable constructs that are immediately nested with no other
      // statements or directives preventing them from being combined need the
      // attribute as well. Disallow block constructs that can only be outermost
      // leafs and loop transformation constructs.
      OmpDirectiveSet combinableDirs =
          (llvm::omp::blockConstructSet &
           ~OmpDirectiveSet{llvm::omp::Directive::OMPD_ordered,
                            llvm::omp::Directive::OMPD_scope,
                            llvm::omp::Directive::OMPD_taskgroup}) |
          (llvm::omp::loopConstructSet & ~llvm::omp::loopTransformationSet);
      const auto &ompEval = nestedEval->get<parser::OpenMPConstruct>();
      llvm::omp::Directive nestedDir =
          parser::omp::GetOmpDirectiveName(ompEval).v;
      llvm::omp::Directive firstLeafDir =
          llvm::omp::getLeafConstructsOrSelf(nestedDir).front();

      if (combinableDirs.test(firstLeafDir))
        isCombined = true;
    }
    if (isCombined)
      llvm::cast<mlir::omp::ComposableOpInterface>(newOp).setCombined(true);
  }
}

//===----------------------------------------------------------------------===//
// OpenMPDeclarativeConstruct visitors
//===----------------------------------------------------------------------===//
static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpUtilityDirective &);

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpAllocateDirective &allocate) {
  lower::StatementContext stmtCtx;
  ObjectList objects = makeObjects((allocate.BeginDir().Arguments()), semaCtx);
  const auto &clauseList = (allocate.BeginDir().Clauses());
  List<Clause> clauses = makeClauses(clauseList, semaCtx);
  mlir::Location loc = converter.genLocation(allocate.source);

  ConstructQueue queue{buildConstructQueue(
      converter.getFirOpBuilder().getModule(), semaCtx, eval, allocate.source,
      llvm::omp::Directive::OMPD_allocate, clauses)};

  genAllocateDirOp(converter, semaCtx, stmtCtx, eval, loc, objects, queue,
                   queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpAssumesDirective &assumesConstruct) {
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

static ReductionProcessor::GenCombinerCBTy processReductionCombiner(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, const clause::Combiner &combiner,
    const parser::OmpStylizedInstance &parserInst) {
  // Extract the typed assignment from the parser-level instance, if
  // the combiner is an assignment statement (as opposed to a call).
  const evaluate::Assignment *assign = nullptr;
  const auto &instance =
      std::get<parser::OmpStylizedInstance::Instance>(parserInst.t);
  if (const auto *assignStmt =
          std::get_if<parser::AssignmentStmt>(&instance.u)) {
    if (auto *wrapper = assignStmt->typedAssignment.get())
      if (wrapper->v)
        assign = &*wrapper->v;
  }
  ReductionProcessor::GenCombinerCBTy genCombinerCB;
  const StylizedInstance &inst = combiner.v.front();
  semantics::SomeExpr evalExpr = std::get<StylizedInstance::Instance>(inst.t);

  genCombinerCB = [&, evalExpr, assign](fir::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Type type,
                                        mlir::Value lhs, mlir::Value rhs,
                                        bool isByRef) {
    lower::SymMapScope scope(symTable);
    mlir::Value ompOutVar;
    for (const Object &object : std::get<StylizedInstance::Variables>(inst.t)) {
      mlir::Value addr = lhs;
      mlir::Type type = lhs.getType();
      std::string name = object.sym()->name().ToString();
      bool isRhs = name == "omp_in";
      if (isRhs) {
        addr = rhs;
        type = rhs.getType();
      }

      if (!fir::conformsWithPassByRef(type)) {
        addr = builder.createTemporary(loc, type);
        fir::StoreOp::create(builder, loc, isRhs ? rhs : lhs, addr);
      }
      fir::FortranVariableFlagsEnum extraFlags = {};
      fir::FortranVariableFlagsAttr attributes =
          Fortran::lower::translateSymbolAttributes(builder.getContext(),
                                                    *object.sym(), extraFlags);
      // For character types, we need to provide the length parameter
      llvm::SmallVector<mlir::Value> typeParams;
      if (hlfir::isFortranEntity(addr)) {
        hlfir::genLengthParameters(loc, builder, hlfir::Entity{addr},
                                   typeParams);
      }
      auto declareOp =
          hlfir::DeclareOp::create(builder, loc, addr, name, nullptr,
                                   typeParams, nullptr, nullptr, 0, attributes);
      if (name == "omp_out")
        ompOutVar = declareOp.getResult(0);
      symTable.addVariableDefinition(*object.sym(), declareOp);
    }

    // For derived types with a typed assignment available, use
    // hlfir::AssignOp or user-defined assignment directly instead of
    // trying to convert the expression to a value (which doesn't work
    // for record types).  Only take this path when the assignment RHS
    // itself is a derived type -- i.e. the combiner assigns to the whole
    // derived-type variable (e.g. omp_out = mycombine(omp_out, omp_in)).
    // When the combiner assigns to a component (e.g. omp_out%x = ...),
    // the RHS is a scalar intrinsic type and the existing convertExprToValue
    // path handles it correctly.
    bool rhsIsDerived =
        assign && assign->rhs.GetType() &&
        assign->rhs.GetType()->category() == common::TypeCategory::Derived;
    if (rhsIsDerived && isByRef &&
        mlir::isa<fir::RecordType>(fir::unwrapRefType(lhs.getType()))) {
      lower::StatementContext stmtCtx;
      hlfir::Entity lhsEntity{ompOutVar};
      hlfir::Entity rhsEntity = lower::convertExprToHLFIR(
          loc, converter, assign->rhs, symTable, stmtCtx);
      common::visit(
          common::visitors{
              [&](const evaluate::Assignment::Intrinsic &) {
                hlfir::AssignOp::create(builder, loc, rhsEntity, lhsEntity);
              },
              [&](const evaluate::ProcedureRef &procRef) {
                lower::convertUserDefinedAssignmentToHLFIR(
                    loc, converter, procRef, lhsEntity, rhsEntity, symTable);
              },
              [&](const auto &) {
                llvm_unreachable(
                    "Unexpected assignment type in reduction combiner");
              },
          },
          assign->u);
      stmtCtx.finalizeAndPop();
      mlir::omp::YieldOp::create(builder, loc, lhs);
      return;
    }

    lower::StatementContext stmtCtx;
    mlir::Value result = common::visit(
        common::visitors{
            [&](const evaluate::ProcedureRef &procRef) -> mlir::Value {
              convertCallToHLFIR(loc, converter, procRef, std::nullopt,
                                 symTable, stmtCtx);
              auto outVal = fir::LoadOp::create(builder, loc, ompOutVar);
              if (isByRef) {
                fir::StoreOp::create(builder, loc, outVal, lhs);
                return mlir::Value{};
              }
              return outVal;
            },
            [&](const auto &expr) -> mlir::Value {
              mlir::Value exprResult = fir::getBase(convertExprToValue(
                  loc, converter, evalExpr, symTable, stmtCtx));
              // Optional load may be generated if we get a reference to the
              // reduction type.
              if (auto refType = llvm::dyn_cast<fir::ReferenceType>(
                      exprResult.getType())) {
                mlir::Type expectedType =
                    isByRef ? fir::unwrapRefType(lhs.getType()) : lhs.getType();
                if (expectedType == refType.getElementType())
                  exprResult = fir::LoadOp::create(builder, loc, exprResult);
              }
              // For component-level derived-type combiners (e.g.
              // omp_out%x = omp_out%x + omp_in%x), the assignment was
              // not performed during expression lowering since
              // convertExprToValue only evaluates the RHS value.
              // The result type won't match the reduction variable type.
              // Use the typed assignment LHS to store to the correct
              // component, then skip the whole-variable store.
              if (isByRef &&
                  exprResult.getType() != fir::unwrapRefType(lhs.getType())) {
                if (assign) {
                  lower::StatementContext assignCtx;
                  hlfir::Entity lhsEntity = lower::convertExprToHLFIR(
                      loc, converter, assign->lhs, symTable, assignCtx);
                  hlfir::AssignOp::create(builder, loc, exprResult, lhsEntity);
                  assignCtx.finalizeAndPop();
                } else {
                  fir::StoreOp::create(builder, loc, exprResult, ompOutVar);
                }
                return mlir::Value{};
              }
              if (isByRef) {
                fir::StoreOp::create(builder, loc, exprResult, lhs);
                return mlir::Value{};
              }
              return exprResult;
            }},
        evalExpr.u);
    stmtCtx.finalizeAndPop();
    if (isByRef) {
      mlir::omp::YieldOp::create(builder, loc, lhs);
    } else {
      mlir::omp::YieldOp::create(builder, loc, result);
    }
  };
  return genCombinerCB;
}

// Checks that the reduction type is either a trivial type, a fixed-length
// character type, or a derived type composed of such types.
static bool isSimpleReductionType(mlir::Type reductionType) {
  if (fir::isa_trivial(reductionType))
    return true;
  // Fixed-length CHARACTER is not trivial but can be zero-initialized.
  // Reject dynamic-length CHARACTER (len == unknownLen()).
  if (auto charTy = mlir::dyn_cast<fir::CharacterType>(reductionType))
    return charTy.getLen() != fir::CharacterType::unknownLen();
  if (auto recordTy = mlir::dyn_cast<fir::RecordType>(reductionType)) {
    for (auto [_, fieldType] : recordTy.getTypeList()) {
      if (!isSimpleReductionType(fieldType))
        return false;
    }
    return true;
  }
  // Reject array and descriptor-based types.
  return false;
}

// Getting the type from a symbol compared to a DeclSpec is simpler since we do
// not need to consider derived vs intrinsic types. Semantics is guaranteed to
// generate these symbols.
static mlir::Type
getReductionType(lower::AbstractConverter &converter,
                 const parser::OmpReductionSpecifier &specifier) {
  const auto &combinerExpression =
      std::get<std::optional<parser::OmpCombinerExpression>>(specifier.t)
          .value();
  const parser::OmpStylizedInstance &combinerInstance =
      combinerExpression.v.front();
  const std::list<parser::OmpStylizedDeclaration> &declList =
      std::get<std::list<parser::OmpStylizedDeclaration>>(combinerInstance.t);
  const parser::OmpStylizedDeclaration &decl = declList.front();
  const auto &name = std::get<parser::ObjectName>(decl.var.t);
  const auto &symbol = semantics::SymbolRef(*name.symbol);
  mlir::Type reductionType = converter.genType(symbol);

  if (!isSimpleReductionType(reductionType))
    TODO(converter.getCurrentLocation(),
         "declare reduction currently only supports trivial types, "
         "fixed-length CHARACTER, or derived types containing them");
  return reductionType;
}

// Represent the reduction combiner as a clause, return reference to it.
// If there is a "combiner" clause already present, do nothing. Otherwise
// manufacture a combiner clause from the combiner expression on the reduction
// specifier and append it to the list of clauses.
static const clause::Combiner &
appendCombiner(const parser::OmpDeclareReductionDirective &construct,
               List<Clause> &clauses, semantics::SemanticsContext &semaCtx) {
  for (const Clause &clause : clauses) {
    if (clause.id == llvm::omp::Clause::OMPC_combiner)
      return std::get<clause::Combiner>(clause.u);
  }

  using namespace parser::omp;
  const parser::OmpDirectiveSpecification &dirSpec = construct.v;
  auto *specifier = GetFirstArgument<parser::OmpReductionSpecifier>(dirSpec);
  assert(specifier && "Expecting reduction specifier");
  if (auto *expr = GetCombinerExpr(*specifier)) {
    clause::Combiner combiner;
    for (const parser::OmpStylizedInstance &sinst : expr->v)
      combiner.v.push_back(makeStylizedInstance(sinst, semaCtx));
    clauses.push_back(makeClause(llvm::omp::Clause::OMPC_combiner,
                                 std::move(combiner), expr->source));
    return std::get<clause::Combiner>(clauses.back().u);
  }

  llvm_unreachable("Expecting reduction combiner");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpDeclareReductionDirective &construct) {
  if (semaCtx.langOptions().OpenMPSimd)
    return;

  const auto &specifier =
      DEREF(parser::omp::GetFirstArgument<parser::OmpReductionSpecifier>(
          construct.v));
  const auto &typeNameList = std::get<parser::OmpTypeNameList>(specifier.t);
  List<Clause> clauses = makeClauses(construct.v.Clauses(), semaCtx);
  const clause::Combiner &combiner =
      appendCombiner(construct, clauses, semaCtx);
  const auto &identifier =
      std::get<parser::OmpReductionIdentifier>(specifier.t);

  // Convert the parser-level reduction identifier to the clause-level
  // representation, then use ReductionProcessor to derive the canonical name.
  clause::ReductionOperator redOp =
      clause::makeReductionOperator(identifier, semaCtx);

  // Get the parser-level combiner expression so we can pass each
  // parser::OmpStylizedInstance to processReductionCombiner.
  // The combiner expression's instances correspond 1:1 to typeNameList entries.
  const auto *combinerExpr = parser::omp::GetCombinerExpr(specifier);
  assert(combinerExpr && "Expecting combiner expression");
  auto parserInstIt = combinerExpr->v.begin();

  // Get the parser-level initializer expression (if present) so we can
  // pass each parser::OmpStylizedInstance to processInitializer.
  const parser::OmpInitializerExpression *initExpr = nullptr;
  for (const auto &clause : construct.v.Clauses().v) {
    initExpr = parser::omp::GetInitializerExpr(clause);
    if (initExpr)
      break;
  }
  auto parserInitInstIt =
      initExpr ? initExpr->v.begin()
               : std::list<parser::OmpStylizedInstance>::const_iterator{};

  for (const auto &typeSpec : typeNameList.v) {
    (void)typeSpec; // Currently unused

    assert(parserInstIt != combinerExpr->v.end() &&
           "Mismatched combiner instance count");
    const parser::OmpStylizedInstance &parserInst = *parserInstIt++;

    mlir::Type reductionType = getReductionType(converter, specifier);
    bool isByRef = ReductionProcessor::doReductionByRef(reductionType);
    // Compute the canonical reduction name the same way
    // processReductionArguments does.
    std::string reductionNameStr = common::visit(
        common::visitors{
            [&](const clause::DefinedOperator &defOp) -> std::string {
              return common::visit(
                  common::visitors{
                      [&](const clause::DefinedOperator::IntrinsicOperator
                              &intrOp) -> std::string {
                        ReductionProcessor::ReductionIdentifier redId =
                            ReductionProcessor::getReductionType(intrOp);
                        return ReductionProcessor::getReductionName(
                            redId, converter.getFirOpBuilder().getKindMap(),
                            reductionType, isByRef);
                      },
                      [&](const clause::DefinedOperator::DefinedOpName &opName)
                          -> std::string {
                        // Directive side of the user-defined operator reduction
                        // naming contract (the clause side is in
                        // ReductionProcessor::processReductionArguments).
                        // opName.v.sym() is the reduction symbol
                        // "op<spelling>". Only single-declaration, single-type
                        // reductions are supported; otherwise emit a clean
                        // TODO.
                        const semantics::Symbol &redSym =
                            opName.v.sym()->GetUltimate();
                        const auto *userDetails =
                            redSym.detailsIf<semantics::UserReductionDetails>();
                        if (!userDetails || typeNameList.v.size() != 1 ||
                            userDetails->GetDeclList().size() != 1 ||
                            userDetails->GetTypeList().size() != 1)
                          TODO(converter.getCurrentLocation(),
                               "OpenMP user-defined operator declare reduction "
                               "with multiple declarations or multiple types");
                        return ReductionProcessor::getScopedUserReductionName(
                            converter, redSym);
                      },
                  },
                  defOp.u);
            },
            [&](const clause::ProcedureDesignator &pd) -> std::string {
              // Qualify the name with the scope in which the user-defined
              // reduction is declared so that reductions with the same name
              // in different scopes produce distinct omp.declare_reduction ops.
              const semantics::Symbol *sym = pd.v.sym();
              std::string name = sym->name().ToString();
              return converter.mangleName(name, sym->GetUltimate().owner());
            },
        },
        redOp.u);

    ReductionProcessor::GenCombinerCBTy genCombinerCB =
        processReductionCombiner(converter, symTable, semaCtx, combiner,
                                 parserInst);
    const parser::OmpStylizedInstance *parserInitInst = nullptr;
    if (initExpr) {
      assert(parserInitInstIt != initExpr->v.end() &&
             "Mismatched initializer instance count");
      parserInitInst = &*parserInitInstIt++;
    }

    // Get the omp_out symbol from the combiner. Used for finalization checks
    // in populateByRefInitAndCleanupRegions and for generating default
    // initialization via genScalarDefaultInitializerValue.
    const semantics::Symbol *reductionSym = nullptr;
    const auto &declList =
        std::get<std::list<parser::OmpStylizedDeclaration>>(parserInst.t);
    for (const auto &decl : declList) {
      const auto &name = std::get<parser::ObjectName>(decl.var.t);
      if (name.ToString() == "omp_out") {
        reductionSym = name.symbol;
        break;
      }
    }

    ReductionProcessor::GenInitValueCBTy genInitValueCB;
    ClauseProcessor cp(converter, semaCtx, clauses);
    if (!cp.processInitializer(symTable, genInitValueCB, parserInitInst)) {
      // No initializer clause provided. Per OpenMP, initialize as
      // default-initialized using the shared inline init helper.
      const semantics::DerivedTypeSpec *derivedTypeSpec = nullptr;
      if (const semantics::DeclTypeSpec *declTypeSpec = typeSpec.declTypeSpec)
        derivedTypeSpec = declTypeSpec->AsDerived();

      mlir::Type unwrappedType = fir::unwrapRefType(reductionType);
      if (fir::isa_trivial(unwrappedType)) {
        // Trivial types return the zero value directly (by-value init).
        genInitValueCB = [](fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Type type, mlir::Value,
                            mlir::Value) -> mlir::Value {
          mlir::Type ty = fir::unwrapRefType(type);
          if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
            ty = seqTy.getEleTy();
          else if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty)) {
            auto eleTy = fir::unwrapRefType(boxTy.getEleTy());
            if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(eleTy))
              ty = seqTy.getEleTy();
            else
              ty = eleTy;
          }
          return fir::ZeroOp::create(builder, loc, ty);
        };
      } else if (mlir::isa<fir::CharacterType>(unwrappedType) ||
                 fir::isa_derived(unwrappedType)) {
        // CHARACTER and derived types use by-ref init via the shared helper.
        genInitValueCB = [&converter, derivedTypeSpec, reductionSym](
                             fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Type type, mlir::Value,
                             mlir::Value) -> mlir::Value {
          mlir::Block *initBlock = builder.getInsertionBlock();
          mlir::Value privVar = initBlock->getArgument(1);
          lower::genInlineTypeDefaultInit(converter, builder, loc, type,
                                          privVar, derivedTypeSpec,
                                          reductionSym);
          return mlir::Value{};
        };
      } else {
        llvm_unreachable(
            "unhandled type in declare reduction without initializer");
      }
    }
    mlir::Type redType =
        isByRef
            ? static_cast<mlir::Type>(fir::ReferenceType::get(reductionType))
            : reductionType;

    ReductionProcessor::createDeclareReductionHelper<
        mlir::omp::DeclareReductionOp>(
        converter, reductionNameStr, redType, converter.getCurrentLocation(),
        isByRef, genCombinerCB, genInitValueCB, reductionSym);
  }
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OmpDeclareSimdDirective &declareSimdConstruct) {
  mlir::Location loc = converter.getCurrentLocation();
  const parser::OmpDirectiveSpecification &beginSpec = declareSimdConstruct.v;

  // A `declare simd` directive may appear in the specification part of an
  // interface body. In that case the PFT records the directive as an
  // evaluation of the enclosing program unit rather than of the interface
  // body's subprogram, and the clause operands (linear/aligned/uniform)
  // reference dummy arguments that are local to the interface body and
  // therefore have no address in the enclosing scope. Detect this by
  // comparing the program unit lexically containing the directive with the
  // procedure currently being lowered; if they differ, this evaluation is
  // for a different procedure (the interface-body subprogram) and emitting
  // an `omp.declare_simd` op here would create it with null operands. Skip
  // emission: lowering for `declare simd` on an external procedure declared
  // only via an interface body is not handled by this op-based form.
  const semantics::Scope &progUnitScope =
      semantics::GetProgramUnitContaining(semaCtx.FindScope(beginSpec.source));
  lower::pft::FunctionLikeUnit *owningProc = eval.getOwningProcedure();
  const semantics::Symbol *owningSym =
      (owningProc && !owningProc->isMainProgram())
          ? &owningProc->getSubprogramSymbol()
          : (owningProc ? owningProc->getMainProgramSymbol() : nullptr);
  if (progUnitScope.symbol() != owningSym)
    return;

  List<Clause> clauses = makeClauses(beginSpec.Clauses(), semaCtx);

  mlir::omp::DeclareSimdOperands clauseOps;
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAligned(clauseOps);
  cp.processInbranch(clauseOps);
  cp.processLinear(clauseOps, /*isDeclareSimd=*/true);
  cp.processNotinbranch(clauseOps);
  cp.processSimdlen(clauseOps);
  cp.processUniform(clauseOps);

  mlir::omp::DeclareSimdOp::create(converter.getFirOpBuilder(), loc, clauseOps);
}

static void
genOpenMPDeclareMapperImpl(lower::AbstractConverter &converter,
                           semantics::SemanticsContext &semaCtx,
                           const parser::OmpDeclareMapperDirective &construct,
                           const semantics::Symbol *mapperSymOpt = nullptr) {
  mlir::Location loc = converter.genLocation(construct.source);
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  const parser::OmpArgumentList &args = construct.v.Arguments();
  assert(args.v.size() == 1 && "Expecting single argument");
  lower::StatementContext stmtCtx;
  const auto *spec = std::get_if<parser::OmpMapperSpecifier>(&args.v.front().u);
  assert(spec && "Expecting mapper specifier");
  const auto &mapperName{std::get<std::string>(spec->t)};
  const auto &varType{std::get<parser::TypeSpec>(spec->t)};
  const auto &varName{std::get<parser::Name>(spec->t)};
  assert(varType.declTypeSpec->category() ==
             semantics::DeclTypeSpec::Category::TypeDerived &&
         "Expected derived type");

  std::string mapperNameStr = mapperName;
  if (mapperSymOpt && mapperNameStr != "default") {
    mapperNameStr = converter.mangleName(mapperNameStr, mapperSymOpt->owner());
  } else if (auto *sym =
                 converter.getCurrentScope().FindSymbol(mapperNameStr)) {
    mapperNameStr = converter.mangleName(mapperNameStr, sym->owner());
  }

  // If the mapper op already exists (e.g., created by regular lowering or by
  // materialization of imported mappers), do not recreate it.
  if (converter.getModuleOp().lookupSymbol(mapperNameStr))
    return;

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
  List<Clause> clauses = makeClauses(construct.v.Clauses(), semaCtx);
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processMap(loc, stmtCtx, clauseOps);
  mlir::omp::DeclareMapperInfoOp::create(firOpBuilder, loc, clauseOps);
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpDeclareMapperDirective &construct) {
  genOpenMPDeclareMapperImpl(converter, semaCtx, construct);
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OmpDeclareTargetDirective &declareTargetConstruct) {
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
                   const parser::OmpGroupprivateDirective &directive) {
  // The semantic analysis sets the flag and device_type on the
  // symbols; omp.groupprivate is materialised by groupprivatizeVars.
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpRequiresDirective &requiresConstruct) {
  // Requires directives are gathered and processed in semantics and
  // then combined in the lowering bridge before triggering codegen
  // just once. Hence, there is no need to lower each individual
  // occurrence here.
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpThreadprivateDirective &threadprivate) {
  // The directive is lowered when instantiating the variable to
  // support the case of threadprivate variable declared in module.
}

namespace {
struct MetadirectiveCandidate {
  MetadirectiveCandidate(const parser::OmpDirectiveSpecification *spec,
                         llvm::omp::VariantMatchInfo vmi, bool isExplicit,
                         std::optional<semantics::omp::DynamicUserCondition>
                             dynamicCond = std::nullopt,
                         bool conditionShouldBeTrue = true)
      : spec(spec), vmi(vmi), isExplicit(isExplicit), dynamicCond(dynamicCond),
        conditionShouldBeTrue(conditionShouldBeTrue) {}

  const parser::OmpDirectiveSpecification *spec = nullptr;
  llvm::omp::VariantMatchInfo vmi;
  bool isExplicit = false;
  std::optional<semantics::omp::DynamicUserCondition> dynamicCond;
  bool conditionShouldBeTrue = true;
};
} // namespace

static void genMetadirective(lower::AbstractConverter &converter,
                             lower::SymMap &symTable,
                             semantics::SemanticsContext &semaCtx,
                             lower::pft::Evaluation &eval,
                             const parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  llvm::SmallVector<llvm::omp::TraitProperty, 8> constructTraits;
  collectEnclosingConstructTraits(builder.getInsertionBlock()->getParentOp(),
                                  constructTraits);
  semantics::omp::OmpVariantMatchContext ompCtx =
      makeVariantMatchContext(builder.getModule(), constructTraits);

  llvm::SmallVector<MetadirectiveCandidate, 4> candidates;
  // A null directive specification represents either the implicit `nothing`
  // variant or the absence of an explicit otherwise/default clause.
  const parser::OmpDirectiveSpecification *fallback = nullptr;

  // Extract the context-selector that controls whether a WHEN variant is
  // applicable. Modifier validation requires exactly one selector per clause.
  auto getContextSelector = [](const parser::OmpClause::When &whenClause)
      -> const parser::modifier::OmpContextSelector & {
    const auto &modifiers = std::get<0>(whenClause.v.t);
    assert(modifiers && modifiers->size() == 1 &&
           "WHEN clause should contain one context-selector");
    return std::get<parser::modifier::OmpContextSelector>(modifiers->front().u);
  };

  // Extract the directive variant spec from a when clause.
  // Returns {spec_ptr, isExplicit}. A null spec means "nothing".
  auto getDirectiveVariant = [](const parser::OmpClause::When &whenClause)
      -> std::pair<const parser::OmpDirectiveSpecification *, bool> {
    const auto &opt = std::get<1>(whenClause.v.t);
    if (!opt)
      return {nullptr, false};
    if (opt->value().DirId() == llvm::omp::Directive::OMPD_nothing)
      return {nullptr, true};
    return {&opt->value(), true};
  };

  // Return the directive spec pointer, or nullptr for "nothing".
  auto getFallbackVariant = [](const parser::OmpDirectiveSpecification &spec)
      -> const parser::OmpDirectiveSpecification * {
    if (spec.DirId() == llvm::omp::Directive::OMPD_nothing)
      return nullptr;
    return &spec;
  };

  for (const auto &clause : clauseList.v) {
    if (const auto *whenClause =
            std::get_if<parser::OmpClause::When>(&clause.u)) {
      const auto &ctxSel = getContextSelector(*whenClause);
      auto [spec, isExplicit] = getDirectiveVariant(*whenClause);

      // METADIRECTIVE cannot yet honour some selector features that are
      // otherwise accepted; reject them before building the match info.
      switch (semantics::omp::FindUnsupportedSelectorFeature(ctxSel, semaCtx)) {
      case semantics::omp::UnsupportedSelectorFeature::TargetDevice:
        TODO(converter.genLocation(clause.source),
             "target_device selector in METADIRECTIVE");
        break;
      case semantics::omp::UnsupportedSelectorFeature::
          ClauseOrExtensionProperty:
        TODO(converter.genLocation(clause.source),
             "clause or extension trait matching in METADIRECTIVE");
        break;
      case semantics::omp::UnsupportedSelectorFeature::None:
        break;
      }

      llvm::omp::VariantMatchInfo rawVMI;
      std::optional<semantics::omp::DynamicUserCondition> dynamicCond =
          semantics::omp::MakeVariantMatchInfo(rawVMI, ctxSel, semaCtx);

      if (dynamicCond) {
        constexpr llvm::omp::TraitProperty dynamicConditionTrait =
            llvm::omp::TraitProperty::user_condition_unknown;
        constexpr llvm::omp::TraitProperty matchAnyTrait =
            llvm::omp::TraitProperty::implementation_extension_match_any;
        constexpr llvm::omp::TraitProperty matchNoneTrait =
            llvm::omp::TraitProperty::implementation_extension_match_none;

        // Static applicability must only use traits known at lowering time.
        // For example, in
        //   when(implementation={vendor(llvm)},
        //        user={condition(score(5): flag)}: barrier)
        // vendor(llvm) can be checked now, but flag cannot. Drop the
        // runtime-only user_condition_unknown for applicability, while keeping
        // score(5) so ranking can still honor the user-condition selector.
        llvm::omp::VariantMatchInfo staticVMI = rawVMI;
        std::optional<llvm::APInt> conditionScore;
        auto scoreIt = staticVMI.ScoreMap.find(dynamicConditionTrait);
        if (scoreIt != staticVMI.ScoreMap.end()) {
          conditionScore = scoreIt->second;
          staticVMI.ScoreMap.erase(scoreIt);
        }
        staticVMI.RequiredTraits.reset(unsigned(dynamicConditionTrait));
        llvm::APInt *conditionScorePtr =
            conditionScore ? &*conditionScore : nullptr;

        bool hasMatchAny = rawVMI.RequiredTraits.test(unsigned(matchAnyTrait));
        bool hasMatchNone =
            rawVMI.RequiredTraits.test(unsigned(matchNoneTrait));
        bool isStaticVMIApplicable =
            llvm::omp::isVariantApplicableInContext(staticVMI, ompCtx);
        // If staticVMI does not match, only match_any can still apply. Check
        // conditionTrueVMI because the runtime condition may satisfy match_any.
        if (!isStaticVMIApplicable) {
          if (!hasMatchAny || staticVMI.RequiredTraits.test(
                                  unsigned(llvm::omp::TraitProperty::invalid)))
            continue;

          llvm::omp::VariantMatchInfo conditionTrueVMI = staticVMI;
          conditionTrueVMI.addTrait(
              llvm::omp::TraitProperty::user_condition_true, "<condition>",
              conditionScorePtr);
          if (!llvm::omp::isVariantApplicableInContext(conditionTrueVMI,
                                                       ompCtx))
            continue;
        }

        auto addConditionTraitForRanking =
            [&](llvm::omp::VariantMatchInfo &rankingVMI) {
              rankingVMI.addTrait(
                  hasMatchNone ? dynamicConditionTrait
                               : llvm::omp::TraitProperty::user_condition_true,
                  "<condition>", conditionScorePtr);
            };

        if (hasMatchAny && isStaticVMIApplicable) {
          // A statically matched match_any selector needs two candidates: a
          // guarded candidate with the user condition and score, and an
          // unguarded candidate with only the statically matched traits. If the
          // when clause omits its directive, only add the unguarded candidate.
          if (isExplicit) {
            llvm::omp::VariantMatchInfo conditionTrueVMI = staticVMI;
            addConditionTraitForRanking(conditionTrueVMI);
            candidates.emplace_back(spec, conditionTrueVMI, isExplicit,
                                    dynamicCond);
          }
          candidates.emplace_back(spec, staticVMI, isExplicit);
          continue;
        }

        llvm::omp::VariantMatchInfo rankingVMI = staticVMI;
        // An omitted directive is implicit nothing, so do not let the runtime
        // condition raise its rank. Explicit `nothing` is still a variant.
        if (!isExplicit && hasMatchAny && !isStaticVMIApplicable)
          rankingVMI = llvm::omp::VariantMatchInfo();
        else if (isExplicit)
          addConditionTraitForRanking(rankingVMI);
        candidates.emplace_back(spec, rankingVMI, isExplicit, dynamicCond,
                                /*conditionShouldBeTrue=*/!hasMatchNone);
        continue;
      }

      if (!llvm::omp::isVariantApplicableInContext(rawVMI, ompCtx))
        continue;

      candidates.emplace_back(spec, rawVMI, isExplicit);
    } else if (const auto *otherwiseClause =
                   std::get_if<parser::OmpClause::Otherwise>(&clause.u)) {
      if (otherwiseClause->v && otherwiseClause->v->v)
        fallback = getFallbackVariant(otherwiseClause->v->v->value());
    } else if (const auto *defaultClause =
                   std::get_if<parser::OmpClause::Default>(&clause.u)) {
      if (const auto *dirSpecPtr = std::get_if<
              common::Indirection<parser::OmpDirectiveSpecification>>(
              &defaultClause->v.u))
        fallback = getFallbackVariant(dirSpecPtr->value());
    }
  }

  // Lower a single resolved candidate.
  auto genVariant = [&](const parser::OmpDirectiveSpecification *spec) {
    if (!spec) {
      genNestedEvaluations(converter, eval);
      return;
    }
    List<Clause> variantClauses = makeClauses(spec->Clauses(), semaCtx);
    mlir::Location variantLoc = converter.genLocation(spec->source);
    ConstructQueue queue{
        buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                            eval, spec->source, spec->DirId(), variantClauses)};

    if (llvm::any_of(queue, [](const auto &item) {
          return llvm::omp::getDirectiveAssociation(item.id) ==
                 llvm::omp::Association::LoopNest;
        })) {
      TODO(variantLoc, "loop-associated METADIRECTIVE variant");
    }

    if (llvm::any_of(queue, [](const auto &item) {
          return llvm::omp::getDirectiveAssociation(item.id) ==
                     llvm::omp::Association::Declaration ||
                 llvm::omp::getDirectiveCategory(item.id) ==
                     llvm::omp::Category::Declarative;
        })) {
      TODO(variantLoc, "declarative METADIRECTIVE variant");
    }

    genOMPDispatch(converter, symTable, semaCtx, eval, variantLoc, queue,
                   queue.begin());
  };

  auto selectBestCandidate =
      [](llvm::ArrayRef<unsigned> candidateIndices,
         llvm::ArrayRef<MetadirectiveCandidate> candidates,
         const semantics::omp::OmpVariantMatchContext &ompCtx)
      -> std::optional<unsigned> {
    if (candidateIndices.empty())
      return std::nullopt;
    if (candidateIndices.size() == 1)
      return candidateIndices.front();

    // The OpenMP context scorer preserves input order for tied candidates.
    // Put explicit variants first so they take precedence over implicit
    // `nothing`, as required by metadirective selection.
    llvm::SmallVector<unsigned, 4> candidateOrder;
    candidateOrder.reserve(candidateIndices.size());
    for (unsigned idx : candidateIndices)
      if (candidates[idx].isExplicit)
        candidateOrder.push_back(idx);
    for (unsigned idx : candidateIndices)
      if (!candidates[idx].isExplicit)
        candidateOrder.push_back(idx);

    llvm::SmallVector<llvm::omp::VariantMatchInfo, 4> orderedVMIs;
    orderedVMIs.reserve(candidateOrder.size());
    for (unsigned idx : candidateOrder)
      orderedVMIs.push_back(candidates[idx].vmi);

    int bestIdx = llvm::omp::getBestVariantMatchForContext(orderedVMIs, ompCtx);
    if (bestIdx >= 0) {
      assert(static_cast<size_t>(bestIdx) < candidateOrder.size() &&
             "best variant index out of range");
      return candidateOrder[bestIdx];
    }
    return std::nullopt;
  };

  llvm::SmallVector<unsigned, 4> remainingCandidates;
  remainingCandidates.reserve(candidates.size());
  for (unsigned idx = 0, end = candidates.size(); idx < end; ++idx)
    remainingCandidates.push_back(idx);

  lower::StatementContext stmtCtx;

  // Candidates that reach this loop passed static filtering. Runtime user
  // conditions are lowered as a ranked if/else cascade:
  //
  //   when(user={condition(a)}: barrier)
  //   when(user={condition(b)}: taskwait)
  //   otherwise(nothing)
  //
  // becomes:
  //
  //   if (a) barrier
  //   else if (b) taskwait
  //   else nothing
  //
  // If the else path selects the same unguarded directive, lower it directly.
  // Stop when selection reaches an unguarded candidate or the fallback.
  while (!remainingCandidates.empty()) {
    std::optional<unsigned> selected =
        selectBestCandidate(remainingCandidates, candidates, ompCtx);
    if (!selected) {
      genVariant(fallback);
      return;
    }

    const MetadirectiveCandidate &candidate = candidates[*selected];
    if (!candidate.dynamicCond) {
      genVariant(candidate.spec);
      return;
    }

    llvm::SmallVector<unsigned, 4> elsePathCandidates(remainingCandidates);
    auto *remainingIt = llvm::find(elsePathCandidates, *selected);
    assert(remainingIt != elsePathCandidates.end() &&
           "selected candidate missing from remaining candidates");
    elsePathCandidates.erase(remainingIt);

    // match_any may create a guarded condition-true candidate and an unguarded
    // static candidate for the same directive. If the else path picks the
    // unguarded one then fold it:
    //
    //   if (flag) barrier    into just    barrier
    //   else barrier
    if (std::optional<unsigned> selectedInElse =
            selectBestCandidate(elsePathCandidates, candidates, ompCtx)) {
      const MetadirectiveCandidate &candidateInElse =
          candidates[*selectedInElse];
      if (!candidateInElse.dynamicCond &&
          candidateInElse.spec == candidate.spec) {
        genVariant(candidate.spec);
        return;
      }
    }

    mlir::Location condLoc =
        converter.genLocation(candidate.dynamicCond->source);
    const auto *condExpr =
        semantics::GetExpr(semaCtx, *candidate.dynamicCond->expr);
    assert(condExpr && "missing expression for user condition");
    mlir::Value condVal =
        fir::getBase(converter.genExprValue(*condExpr, stmtCtx, &condLoc));

    if (condVal.getType() != builder.getI1Type())
      condVal = builder.createConvert(condLoc, builder.getI1Type(), condVal);
    if (!candidate.conditionShouldBeTrue) {
      mlir::Value trueVal =
          builder.createIntegerConstant(condLoc, builder.getI1Type(), 1);
      condVal = mlir::arith::XOrIOp::create(builder, condLoc, condVal, trueVal);
    }

    stmtCtx.finalizeAndReset();
    auto ifOp = fir::IfOp::create(builder, condLoc, condVal,
                                  /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    genVariant(candidate.spec);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    remainingCandidates = std::move(elsePathCandidates);
  }
  genVariant(fallback);
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpMetadirectiveDirective &meta) {
  genMetadirective(converter, symTable, semaCtx, eval, meta.v.Clauses());
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
                   const parser::OmpDelimitedMetadirectiveDirective &meta) {
  genMetadirective(converter, symTable, semaCtx, eval,
                   meta.BeginDir().Clauses());
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
        !std::holds_alternative<clause::Detach>(clause.u) &&
        !std::holds_alternative<clause::Device>(clause.u) &&
        !std::holds_alternative<clause::DynGroupprivate>(clause.u)) {
      const common::LangOptions &options = semaCtx.langOptions();
      if (!options.OpenMPSimd) {
        std::string name =
            parser::omp::GetUpperName(clause.id, options.OpenMPVersion);
        TODO(clauseLocation, name + " clause is not implemented yet");
      }
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
                   const parser::OmpAssumeDirective &assumeConstruct) {
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
    auto *designator = parser::omp::GetDesignatorFromObj(*object);
    assert(designator && "Expecting desginator in argument");
    auto *name = parser::GetDesignatorNameIfDataRef(*designator);
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
                   const parser::OmpUtilityDirective &dir) {
  common::visit(common::visitors{
                    [&](const parser::OmpNothingDirective &) {
                      // nothing-directive is a no-op (OpenMP 5.2 [8.4])
                    },
                    [&](const parser::OmpErrorDirective &) {
                      if (!semaCtx.langOptions().OpenMPSimd)
                        TODO(converter.getCurrentLocation(),
                             "OmpErrorDirective");
                    },
                },
                dir.u);
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
                   const parser::OpenMPLoopConstruct &loopConstruct) {
  const parser::OmpDirectiveSpecification &beginSpec = loopConstruct.BeginDir();
  List<Clause> clauses = makeClauses(beginSpec.Clauses(), semaCtx);
  if (auto &endSpec = loopConstruct.EndDir())
    clauses.append(makeClauses(endSpec->Clauses(), semaCtx));

  mlir::Location currentLocation = converter.genLocation(beginSpec.source);

  for (auto &construct : std::get<parser::Block>(loopConstruct.t)) {
    if (const parser::OpenMPLoopConstruct *ompNestedLoopCons =
            parser::omp::GetOmpLoop(construct)) {
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

  const parser::OmpDirectiveName &beginName = beginSpec.DirName();
  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, beginName.source, beginName.v, clauses)};
  genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OmpSectionDirective &sectionConstruct) {
  // Do nothing here. SECTION is lowered inside of the lowering for Sections
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPSectionsConstruct &construct) {
  const parser::OmpDirectiveSpecification &beginSpec{construct.BeginDir()};
  List<Clause> clauses = makeClauses(beginSpec.Clauses(), semaCtx);
  const auto &endSpec{construct.EndDir()};
  assert(endSpec &&
         "Missing end section directive should have been handled in semantics");
  clauses.append(makeClauses(endSpec->Clauses(), semaCtx));
  mlir::Location currentLocation = converter.getCurrentLocation();

  const parser::OmpDirectiveName &beginName{beginSpec.DirName()};
  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, beginName.source, beginName.v, clauses)};

  mlir::SaveStateStack<SectionsConstructStackFrame> saveStateStack{
      converter.getStateStack(), construct};
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

  if (sym.test(semantics::Symbol::Flag::OmpGroupPrivate))
    lower::genGroupprivateOp(converter, var);

  if (sym.test(semantics::Symbol::Flag::OmpThreadprivate))
    lower::genThreadprivateOp(converter, var);

  if (sym.test(semantics::Symbol::Flag::OmpDeclareTarget))
    lower::genDeclareTargetIntGlobal(converter, var);
}

void Fortran::lower::genGroupprivateOp(lower::AbstractConverter &converter,
                                       const lower::pft::Variable &var) {
  const semantics::Symbol &sym = var.getSymbol();

  // For common block members, the groupprivate op is generated for the entire
  // common block in groupprivatizeVars, not for individual members here.
  // The common block already has a global, so nothing to do here.
  if (semantics::FindCommonBlockContaining(sym.GetUltimate()))
    return;

  // Handle non-global variables: local variables with the SAVE attribute can
  // appear in a groupprivate directive. Promote them to fir.global so that
  // omp.groupprivate can reference them by symbol name.
  if (!var.isGlobal()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    mlir::Location currentLocation = converter.getCurrentLocation();
    auto module = converter.getModuleOp();
    std::string globalName = converter.mangleName(sym);
    if (!module.lookupSymbol<fir::GlobalOp>(globalName))
      globalInitialization(converter, firOpBuilder, sym, var, currentLocation);
  }

  // The actual omp.groupprivate operations are created by groupprivatizeVars.
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
    dir = loop->BeginDir().DirId();
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
          [&](const parser::OmpDeclareTargetDirective &ompReq) {
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
          [&](const parser::OmpDeclareTargetDirective &ompReq) {
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

  if (auto offloadMod =
          llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(mod)) {
    semantics::WithOmpDeclarative::OmpClauseSet reqs;
    if (symbol) {
      common::visit(
          [&](const auto &details) {
            if constexpr (std::is_base_of_v<semantics::WithOmpDeclarative,
                                            std::decay_t<decltype(details)>>) {
              reqs = details.ompRequires();
            }
          },
          symbol->details());
    }

    // Use pre-populated omp.requires module attribute if it was set, so that
    // the "-fopenmp-force-usm" compiler option is honored.
    MlirRequires mlirFlags = offloadMod.getRequires();
    if (reqs.test(llvm::omp::Clause::OMPC_dynamic_allocators))
      mlirFlags = mlirFlags | MlirRequires::dynamic_allocators;
    if (reqs.test(llvm::omp::Clause::OMPC_reverse_offload))
      mlirFlags = mlirFlags | MlirRequires::reverse_offload;
    if (reqs.test(llvm::omp::Clause::OMPC_unified_address))
      mlirFlags = mlirFlags | MlirRequires::unified_address;
    if (reqs.test(llvm::omp::Clause::OMPC_unified_shared_memory))
      mlirFlags = mlirFlags | MlirRequires::unified_shared_memory;

    offloadMod.setRequires(mlirFlags);
  }
}

// Walk scopes and materialize omp.declare_mapper ops for mapper declarations
// found in imported modules. If \p scope is null, start from the global scope.
void Fortran::lower::materializeOpenMPDeclareMappers(
    Fortran::lower::AbstractConverter &converter,
    semantics::SemanticsContext &semaCtx, const semantics::Scope *scope) {
  const semantics::Scope &root = scope ? *scope : semaCtx.globalScope();

  // Recurse into child scopes first (modules, submodules, etc.).
  for (const semantics::Scope &child : root.children())
    materializeOpenMPDeclareMappers(converter, semaCtx, &child);

  // Only consider module scopes to avoid duplicating local constructs.
  if (!root.IsModule())
    return;

  // Only materialize for modules coming from mod files to avoid duplicates.
  if (!root.symbol() || !root.symbol()->test(semantics::Symbol::Flag::ModFile))
    return;

  // Scan symbols in this module scope for MapperDetails.
  for (auto &it : root) {
    const semantics::Symbol &sym = *it.second;
    if (auto *md = sym.detailsIf<semantics::MapperDetails>()) {
      for (const auto *decl : md->GetDeclList()) {
        if (const auto *mapperDecl =
                std::get_if<parser::OmpDeclareMapperDirective>(&decl->u)) {
          genOpenMPDeclareMapperImpl(converter, semaCtx, *mapperDecl, &sym);
        }
      }
    }
  }
}
