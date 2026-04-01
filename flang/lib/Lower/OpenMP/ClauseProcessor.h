//===-- Lower/OpenMP/ClauseProcessor.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_CLAUSEPROCESSOR_H
#define FORTRAN_LOWER_CLAUSEPROCESSOR_H

#include "ClauseFinder.h"
#include "Utils.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/DirectivesCommon.h"
#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Lower/Support/ReductionProcessor.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/char-block.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace Fortran {
namespace lower {
namespace omp {

// Container type for tracking user specified Defaultmaps for a target region
using DefaultMapsTy = std::map<clause::Defaultmap::VariableCategory,
                               clause::Defaultmap::ImplicitBehavior>;

/// Class that handles the processing of OpenMP clauses.
///
/// Its `process<ClauseName>()` methods perform AIIR code generation for their
/// corresponding clause if it is present in the clause list. Otherwise, they
/// will return `false` to signal that the clause was not found.
///
/// The intended use of this class is to move clause processing outside of
/// construct processing, since the same clauses can appear attached to
/// different constructs and constructs can be combined, so that code
/// duplication is minimized.
///
/// Each construct-lowering function only calls the `process<ClauseName>()`
/// methods that relate to clauses that can impact the lowering of that
/// construct.
class ClauseProcessor {
public:
  ClauseProcessor(lower::AbstractConverter &converter,
                  semantics::SemanticsContext &semaCtx,
                  const List<Clause> &clauses)
      : converter(converter), semaCtx(semaCtx), clauses(clauses) {}

  // 'Unique' clauses: They can appear at most once in the clause list.
  bool processBare(aiir::omp::BareClauseOps &result) const;
  bool processBind(aiir::omp::BindClauseOps &result) const;
  bool processCancelDirectiveName(
      aiir::omp::CancelDirectiveNameClauseOps &result) const;
  bool
  processCollapse(aiir::Location currentLocation, lower::pft::Evaluation &eval,
                  aiir::omp::LoopRelatedClauseOps &loopResult,
                  aiir::omp::CollapseClauseOps &collapseResult,
                  llvm::SmallVectorImpl<const semantics::Symbol *> &iv) const;
  bool processSizes(StatementContext &stmtCtx,
                    aiir::omp::SizesClauseOps &result) const;
  bool processLooprange(StatementContext &stmtCtx,
                        aiir::omp::LooprangeClauseOps &result,
                        int64_t &count) const;
  bool processDevice(lower::StatementContext &stmtCtx,
                     aiir::omp::DeviceClauseOps &result) const;
  bool processDeviceType(aiir::omp::DeviceTypeClauseOps &result) const;
  bool processDistSchedule(lower::StatementContext &stmtCtx,
                           aiir::omp::DistScheduleClauseOps &result) const;
  bool processExclusive(aiir::Location currentLocation,
                        aiir::omp::ExclusiveClauseOps &result) const;
  bool processFilter(lower::StatementContext &stmtCtx,
                     aiir::omp::FilterClauseOps &result) const;
  bool processFinal(lower::StatementContext &stmtCtx,
                    aiir::omp::FinalClauseOps &result) const;
  bool processGrainsize(lower::StatementContext &stmtCtx,
                        aiir::omp::GrainsizeClauseOps &result) const;
  bool processHasDeviceAddr(
      lower::StatementContext &stmtCtx,
      aiir::omp::HasDeviceAddrClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &hasDeviceSyms) const;
  bool processHint(aiir::omp::HintClauseOps &result) const;
  bool processInbranch(aiir::omp::InbranchClauseOps &result) const;
  bool processInclusive(aiir::Location currentLocation,
                        aiir::omp::InclusiveClauseOps &result) const;
  bool processInitializer(
      lower::SymMap &symMap,
      ReductionProcessor::GenInitValueCBTy &genInitValueCB) const;
  bool processMergeable(aiir::omp::MergeableClauseOps &result) const;
  bool processNogroup(aiir::omp::NogroupClauseOps &result) const;
  bool processNotinbranch(aiir::omp::NotinbranchClauseOps &result) const;
  bool processNowait(aiir::omp::NowaitClauseOps &result) const;
  bool processNumTasks(lower::StatementContext &stmtCtx,
                       aiir::omp::NumTasksClauseOps &result) const;
  bool processNumTeams(lower::StatementContext &stmtCtx,
                       aiir::omp::NumTeamsClauseOps &result) const;
  bool processNumThreads(lower::StatementContext &stmtCtx,
                         aiir::omp::NumThreadsClauseOps &result) const;
  bool processOrder(aiir::omp::OrderClauseOps &result) const;
  bool processOrdered(aiir::omp::OrderedClauseOps &result) const;
  bool processPriority(lower::StatementContext &stmtCtx,
                       aiir::omp::PriorityClauseOps &result) const;
  bool processProcBind(aiir::omp::ProcBindClauseOps &result) const;
  bool processTileSizes(lower::pft::Evaluation &eval,
                        aiir::omp::LoopNestOperands &result) const;
  bool processSafelen(aiir::omp::SafelenClauseOps &result) const;
  bool processSchedule(lower::StatementContext &stmtCtx,
                       aiir::omp::ScheduleClauseOps &result) const;
  bool processSimdlen(aiir::omp::SimdlenClauseOps &result) const;
  bool processSimd(aiir::omp::OrderedRegionOperands &result) const;
  bool processThreadLimit(lower::StatementContext &stmtCtx,
                          aiir::omp::ThreadLimitClauseOps &result) const;
  bool processUntied(aiir::omp::UntiedClauseOps &result) const;

  bool processDetach(aiir::omp::DetachClauseOps &result) const;
  // 'Repeatable' clauses: They can appear multiple times in the clause list.
  bool processAffinity(aiir::omp::AffinityClauseOps &result) const;
  bool processAligned(aiir::omp::AlignedClauseOps &result) const;
  bool processAllocate(aiir::omp::AllocateClauseOps &result) const;
  bool processCopyin() const;
  bool processCopyprivate(aiir::Location currentLocation,
                          aiir::omp::CopyprivateClauseOps &result) const;
  bool processDefaultMap(lower::StatementContext &stmtCtx,
                         DefaultMapsTy &result) const;
  bool processDepend(lower::SymMap &symMap, lower::StatementContext &stmtCtx,
                     aiir::omp::DependClauseOps &result) const;
  bool
  processEnter(llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &result) const;
  bool processIf(omp::clause::If::DirectiveNameModifier directiveName,
                 aiir::omp::IfClauseOps &result) const;
  bool processInReduction(
      aiir::Location currentLocation, aiir::omp::InReductionClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &outReductionSyms) const;
  bool processIsDevicePtr(
      lower::StatementContext &stmtCtx, aiir::omp::IsDevicePtrClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &isDeviceSyms) const;
  bool processLinear(aiir::omp::LinearClauseOps &result,
                     bool isDeclareSimd = false) const;
  bool
  processLink(llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &result) const;

  // This method is used to process a map clause.
  // The optional parameter mapSyms is used to store the original Fortran symbol
  // for the map operands. It may be used later on to create the block_arguments
  // for some of the directives that require it.
  bool processMap(aiir::Location currentLocation,
                  lower::StatementContext &stmtCtx,
                  aiir::omp::MapClauseOps &result,
                  llvm::omp::Directive directive = llvm::omp::OMPD_unknown,
                  llvm::SmallVectorImpl<const semantics::Symbol *> *mapSyms =
                      nullptr) const;
  bool processMotionClauses(lower::StatementContext &stmtCtx,
                            aiir::omp::MapClauseOps &result);
  bool processNontemporal(aiir::omp::NontemporalClauseOps &result) const;
  bool processReduction(
      aiir::Location currentLocation, aiir::omp::ReductionClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) const;
  bool processTaskReduction(
      aiir::Location currentLocation, aiir::omp::TaskReductionClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &outReductionSyms) const;
  bool processTo(llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &result) const;
  bool processUseDeviceAddr(
      lower::StatementContext &stmtCtx,
      aiir::omp::UseDeviceAddrClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const;
  bool processUseDevicePtr(
      lower::StatementContext &stmtCtx,
      aiir::omp::UseDevicePtrClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const;
  bool processUniform(aiir::omp::UniformClauseOps &result) const;

  // Call this method for these clauses that should be supported but are not
  // implemented yet. It triggers a compilation error if any of the given
  // clauses is found.
  template <typename... Ts>
  void processTODO(aiir::Location currentLocation,
                   llvm::omp::Directive directive) const;

private:
  using ClauseIterator = List<Clause>::const_iterator;

  /// Return the first instance of the given clause found in the clause list or
  /// `nullptr` if not present. If more than one instance is expected, use
  /// `findRepeatableClause` instead.
  template <typename T>
  const T *findUniqueClause(const parser::CharBlock **source = nullptr) const;

  /// Call `callbackFn` for each occurrence of the given clause. Return `true`
  /// if at least one instance was found.
  template <typename T>
  bool findRepeatableClause(
      std::function<void(const T &, const parser::CharBlock &source)>
          callbackFn) const;

  /// Set the `result` to a new `aiir::UnitAttr` if the clause is present.
  template <typename T>
  bool markClauseOccurrence(aiir::UnitAttr &result) const;

  void processMapObjects(
      lower::StatementContext &stmtCtx, aiir::Location clauseLocation,
      const omp::ObjectList &objects, aiir::omp::ClauseMapFlags mapTypeBits,
      std::map<Object, OmpMapParentAndMemberData> &parentMemberIndices,
      llvm::SmallVectorImpl<aiir::Value> &mapVars,
      llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms,
      llvm::StringRef mapperIdNameRef = "",
      bool isMotionModifier = false) const;

  lower::AbstractConverter &converter;
  semantics::SemanticsContext &semaCtx;
  List<Clause> clauses;
};

template <typename... Ts>
void ClauseProcessor::processTODO(aiir::Location currentLocation,
                                  llvm::omp::Directive directive) const {
  auto checkUnhandledClause = [&](llvm::omp::Clause id, const auto *x) {
    if (!x)
      return;
    unsigned version = semaCtx.langOptions().OpenMPVersion;
    bool isSimdDirective = llvm::omp::getOpenMPDirectiveName(directive, version)
                               .upper()
                               .find("SIMD") != llvm::StringRef::npos;
    if (!semaCtx.langOptions().OpenMPSimd || isSimdDirective)
      TODO(currentLocation,
           "Unhandled clause " + llvm::omp::getOpenMPClauseName(id).upper() +
               " in " +
               llvm::omp::getOpenMPDirectiveName(directive, version).upper() +
               " construct");
  };

  for (ClauseIterator it = clauses.begin(); it != clauses.end(); ++it)
    (checkUnhandledClause(it->id, std::get_if<Ts>(&it->u)), ...);
}

template <typename T>
const T *
ClauseProcessor::findUniqueClause(const parser::CharBlock **source) const {
  return ClauseFinder::findUniqueClause<T>(clauses, source);
}

template <typename T>
bool ClauseProcessor::findRepeatableClause(
    std::function<void(const T &, const parser::CharBlock &source)> callbackFn)
    const {
  return ClauseFinder::findRepeatableClause<T>(clauses, callbackFn);
}

template <typename T>
bool ClauseProcessor::markClauseOccurrence(aiir::UnitAttr &result) const {
  if (findUniqueClause<T>()) {
    result = converter.getFirOpBuilder().getUnitAttr();
    return true;
  }
  return false;
}

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CLAUSEPROCESSOR_H
