//===-- Lower/OpenMP/ClauseProcessor.h --------------------------*- C++ -*-===//
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
#ifndef FORTRAN_LOWER_CLAUSEPROCESSOR_H
#define FORTRAN_LOWER_CLAUSEPROCESSOR_H

#include "Clauses.h"
#include "ReductionProcessor.h"
#include "Utils.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/DirectivesCommon.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace Fortran {
namespace lower {
namespace omp {

/// Class that handles the processing of OpenMP clauses.
///
/// Its `process<ClauseName>()` methods perform MLIR code generation for their
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
  bool processBare(mlir::omp::BareClauseOps &result) const;
  bool processBind(mlir::omp::BindClauseOps &result) const;
  bool
  processCollapse(mlir::Location currentLocation, lower::pft::Evaluation &eval,
                  mlir::omp::LoopRelatedClauseOps &result,
                  llvm::SmallVectorImpl<const semantics::Symbol *> &iv) const;
  bool processDevice(lower::StatementContext &stmtCtx,
                     mlir::omp::DeviceClauseOps &result) const;
  bool processDeviceType(mlir::omp::DeviceTypeClauseOps &result) const;
  bool processDistSchedule(lower::StatementContext &stmtCtx,
                           mlir::omp::DistScheduleClauseOps &result) const;
  bool processExclusive(mlir::Location currentLocation,
                        mlir::omp::ExclusiveClauseOps &result) const;
  bool processFilter(lower::StatementContext &stmtCtx,
                     mlir::omp::FilterClauseOps &result) const;
  bool processFinal(lower::StatementContext &stmtCtx,
                    mlir::omp::FinalClauseOps &result) const;
  bool processHasDeviceAddr(
      mlir::omp::HasDeviceAddrClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &isDeviceSyms) const;
  bool processHint(mlir::omp::HintClauseOps &result) const;
  bool processInclusive(mlir::Location currentLocation,
                        mlir::omp::InclusiveClauseOps &result) const;
  bool processMergeable(mlir::omp::MergeableClauseOps &result) const;
  bool processNowait(mlir::omp::NowaitClauseOps &result) const;
  bool processNumTeams(lower::StatementContext &stmtCtx,
                       mlir::omp::NumTeamsClauseOps &result) const;
  bool processNumThreads(lower::StatementContext &stmtCtx,
                         mlir::omp::NumThreadsClauseOps &result) const;
  bool processOrder(mlir::omp::OrderClauseOps &result) const;
  bool processOrdered(mlir::omp::OrderedClauseOps &result) const;
  bool processPriority(lower::StatementContext &stmtCtx,
                       mlir::omp::PriorityClauseOps &result) const;
  bool processProcBind(mlir::omp::ProcBindClauseOps &result) const;
  bool processSafelen(mlir::omp::SafelenClauseOps &result) const;
  bool processSchedule(lower::StatementContext &stmtCtx,
                       mlir::omp::ScheduleClauseOps &result) const;
  bool processSimdlen(mlir::omp::SimdlenClauseOps &result) const;
  bool processThreadLimit(lower::StatementContext &stmtCtx,
                          mlir::omp::ThreadLimitClauseOps &result) const;
  bool processUntied(mlir::omp::UntiedClauseOps &result) const;

  bool processDetach(mlir::omp::DetachClauseOps &result) const;
  // 'Repeatable' clauses: They can appear multiple times in the clause list.
  bool processAligned(mlir::omp::AlignedClauseOps &result) const;
  bool processAllocate(mlir::omp::AllocateClauseOps &result) const;
  bool processCopyin() const;
  bool processCopyprivate(mlir::Location currentLocation,
                          mlir::omp::CopyprivateClauseOps &result) const;
  bool processDepend(mlir::omp::DependClauseOps &result) const;
  bool
  processEnter(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;
  bool processIf(omp::clause::If::DirectiveNameModifier directiveName,
                 mlir::omp::IfClauseOps &result) const;
  bool processIsDevicePtr(
      mlir::omp::IsDevicePtrClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &isDeviceSyms) const;
  bool
  processLink(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;

  // This method is used to process a map clause.
  // The optional parameter mapSyms is used to store the original Fortran symbol
  // for the map operands. It may be used later on to create the block_arguments
  // for some of the directives that require it.
  bool processMap(mlir::Location currentLocation,
                  lower::StatementContext &stmtCtx,
                  mlir::omp::MapClauseOps &result,
                  llvm::SmallVectorImpl<const semantics::Symbol *> *mapSyms =
                      nullptr) const;
  bool processMotionClauses(lower::StatementContext &stmtCtx,
                            mlir::omp::MapClauseOps &result);
  bool processNontemporal(mlir::omp::NontemporalClauseOps &result) const;
  bool processReduction(
      mlir::Location currentLocation, mlir::omp::ReductionClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) const;
  bool processTo(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;
  bool processUseDeviceAddr(
      lower::StatementContext &stmtCtx,
      mlir::omp::UseDeviceAddrClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const;
  bool processUseDevicePtr(
      lower::StatementContext &stmtCtx,
      mlir::omp::UseDevicePtrClauseOps &result,
      llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const;

  // Call this method for these clauses that should be supported but are not
  // implemented yet. It triggers a compilation error if any of the given
  // clauses is found.
  template <typename... Ts>
  void processTODO(mlir::Location currentLocation,
                   llvm::omp::Directive directive) const;

private:
  using ClauseIterator = List<Clause>::const_iterator;

  /// Utility to find a clause within a range in the clause list.
  template <typename T>
  static ClauseIterator findClause(ClauseIterator begin, ClauseIterator end);

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

  /// Set the `result` to a new `mlir::UnitAttr` if the clause is present.
  template <typename T>
  bool markClauseOccurrence(mlir::UnitAttr &result) const;

  void processMapObjects(
      lower::StatementContext &stmtCtx, mlir::Location clauseLocation,
      const omp::ObjectList &objects,
      llvm::omp::OpenMPOffloadMappingFlags mapTypeBits,
      std::map<Object, OmpMapParentAndMemberData> &parentMemberIndices,
      llvm::SmallVectorImpl<mlir::Value> &mapVars,
      llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms,
      std::string mapperIdName = "") const;

  lower::AbstractConverter &converter;
  semantics::SemanticsContext &semaCtx;
  List<Clause> clauses;
};

template <typename... Ts>
void ClauseProcessor::processTODO(mlir::Location currentLocation,
                                  llvm::omp::Directive directive) const {
  auto checkUnhandledClause = [&](llvm::omp::Clause id, const auto *x) {
    if (!x)
      return;
    TODO(currentLocation,
         "Unhandled clause " + llvm::omp::getOpenMPClauseName(id).upper() +
             " in " + llvm::omp::getOpenMPDirectiveName(directive).upper() +
             " construct");
  };

  for (ClauseIterator it = clauses.begin(); it != clauses.end(); ++it)
    (checkUnhandledClause(it->id, std::get_if<Ts>(&it->u)), ...);
}

template <typename T>
ClauseProcessor::ClauseIterator
ClauseProcessor::findClause(ClauseIterator begin, ClauseIterator end) {
  for (ClauseIterator it = begin; it != end; ++it) {
    if (std::get_if<T>(&it->u))
      return it;
  }

  return end;
}

template <typename T>
const T *
ClauseProcessor::findUniqueClause(const parser::CharBlock **source) const {
  ClauseIterator it = findClause<T>(clauses.begin(), clauses.end());
  if (it != clauses.end()) {
    if (source)
      *source = &it->source;
    return &std::get<T>(it->u);
  }
  return nullptr;
}

template <typename T>
bool ClauseProcessor::findRepeatableClause(
    std::function<void(const T &, const parser::CharBlock &source)> callbackFn)
    const {
  bool found = false;
  ClauseIterator nextIt, endIt = clauses.end();
  for (ClauseIterator it = clauses.begin(); it != endIt; it = nextIt) {
    nextIt = findClause<T>(it, endIt);

    if (nextIt != endIt) {
      callbackFn(std::get<T>(nextIt->u), nextIt->source);
      found = true;
      ++nextIt;
    }
  }
  return found;
}

template <typename T>
bool ClauseProcessor::markClauseOccurrence(mlir::UnitAttr &result) const {
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
