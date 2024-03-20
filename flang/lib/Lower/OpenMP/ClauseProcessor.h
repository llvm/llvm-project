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
#ifndef FORTRAN_LOWER_CLAUASEPROCESSOR_H
#define FORTRAN_LOWER_CLAUASEPROCESSOR_H

#include "Clauses.h"
#include "DirectivesCommon.h"
#include "ReductionProcessor.h"
#include "Utils.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Bridge.h"
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
/// The intended use is of this class is to move clause processing outside of
/// construct processing, since the same clauses can appear attached to
/// different constructs and constructs can be combined, so that code
/// duplication is minimized.
///
/// Each construct-lowering function only calls the `process<ClauseName>()`
/// methods that relate to clauses that can impact the lowering of that
/// construct.
class ClauseProcessor {
  using ClauseTy = Fortran::parser::OmpClause;

public:
  ClauseProcessor(Fortran::lower::AbstractConverter &converter,
                  Fortran::semantics::SemanticsContext &semaCtx,
                  const Fortran::parser::OmpClauseList &clauses)
      : converter(converter), semaCtx(semaCtx), clauses2(clauses),
        clauses(makeList(clauses, semaCtx)) {}

  // 'Unique' clauses: They can appear at most once in the clause list.
  bool processCollapse(
      mlir::Location currentLocation, Fortran::lower::pft::Evaluation &eval,
      llvm::SmallVectorImpl<mlir::Value> &lowerBound,
      llvm::SmallVectorImpl<mlir::Value> &upperBound,
      llvm::SmallVectorImpl<mlir::Value> &step,
      llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> &iv) const;
  bool processDefault() const;
  bool processDevice(Fortran::lower::StatementContext &stmtCtx,
                     mlir::Value &result) const;
  bool processDeviceType(mlir::omp::DeclareTargetDeviceType &result) const;
  bool processFinal(Fortran::lower::StatementContext &stmtCtx,
                    mlir::Value &result) const;
  bool processHint(mlir::IntegerAttr &result) const;
  bool processMergeable(mlir::UnitAttr &result) const;
  bool processNowait(mlir::UnitAttr &result) const;
  bool processNumTeams(Fortran::lower::StatementContext &stmtCtx,
                       mlir::Value &result) const;
  bool processNumThreads(Fortran::lower::StatementContext &stmtCtx,
                         mlir::Value &result) const;
  bool processOrdered(mlir::IntegerAttr &result) const;
  bool processPriority(Fortran::lower::StatementContext &stmtCtx,
                       mlir::Value &result) const;
  bool processProcBind(mlir::omp::ClauseProcBindKindAttr &result) const;
  bool processSafelen(mlir::IntegerAttr &result) const;
  bool processSchedule(mlir::omp::ClauseScheduleKindAttr &valAttr,
                       mlir::omp::ScheduleModifierAttr &modifierAttr,
                       mlir::UnitAttr &simdModifierAttr) const;
  bool processScheduleChunk(Fortran::lower::StatementContext &stmtCtx,
                            mlir::Value &result) const;
  bool processSimdlen(mlir::IntegerAttr &result) const;
  bool processThreadLimit(Fortran::lower::StatementContext &stmtCtx,
                          mlir::Value &result) const;
  bool processUntied(mlir::UnitAttr &result) const;

  // 'Repeatable' clauses: They can appear multiple times in the clause list.
  bool
  processAllocate(llvm::SmallVectorImpl<mlir::Value> &allocatorOperands,
                  llvm::SmallVectorImpl<mlir::Value> &allocateOperands) const;
  bool processCopyin() const;
  bool processCopyPrivate(
      mlir::Location currentLocation,
      llvm::SmallVectorImpl<mlir::Value> &copyPrivateVars,
      llvm::SmallVectorImpl<mlir::Attribute> &copyPrivateFuncs) const;
  bool processDepend(llvm::SmallVectorImpl<mlir::Attribute> &dependTypeOperands,
                     llvm::SmallVectorImpl<mlir::Value> &dependOperands) const;
  bool
  processEnter(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;
  bool processIf(omp::clause::If::DirectiveNameModifier directiveName,
                 mlir::Value &result) const;
  bool
  processLink(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;

  // This method is used to process a map clause.
  // The optional parameters - mapSymTypes, mapSymLocs & mapSymbols are used to
  // store the original type, location and Fortran symbol for the map operands.
  // They may be used later on to create the block_arguments for some of the
  // target directives that require it.
  bool processMap(mlir::Location currentLocation,
                  const llvm::omp::Directive &directive,
                  Fortran::lower::StatementContext &stmtCtx,
                  llvm::SmallVectorImpl<mlir::Value> &mapOperands,
                  llvm::SmallVectorImpl<mlir::Type> *mapSymTypes = nullptr,
                  llvm::SmallVectorImpl<mlir::Location> *mapSymLocs = nullptr,
                  llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                      *mapSymbols = nullptr) const;
  bool
  processReduction(mlir::Location currentLocation,
                   llvm::SmallVectorImpl<mlir::Value> &reductionVars,
                   llvm::SmallVectorImpl<mlir::Type> &reductionTypes,
                   llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
                   llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                       *reductionSymbols = nullptr) const;
  bool processSectionsReduction(mlir::Location currentLocation) const;
  bool processTo(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;
  bool
  processUseDeviceAddr(llvm::SmallVectorImpl<mlir::Value> &operands,
                       llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
                       llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
                       llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                           &useDeviceSymbols) const;
  bool
  processUseDevicePtr(llvm::SmallVectorImpl<mlir::Value> &operands,
                      llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
                      llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
                      llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                          &useDeviceSymbols) const;

  template <typename T>
  bool processMotionClauses(Fortran::lower::StatementContext &stmtCtx,
                            llvm::SmallVectorImpl<mlir::Value> &mapOperands);

  // Call this method for these clauses that should be supported but are not
  // implemented yet. It triggers a compilation error if any of the given
  // clauses is found.
  template <typename... Ts>
  void processTODO(mlir::Location currentLocation,
                   llvm::omp::Directive directive) const;

private:
  using ClauseIterator = List<Clause>::const_iterator;
  using ClauseIterator2 = std::list<ClauseTy>::const_iterator;

  /// Utility to find a clause within a range in the clause list.
  template <typename T>
  static ClauseIterator findClause(ClauseIterator begin, ClauseIterator end);
  template <typename T>
  static ClauseIterator2 findClause2(ClauseIterator2 begin,
                                     ClauseIterator2 end);

  /// Return the first instance of the given clause found in the clause list or
  /// `nullptr` if not present. If more than one instance is expected, use
  /// `findRepeatableClause` instead.
  template <typename T>
  const T *
  findUniqueClause(const Fortran::parser::CharBlock **source = nullptr) const;

  /// Call `callbackFn` for each occurrence of the given clause. Return `true`
  /// if at least one instance was found.
  template <typename T>
  bool findRepeatableClause(
      std::function<void(const T &, const Fortran::parser::CharBlock &source)>
          callbackFn) const;
  template <typename T>
  bool findRepeatableClause2(
      std::function<void(const T *, const Fortran::parser::CharBlock &source)>
          callbackFn) const;

  /// Set the `result` to a new `mlir::UnitAttr` if the clause is present.
  template <typename T>
  bool markClauseOccurrence(mlir::UnitAttr &result) const;

  Fortran::lower::AbstractConverter &converter;
  Fortran::semantics::SemanticsContext &semaCtx;
  const Fortran::parser::OmpClauseList &clauses2;
  List<Clause> clauses;
};

template <typename T>
bool ClauseProcessor::processMotionClauses(
    Fortran::lower::StatementContext &stmtCtx,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands) {
  return findRepeatableClause2<T>(
      [&](const T *motionClause, const Fortran::parser::CharBlock &source) {
        mlir::Location clauseLocation = converter.genLocation(source);
        fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

        static_assert(std::is_same_v<T, ClauseProcessor::ClauseTy::To> ||
                      std::is_same_v<T, ClauseProcessor::ClauseTy::From>);

        // TODO Support motion modifiers: present, mapper, iterator.
        constexpr llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
            std::is_same_v<T, ClauseProcessor::ClauseTy::To>
                ? llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO
                : llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;

        for (const Fortran::parser::OmpObject &ompObject : motionClause->v.v) {
          llvm::SmallVector<mlir::Value> bounds;
          std::stringstream asFortran;
          Fortran::lower::AddrAndBoundsInfo info =
              Fortran::lower::gatherDataOperandAddrAndBounds<
                  Fortran::parser::OmpObject, mlir::omp::DataBoundsOp,
                  mlir::omp::DataBoundsType>(
                  converter, firOpBuilder, semaCtx, stmtCtx, ompObject,
                  clauseLocation, asFortran, bounds, treatIndexAsSection);

          auto origSymbol =
              converter.getSymbolAddress(*getOmpObjectSymbol(ompObject));
          mlir::Value symAddr = info.addr;
          if (origSymbol && fir::isTypeWithDescriptor(origSymbol.getType()))
            symAddr = origSymbol;

          // Explicit map captures are captured ByRef by default,
          // optimisation passes may alter this to ByCopy or other capture
          // types to optimise
          mlir::Value mapOp = createMapInfoOp(
              firOpBuilder, clauseLocation, symAddr, mlir::Value{},
              asFortran.str(), bounds, {},
              static_cast<
                  std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                  mapTypeBits),
              mlir::omp::VariableCaptureKind::ByRef, symAddr.getType());

          mapOperands.push_back(mapOp);
        }
      });
}

template <typename... Ts>
void ClauseProcessor::processTODO(mlir::Location currentLocation,
                                  llvm::omp::Directive directive) const {
  auto checkUnhandledClause = [&](const auto *x) {
    if (!x)
      return;
    TODO(currentLocation,
         "Unhandled clause " +
             llvm::StringRef(Fortran::parser::ParseTreeDumper::GetNodeName(*x))
                 .upper() +
             " in " + llvm::omp::getOpenMPDirectiveName(directive).upper() +
             " construct");
  };

  for (ClauseIterator2 it = clauses2.v.begin(); it != clauses2.v.end(); ++it)
    (checkUnhandledClause(std::get_if<Ts>(&it->u)), ...);
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
ClauseProcessor::ClauseIterator2
ClauseProcessor::findClause2(ClauseIterator2 begin, ClauseIterator2 end) {
  for (ClauseIterator2 it = begin; it != end; ++it) {
    if (std::get_if<T>(&it->u))
      return it;
  }

  return end;
}

template <typename T>
const T *ClauseProcessor::findUniqueClause(
    const Fortran::parser::CharBlock **source) const {
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
    std::function<void(const T &, const Fortran::parser::CharBlock &source)>
        callbackFn) const {
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
bool ClauseProcessor::findRepeatableClause2(
    std::function<void(const T *, const Fortran::parser::CharBlock &source)>
        callbackFn) const {
  bool found = false;
  ClauseIterator2 nextIt, endIt = clauses2.v.end();
  for (ClauseIterator2 it = clauses2.v.begin(); it != endIt; it = nextIt) {
    nextIt = findClause2<T>(it, endIt);

    if (nextIt != endIt) {
      callbackFn(&std::get<T>(nextIt->u), nextIt->source);
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

#endif // FORTRAN_LOWER_CLAUASEPROCESSOR_H
