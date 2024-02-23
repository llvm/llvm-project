//===-- ClauseProcessor.cpp -------------------------------------*- C++ -*-===//
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

#include "ClauseProcessor.h"

#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/tools.h"

namespace Fortran {
namespace lower {
namespace omp {

/// Check for unsupported map operand types.
static void checkMapType(mlir::Location location, mlir::Type type) {
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getElementType();
  if (auto boxType = type.dyn_cast_or_null<fir::BoxType>())
    if (!boxType.getElementType().isa<fir::PointerType>())
      TODO(location, "OMPD_target_data MapOperand BoxType");
}

static mlir::omp::ScheduleModifier
translateScheduleModifier(const Fortran::parser::OmpScheduleModifierType &m) {
  switch (m.v) {
  case Fortran::parser::OmpScheduleModifierType::ModType::Monotonic:
    return mlir::omp::ScheduleModifier::monotonic;
  case Fortran::parser::OmpScheduleModifierType::ModType::Nonmonotonic:
    return mlir::omp::ScheduleModifier::nonmonotonic;
  case Fortran::parser::OmpScheduleModifierType::ModType::Simd:
    return mlir::omp::ScheduleModifier::simd;
  }
  return mlir::omp::ScheduleModifier::none;
}

static mlir::omp::ScheduleModifier
getScheduleModifier(const Fortran::parser::OmpScheduleClause &x) {
  const auto &modifier =
      std::get<std::optional<Fortran::parser::OmpScheduleModifier>>(x.t);
  // The input may have the modifier any order, so we look for one that isn't
  // SIMD. If modifier is not set at all, fall down to the bottom and return
  // "none".
  if (modifier) {
    const auto &modType1 =
        std::get<Fortran::parser::OmpScheduleModifier::Modifier1>(modifier->t);
    if (modType1.v.v ==
        Fortran::parser::OmpScheduleModifierType::ModType::Simd) {
      const auto &modType2 = std::get<
          std::optional<Fortran::parser::OmpScheduleModifier::Modifier2>>(
          modifier->t);
      if (modType2 &&
          modType2->v.v !=
              Fortran::parser::OmpScheduleModifierType::ModType::Simd)
        return translateScheduleModifier(modType2->v);

      return mlir::omp::ScheduleModifier::none;
    }

    return translateScheduleModifier(modType1.v);
  }
  return mlir::omp::ScheduleModifier::none;
}

static mlir::omp::ScheduleModifier
getSimdModifier(const Fortran::parser::OmpScheduleClause &x) {
  const auto &modifier =
      std::get<std::optional<Fortran::parser::OmpScheduleModifier>>(x.t);
  // Either of the two possible modifiers in the input can be the SIMD modifier,
  // so look in either one, and return simd if we find one. Not found = return
  // "none".
  if (modifier) {
    const auto &modType1 =
        std::get<Fortran::parser::OmpScheduleModifier::Modifier1>(modifier->t);
    if (modType1.v.v == Fortran::parser::OmpScheduleModifierType::ModType::Simd)
      return mlir::omp::ScheduleModifier::simd;

    const auto &modType2 = std::get<
        std::optional<Fortran::parser::OmpScheduleModifier::Modifier2>>(
        modifier->t);
    if (modType2 && modType2->v.v ==
                        Fortran::parser::OmpScheduleModifierType::ModType::Simd)
      return mlir::omp::ScheduleModifier::simd;
  }
  return mlir::omp::ScheduleModifier::none;
}

static void
genAllocateClause(Fortran::lower::AbstractConverter &converter,
                  const Fortran::parser::OmpAllocateClause &ompAllocateClause,
                  llvm::SmallVectorImpl<mlir::Value> &allocatorOperands,
                  llvm::SmallVectorImpl<mlir::Value> &allocateOperands) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  Fortran::lower::StatementContext stmtCtx;

  mlir::Value allocatorOperand;
  const Fortran::parser::OmpObjectList &ompObjectList =
      std::get<Fortran::parser::OmpObjectList>(ompAllocateClause.t);
  const auto &allocateModifier = std::get<
      std::optional<Fortran::parser::OmpAllocateClause::AllocateModifier>>(
      ompAllocateClause.t);

  // If the allocate modifier is present, check if we only use the allocator
  // submodifier.  ALIGN in this context is unimplemented
  const bool onlyAllocator =
      allocateModifier &&
      std::holds_alternative<
          Fortran::parser::OmpAllocateClause::AllocateModifier::Allocator>(
          allocateModifier->u);

  if (allocateModifier && !onlyAllocator) {
    TODO(currentLocation, "OmpAllocateClause ALIGN modifier");
  }

  // Check if allocate clause has allocator specified. If so, add it
  // to list of allocators, otherwise, add default allocator to
  // list of allocators.
  if (onlyAllocator) {
    const auto &allocatorValue = std::get<
        Fortran::parser::OmpAllocateClause::AllocateModifier::Allocator>(
        allocateModifier->u);
    allocatorOperand = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(allocatorValue.v), stmtCtx));
    allocatorOperands.insert(allocatorOperands.end(), ompObjectList.v.size(),
                             allocatorOperand);
  } else {
    allocatorOperand = firOpBuilder.createIntegerConstant(
        currentLocation, firOpBuilder.getI32Type(), 1);
    allocatorOperands.insert(allocatorOperands.end(), ompObjectList.v.size(),
                             allocatorOperand);
  }
  genObjectList(ompObjectList, converter, allocateOperands);
}

static mlir::omp::ClauseProcBindKindAttr genProcBindKindAttr(
    fir::FirOpBuilder &firOpBuilder,
    const Fortran::parser::OmpClause::ProcBind *procBindClause) {
  mlir::omp::ClauseProcBindKind procBindKind;
  switch (procBindClause->v.v) {
  case Fortran::parser::OmpProcBindClause::Type::Master:
    procBindKind = mlir::omp::ClauseProcBindKind::Master;
    break;
  case Fortran::parser::OmpProcBindClause::Type::Close:
    procBindKind = mlir::omp::ClauseProcBindKind::Close;
    break;
  case Fortran::parser::OmpProcBindClause::Type::Spread:
    procBindKind = mlir::omp::ClauseProcBindKind::Spread;
    break;
  case Fortran::parser::OmpProcBindClause::Type::Primary:
    procBindKind = mlir::omp::ClauseProcBindKind::Primary;
    break;
  }
  return mlir::omp::ClauseProcBindKindAttr::get(firOpBuilder.getContext(),
                                                procBindKind);
}

static mlir::omp::ClauseTaskDependAttr
genDependKindAttr(fir::FirOpBuilder &firOpBuilder,
                  const Fortran::parser::OmpClause::Depend *dependClause) {
  mlir::omp::ClauseTaskDepend pbKind;
  switch (
      std::get<Fortran::parser::OmpDependenceType>(
          std::get<Fortran::parser::OmpDependClause::InOut>(dependClause->v.u)
              .t)
          .v) {
  case Fortran::parser::OmpDependenceType::Type::In:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependin;
    break;
  case Fortran::parser::OmpDependenceType::Type::Out:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependout;
    break;
  case Fortran::parser::OmpDependenceType::Type::Inout:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependinout;
    break;
  default:
    llvm_unreachable("unknown parser task dependence type");
    break;
  }
  return mlir::omp::ClauseTaskDependAttr::get(firOpBuilder.getContext(),
                                              pbKind);
}

static mlir::Value getIfClauseOperand(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::OmpClause::If *ifClause,
    Fortran::parser::OmpIfClause::DirectiveNameModifier directiveName,
    mlir::Location clauseLocation) {
  // Only consider the clause if it's intended for the given directive.
  auto &directive = std::get<
      std::optional<Fortran::parser::OmpIfClause::DirectiveNameModifier>>(
      ifClause->v.t);
  if (directive && directive.value() != directiveName)
    return nullptr;

  Fortran::lower::StatementContext stmtCtx;
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->v.t);
  mlir::Value ifVal = fir::getBase(
      converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
  return firOpBuilder.createConvert(clauseLocation, firOpBuilder.getI1Type(),
                                    ifVal);
}

static void
addUseDeviceClause(Fortran::lower::AbstractConverter &converter,
                   const Fortran::parser::OmpObjectList &useDeviceClause,
                   llvm::SmallVectorImpl<mlir::Value> &operands,
                   llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
                   llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
                   llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                       &useDeviceSymbols) {
  genObjectList(useDeviceClause, converter, operands);
  for (mlir::Value &operand : operands) {
    checkMapType(operand.getLoc(), operand.getType());
    useDeviceTypes.push_back(operand.getType());
    useDeviceLocs.push_back(operand.getLoc());
  }
  for (const Fortran::parser::OmpObject &ompObject : useDeviceClause.v) {
    Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
    useDeviceSymbols.push_back(sym);
  }
}

//===----------------------------------------------------------------------===//
// ClauseProcessor unique clauses
//===----------------------------------------------------------------------===//

bool ClauseProcessor::processCollapse(
    mlir::Location currentLocation, Fortran::lower::pft::Evaluation &eval,
    llvm::SmallVectorImpl<mlir::Value> &lowerBound,
    llvm::SmallVectorImpl<mlir::Value> &upperBound,
    llvm::SmallVectorImpl<mlir::Value> &step,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> &iv,
    std::size_t &loopVarTypeSize) const {
  bool found = false;
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Collect the loops to collapse.
  Fortran::lower::pft::Evaluation *doConstructEval =
      &eval.getFirstNestedEvaluation();
  if (doConstructEval->getIf<Fortran::parser::DoConstruct>()
          ->IsDoConcurrent()) {
    TODO(currentLocation, "Do Concurrent in Worksharing loop construct");
  }

  std::int64_t collapseValue = 1l;
  if (auto *collapseClause = findUniqueClause<ClauseTy::Collapse>()) {
    const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
    collapseValue = Fortran::evaluate::ToInt64(*expr).value();
    found = true;
  }

  loopVarTypeSize = 0;
  do {
    Fortran::lower::pft::Evaluation *doLoop =
        &doConstructEval->getFirstNestedEvaluation();
    auto *doStmt = doLoop->getIf<Fortran::parser::NonLabelDoStmt>();
    assert(doStmt && "Expected do loop to be in the nested evaluation");
    const auto &loopControl =
        std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
    const Fortran::parser::LoopControl::Bounds *bounds =
        std::get_if<Fortran::parser::LoopControl::Bounds>(&loopControl->u);
    assert(bounds && "Expected bounds for worksharing do loop");
    Fortran::lower::StatementContext stmtCtx;
    lowerBound.push_back(fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(bounds->lower), stmtCtx)));
    upperBound.push_back(fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(bounds->upper), stmtCtx)));
    if (bounds->step) {
      step.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(bounds->step), stmtCtx)));
    } else { // If `step` is not present, assume it as `1`.
      step.push_back(firOpBuilder.createIntegerConstant(
          currentLocation, firOpBuilder.getIntegerType(32), 1));
    }
    iv.push_back(bounds->name.thing.symbol);
    loopVarTypeSize = std::max(loopVarTypeSize,
                               bounds->name.thing.symbol->GetUltimate().size());
    collapseValue--;
    doConstructEval =
        &*std::next(doConstructEval->getNestedEvaluations().begin());
  } while (collapseValue > 0);

  return found;
}

bool ClauseProcessor::processDefault() const {
  if (auto *defaultClause = findUniqueClause<ClauseTy::Default>()) {
    // Private, Firstprivate, Shared, None
    switch (defaultClause->v.v) {
    case Fortran::parser::OmpDefaultClause::Type::Shared:
    case Fortran::parser::OmpDefaultClause::Type::None:
      // Default clause with shared or none do not require any handling since
      // Shared is the default behavior in the IR and None is only required
      // for semantic checks.
      break;
    case Fortran::parser::OmpDefaultClause::Type::Private:
      // TODO Support default(private)
      break;
    case Fortran::parser::OmpDefaultClause::Type::Firstprivate:
      // TODO Support default(firstprivate)
      break;
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processDevice(Fortran::lower::StatementContext &stmtCtx,
                                    mlir::Value &result) const {
  const Fortran::parser::CharBlock *source = nullptr;
  if (auto *deviceClause = findUniqueClause<ClauseTy::Device>(&source)) {
    mlir::Location clauseLocation = converter.genLocation(*source);
    if (auto deviceModifier = std::get<
            std::optional<Fortran::parser::OmpDeviceClause::DeviceModifier>>(
            deviceClause->v.t)) {
      if (deviceModifier ==
          Fortran::parser::OmpDeviceClause::DeviceModifier::Ancestor) {
        TODO(clauseLocation, "OMPD_target Device Modifier Ancestor");
      }
    }
    if (const auto *deviceExpr = Fortran::semantics::GetExpr(
            std::get<Fortran::parser::ScalarIntExpr>(deviceClause->v.t))) {
      result = fir::getBase(converter.genExprValue(*deviceExpr, stmtCtx));
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processDeviceType(
    mlir::omp::DeclareTargetDeviceType &result) const {
  if (auto *deviceTypeClause = findUniqueClause<ClauseTy::DeviceType>()) {
    // Case: declare target ... device_type(any | host | nohost)
    switch (deviceTypeClause->v.v) {
    case Fortran::parser::OmpDeviceTypeClause::Type::Nohost:
      result = mlir::omp::DeclareTargetDeviceType::nohost;
      break;
    case Fortran::parser::OmpDeviceTypeClause::Type::Host:
      result = mlir::omp::DeclareTargetDeviceType::host;
      break;
    case Fortran::parser::OmpDeviceTypeClause::Type::Any:
      result = mlir::omp::DeclareTargetDeviceType::any;
      break;
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processFinal(Fortran::lower::StatementContext &stmtCtx,
                                   mlir::Value &result) const {
  const Fortran::parser::CharBlock *source = nullptr;
  if (auto *finalClause = findUniqueClause<ClauseTy::Final>(&source)) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    mlir::Location clauseLocation = converter.genLocation(*source);

    mlir::Value finalVal = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(finalClause->v), stmtCtx));
    result = firOpBuilder.createConvert(clauseLocation,
                                        firOpBuilder.getI1Type(), finalVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processHint(mlir::IntegerAttr &result) const {
  if (auto *hintClause = findUniqueClause<ClauseTy::Hint>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    const auto *expr = Fortran::semantics::GetExpr(hintClause->v);
    int64_t hintValue = *Fortran::evaluate::ToInt64(*expr);
    result = firOpBuilder.getI64IntegerAttr(hintValue);
    return true;
  }
  return false;
}

bool ClauseProcessor::processMergeable(mlir::UnitAttr &result) const {
  return markClauseOccurrence<ClauseTy::Mergeable>(result);
}

bool ClauseProcessor::processNowait(mlir::UnitAttr &result) const {
  return markClauseOccurrence<ClauseTy::Nowait>(result);
}

bool ClauseProcessor::processNumTeams(Fortran::lower::StatementContext &stmtCtx,
                                      mlir::Value &result) const {
  // TODO Get lower and upper bounds for num_teams when parser is updated to
  // accept both.
  if (auto *numTeamsClause = findUniqueClause<ClauseTy::NumTeams>()) {
    result = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(numTeamsClause->v), stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processNumThreads(
    Fortran::lower::StatementContext &stmtCtx, mlir::Value &result) const {
  if (auto *numThreadsClause = findUniqueClause<ClauseTy::NumThreads>()) {
    // OMPIRBuilder expects `NUM_THREADS` clause as a `Value`.
    result = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(numThreadsClause->v), stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processOrdered(mlir::IntegerAttr &result) const {
  if (auto *orderedClause = findUniqueClause<ClauseTy::Ordered>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    int64_t orderedClauseValue = 0l;
    if (orderedClause->v.has_value()) {
      const auto *expr = Fortran::semantics::GetExpr(orderedClause->v);
      orderedClauseValue = *Fortran::evaluate::ToInt64(*expr);
    }
    result = firOpBuilder.getI64IntegerAttr(orderedClauseValue);
    return true;
  }
  return false;
}

bool ClauseProcessor::processPriority(Fortran::lower::StatementContext &stmtCtx,
                                      mlir::Value &result) const {
  if (auto *priorityClause = findUniqueClause<ClauseTy::Priority>()) {
    result = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(priorityClause->v), stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processProcBind(
    mlir::omp::ClauseProcBindKindAttr &result) const {
  if (auto *procBindClause = findUniqueClause<ClauseTy::ProcBind>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    result = genProcBindKindAttr(firOpBuilder, procBindClause);
    return true;
  }
  return false;
}

bool ClauseProcessor::processSafelen(mlir::IntegerAttr &result) const {
  if (auto *safelenClause = findUniqueClause<ClauseTy::Safelen>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    const auto *expr = Fortran::semantics::GetExpr(safelenClause->v);
    const std::optional<std::int64_t> safelenVal =
        Fortran::evaluate::ToInt64(*expr);
    result = firOpBuilder.getI64IntegerAttr(*safelenVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processSchedule(
    mlir::omp::ClauseScheduleKindAttr &valAttr,
    mlir::omp::ScheduleModifierAttr &modifierAttr,
    mlir::UnitAttr &simdModifierAttr) const {
  if (auto *scheduleClause = findUniqueClause<ClauseTy::Schedule>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    mlir::MLIRContext *context = firOpBuilder.getContext();
    const Fortran::parser::OmpScheduleClause &scheduleType = scheduleClause->v;
    const auto &scheduleClauseKind =
        std::get<Fortran::parser::OmpScheduleClause::ScheduleType>(
            scheduleType.t);

    mlir::omp::ClauseScheduleKind scheduleKind;
    switch (scheduleClauseKind) {
    case Fortran::parser::OmpScheduleClause::ScheduleType::Static:
      scheduleKind = mlir::omp::ClauseScheduleKind::Static;
      break;
    case Fortran::parser::OmpScheduleClause::ScheduleType::Dynamic:
      scheduleKind = mlir::omp::ClauseScheduleKind::Dynamic;
      break;
    case Fortran::parser::OmpScheduleClause::ScheduleType::Guided:
      scheduleKind = mlir::omp::ClauseScheduleKind::Guided;
      break;
    case Fortran::parser::OmpScheduleClause::ScheduleType::Auto:
      scheduleKind = mlir::omp::ClauseScheduleKind::Auto;
      break;
    case Fortran::parser::OmpScheduleClause::ScheduleType::Runtime:
      scheduleKind = mlir::omp::ClauseScheduleKind::Runtime;
      break;
    }

    mlir::omp::ScheduleModifier scheduleModifier =
        getScheduleModifier(scheduleClause->v);

    if (scheduleModifier != mlir::omp::ScheduleModifier::none)
      modifierAttr =
          mlir::omp::ScheduleModifierAttr::get(context, scheduleModifier);

    if (getSimdModifier(scheduleClause->v) != mlir::omp::ScheduleModifier::none)
      simdModifierAttr = firOpBuilder.getUnitAttr();

    valAttr = mlir::omp::ClauseScheduleKindAttr::get(context, scheduleKind);
    return true;
  }
  return false;
}

bool ClauseProcessor::processScheduleChunk(
    Fortran::lower::StatementContext &stmtCtx, mlir::Value &result) const {
  if (auto *scheduleClause = findUniqueClause<ClauseTy::Schedule>()) {
    if (const auto &chunkExpr =
            std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                scheduleClause->v.t)) {
      if (const auto *expr = Fortran::semantics::GetExpr(*chunkExpr)) {
        result = fir::getBase(converter.genExprValue(*expr, stmtCtx));
      }
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processSimdlen(mlir::IntegerAttr &result) const {
  if (auto *simdlenClause = findUniqueClause<ClauseTy::Simdlen>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    const auto *expr = Fortran::semantics::GetExpr(simdlenClause->v);
    const std::optional<std::int64_t> simdlenVal =
        Fortran::evaluate::ToInt64(*expr);
    result = firOpBuilder.getI64IntegerAttr(*simdlenVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processThreadLimit(
    Fortran::lower::StatementContext &stmtCtx, mlir::Value &result) const {
  if (auto *threadLmtClause = findUniqueClause<ClauseTy::ThreadLimit>()) {
    result = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(threadLmtClause->v), stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processUntied(mlir::UnitAttr &result) const {
  return markClauseOccurrence<ClauseTy::Untied>(result);
}

//===----------------------------------------------------------------------===//
// ClauseProcessor repeatable clauses
//===----------------------------------------------------------------------===//

bool ClauseProcessor::processAllocate(
    llvm::SmallVectorImpl<mlir::Value> &allocatorOperands,
    llvm::SmallVectorImpl<mlir::Value> &allocateOperands) const {
  return findRepeatableClause<ClauseTy::Allocate>(
      [&](const ClauseTy::Allocate *allocateClause,
          const Fortran::parser::CharBlock &) {
        genAllocateClause(converter, allocateClause->v, allocatorOperands,
                          allocateOperands);
      });
}

bool ClauseProcessor::processCopyin() const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());
  auto checkAndCopyHostAssociateVar =
      [&](Fortran::semantics::Symbol *sym,
          mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr) {
        assert(sym->has<Fortran::semantics::HostAssocDetails>() &&
               "No host-association found");
        if (converter.isPresentShallowLookup(*sym))
          converter.copyHostAssociateVar(*sym, copyAssignIP);
      };
  bool hasCopyin = findRepeatableClause<ClauseTy::Copyin>(
      [&](const ClauseTy::Copyin *copyinClause,
          const Fortran::parser::CharBlock &) {
        const Fortran::parser::OmpObjectList &ompObjectList = copyinClause->v;
        for (const Fortran::parser::OmpObject &ompObject : ompObjectList.v) {
          Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
          if (const auto *commonDetails =
                  sym->detailsIf<Fortran::semantics::CommonBlockDetails>()) {
            for (const auto &mem : commonDetails->objects())
              checkAndCopyHostAssociateVar(&*mem, &insPt);
            break;
          }
          if (Fortran::semantics::IsAllocatableOrObjectPointer(
                  &sym->GetUltimate()))
            TODO(converter.getCurrentLocation(),
                 "pointer or allocatable variables in Copyin clause");
          assert(sym->has<Fortran::semantics::HostAssocDetails>() &&
                 "No host-association found");
          checkAndCopyHostAssociateVar(sym);
        }
      });

  // [OMP 5.0, 2.19.6.1] The copy is done after the team is formed and prior to
  // the execution of the associated structured block. Emit implicit barrier to
  // synchronize threads and avoid data races on propagation master's thread
  // values of threadprivate variables to local instances of that variables of
  // all other implicit threads.
  if (hasCopyin)
    firOpBuilder.create<mlir::omp::BarrierOp>(converter.getCurrentLocation());
  firOpBuilder.restoreInsertionPoint(insPt);
  return hasCopyin;
}

/// Class that extracts information from the specified type.
class TypeInfo {
public:
  TypeInfo(mlir::Type ty) { typeScan(ty); }

  // Returns the length of character types.
  std::optional<fir::CharacterType::LenType> getCharLength() const {
    return charLen;
  }

  // Returns the shape of array types.
  const llvm::SmallVector<int64_t> &getShape() const { return shape; }

  // Is the type inside a box?
  bool isBox() const { return inBox; }

private:
  void typeScan(mlir::Type type);

  std::optional<fir::CharacterType::LenType> charLen;
  llvm::SmallVector<int64_t> shape;
  bool inBox = false;
};

void TypeInfo::typeScan(mlir::Type ty) {
  if (auto sty = mlir::dyn_cast<fir::SequenceType>(ty)) {
    assert(shape.empty() && !sty.getShape().empty());
    shape = llvm::SmallVector<int64_t>(sty.getShape());
    typeScan(sty.getEleTy());
  } else if (auto bty = mlir::dyn_cast<fir::BoxType>(ty)) {
    inBox = true;
    typeScan(bty.getEleTy());
  } else if (auto cty = mlir::dyn_cast<fir::CharacterType>(ty)) {
    charLen = cty.getLen();
  } else if (auto hty = mlir::dyn_cast<fir::HeapType>(ty)) {
    typeScan(hty.getEleTy());
  } else if (auto pty = mlir::dyn_cast<fir::PointerType>(ty)) {
    typeScan(pty.getEleTy());
  } else {
    // The scan ends when reaching any built-in or record type.
    assert(ty.isIntOrIndexOrFloat() || mlir::isa<fir::ComplexType>(ty) ||
           mlir::isa<fir::LogicalType>(ty) || mlir::isa<fir::RecordType>(ty));
  }
}

// Create a function that performs a copy between two variables, compatible
// with their types and attributes.
static mlir::func::FuncOp
createCopyFunc(mlir::Location loc, Fortran::lower::AbstractConverter &converter,
               mlir::Type varType, fir::FortranVariableFlagsEnum varAttrs) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::ModuleOp module = builder.getModule();
  mlir::Type eleTy = mlir::cast<fir::ReferenceType>(varType).getEleTy();
  TypeInfo typeInfo(eleTy);
  std::string copyFuncName =
      fir::getTypeAsString(eleTy, builder.getKindMap(), "_copy");

  if (auto decl = module.lookupSymbol<mlir::func::FuncOp>(copyFuncName))
    return decl;

  // create function
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  llvm::SmallVector<mlir::Type> argsTy = {varType, varType};
  auto funcType = mlir::FunctionType::get(builder.getContext(), argsTy, {});
  mlir::func::FuncOp funcOp =
      modBuilder.create<mlir::func::FuncOp>(loc, copyFuncName, funcType);
  funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  builder.createBlock(&funcOp.getRegion(), funcOp.getRegion().end(), argsTy,
                      {loc, loc});
  builder.setInsertionPointToStart(&funcOp.getRegion().back());
  // generate body
  fir::FortranVariableFlagsAttr attrs;
  if (varAttrs != fir::FortranVariableFlagsEnum::None)
    attrs = fir::FortranVariableFlagsAttr::get(builder.getContext(), varAttrs);
  llvm::SmallVector<mlir::Value> typeparams;
  if (typeInfo.getCharLength().has_value()) {
    mlir::Value charLen = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), *typeInfo.getCharLength());
    typeparams.push_back(charLen);
  }
  mlir::Value shape;
  if (!typeInfo.isBox() && !typeInfo.getShape().empty()) {
    llvm::SmallVector<mlir::Value> extents;
    for (auto extent : typeInfo.getShape())
      extents.push_back(
          builder.createIntegerConstant(loc, builder.getIndexType(), extent));
    shape = builder.create<fir::ShapeOp>(loc, extents);
  }
  auto declDst = builder.create<hlfir::DeclareOp>(loc, funcOp.getArgument(0),
                                                  copyFuncName + "_dst", shape,
                                                  typeparams, attrs);
  auto declSrc = builder.create<hlfir::DeclareOp>(loc, funcOp.getArgument(1),
                                                  copyFuncName + "_src", shape,
                                                  typeparams, attrs);
  converter.copyVar(loc, declDst.getBase(), declSrc.getBase());
  builder.create<mlir::func::ReturnOp>(loc);
  return funcOp;
}

bool ClauseProcessor::processCopyPrivate(
    mlir::Location currentLocation,
    llvm::SmallVectorImpl<mlir::Value> &copyPrivateVars,
    llvm::SmallVectorImpl<mlir::Attribute> &copyPrivateFuncs) const {
  auto addCopyPrivateVar = [&](Fortran::semantics::Symbol *sym) {
    mlir::Value symVal = converter.getSymbolAddress(*sym);
    auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>();
    if (!declOp)
      fir::emitFatalError(currentLocation,
                          "COPYPRIVATE is supported only in HLFIR mode");
    symVal = declOp.getBase();
    mlir::Type symType = symVal.getType();
    fir::FortranVariableFlagsEnum attrs =
        declOp.getFortranAttrs().has_value()
            ? *declOp.getFortranAttrs()
            : fir::FortranVariableFlagsEnum::None;
    mlir::Value cpVar = symVal;

    // CopyPrivate variables must be passed by reference. However, in the case
    // of assumed shapes/vla the type is not a !fir.ref, but a !fir.box.
    // In these cases to retrieve the appropriate !fir.ref<!fir.box<...>> to
    // access the data we need we must perform an alloca and then store to it
    // and retrieve the data from the new alloca.
    if (mlir::isa<fir::BaseBoxType>(symType)) {
      fir::FirOpBuilder &builder = converter.getFirOpBuilder();
      auto alloca = builder.create<fir::AllocaOp>(currentLocation, symType);
      builder.create<fir::StoreOp>(currentLocation, symVal, alloca);
      cpVar = alloca;
    }

    copyPrivateVars.push_back(cpVar);
    mlir::func::FuncOp funcOp =
        createCopyFunc(currentLocation, converter, cpVar.getType(), attrs);
    copyPrivateFuncs.push_back(mlir::SymbolRefAttr::get(funcOp));
  };

  bool hasCopyPrivate = findRepeatableClause<ClauseTy::Copyprivate>(
      [&](const ClauseTy::Copyprivate *copyPrivateClause,
          const Fortran::parser::CharBlock &) {
        const Fortran::parser::OmpObjectList &ompObjectList =
            copyPrivateClause->v;
        for (const Fortran::parser::OmpObject &ompObject : ompObjectList.v) {
          Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
          if (const auto *commonDetails =
                  sym->detailsIf<Fortran::semantics::CommonBlockDetails>()) {
            for (const auto &mem : commonDetails->objects())
              addCopyPrivateVar(&*mem);
            break;
          }
          addCopyPrivateVar(sym);
        }
      });

  return hasCopyPrivate;
}

bool ClauseProcessor::processDepend(
    llvm::SmallVectorImpl<mlir::Attribute> &dependTypeOperands,
    llvm::SmallVectorImpl<mlir::Value> &dependOperands) const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  return findRepeatableClause<ClauseTy::Depend>(
      [&](const ClauseTy::Depend *dependClause,
          const Fortran::parser::CharBlock &) {
        const std::list<Fortran::parser::Designator> &depVal =
            std::get<std::list<Fortran::parser::Designator>>(
                std::get<Fortran::parser::OmpDependClause::InOut>(
                    dependClause->v.u)
                    .t);
        mlir::omp::ClauseTaskDependAttr dependTypeOperand =
            genDependKindAttr(firOpBuilder, dependClause);
        dependTypeOperands.insert(dependTypeOperands.end(), depVal.size(),
                                  dependTypeOperand);
        for (const Fortran::parser::Designator &ompObject : depVal) {
          Fortran::semantics::Symbol *sym = nullptr;
          std::visit(
              Fortran::common::visitors{
                  [&](const Fortran::parser::DataRef &designator) {
                    if (const Fortran::parser::Name *name =
                            std::get_if<Fortran::parser::Name>(&designator.u)) {
                      sym = name->symbol;
                    } else if (std::get_if<Fortran::common::Indirection<
                                   Fortran::parser::ArrayElement>>(
                                   &designator.u)) {
                      TODO(converter.getCurrentLocation(),
                           "array sections not supported for task depend");
                    }
                  },
                  [&](const Fortran::parser::Substring &designator) {
                    TODO(converter.getCurrentLocation(),
                         "substring not supported for task depend");
                  }},
              (ompObject).u);
          const mlir::Value variable = converter.getSymbolAddress(*sym);
          dependOperands.push_back(variable);
        }
      });
}

bool ClauseProcessor::processIf(
    Fortran::parser::OmpIfClause::DirectiveNameModifier directiveName,
    mlir::Value &result) const {
  bool found = false;
  findRepeatableClause<ClauseTy::If>(
      [&](const ClauseTy::If *ifClause,
          const Fortran::parser::CharBlock &source) {
        mlir::Location clauseLocation = converter.genLocation(source);
        mlir::Value operand = getIfClauseOperand(converter, ifClause,
                                                 directiveName, clauseLocation);
        // Assume that, at most, a single 'if' clause will be applicable to the
        // given directive.
        if (operand) {
          result = operand;
          found = true;
        }
      });
  return found;
}

bool ClauseProcessor::processLink(
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const {
  return findRepeatableClause<ClauseTy::Link>(
      [&](const ClauseTy::Link *linkClause,
          const Fortran::parser::CharBlock &) {
        // Case: declare target link(var1, var2)...
        gatherFuncAndVarSyms(
            linkClause->v, mlir::omp::DeclareTargetCaptureClause::link, result);
      });
}

mlir::omp::MapInfoOp
createMapInfoOp(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                mlir::SmallVector<mlir::Value> bounds,
                mlir::SmallVector<mlir::Value> members, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool isVal) {
  if (auto boxTy = baseAddr.getType().dyn_cast<fir::BaseBoxType>()) {
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  mlir::omp::MapInfoOp op = builder.create<mlir::omp::MapInfoOp>(
      loc, retTy, baseAddr, varType, varPtrPtr, members, bounds,
      builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
      builder.getStringAttr(name));

  return op;
}

bool ClauseProcessor::processMap(
    mlir::Location currentLocation, const llvm::omp::Directive &directive,
    Fortran::lower::StatementContext &stmtCtx,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *mapSymbols)
    const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  return findRepeatableClause<ClauseTy::Map>(
      [&](const ClauseTy::Map *mapClause,
          const Fortran::parser::CharBlock &source) {
        mlir::Location clauseLocation = converter.genLocation(source);
        const auto &oMapType =
            std::get<std::optional<Fortran::parser::OmpMapType>>(
                mapClause->v.t);
        llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE;
        // If the map type is specified, then process it else Tofrom is the
        // default.
        if (oMapType) {
          const Fortran::parser::OmpMapType::Type &mapType =
              std::get<Fortran::parser::OmpMapType::Type>(oMapType->t);
          switch (mapType) {
          case Fortran::parser::OmpMapType::Type::To:
            mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
            break;
          case Fortran::parser::OmpMapType::Type::From:
            mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
            break;
          case Fortran::parser::OmpMapType::Type::Tofrom:
            mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
                           llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
            break;
          case Fortran::parser::OmpMapType::Type::Alloc:
          case Fortran::parser::OmpMapType::Type::Release:
            // alloc and release is the default map_type for the Target Data
            // Ops, i.e. if no bits for map_type is supplied then alloc/release
            // is implicitly assumed based on the target directive. Default
            // value for Target Data and Enter Data is alloc and for Exit Data
            // it is release.
            break;
          case Fortran::parser::OmpMapType::Type::Delete:
            mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_DELETE;
          }

          if (std::get<std::optional<Fortran::parser::OmpMapType::Always>>(
                  oMapType->t))
            mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_ALWAYS;
        } else {
          mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
                         llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
        }

        for (const Fortran::parser::OmpObject &ompObject :
             std::get<Fortran::parser::OmpObjectList>(mapClause->v.t).v) {
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
          if (mapSymTypes)
            mapSymTypes->push_back(symAddr.getType());
          if (mapSymLocs)
            mapSymLocs->push_back(symAddr.getLoc());

          if (mapSymbols)
            mapSymbols->push_back(getOmpObjectSymbol(ompObject));
        }
      });
}

bool ClauseProcessor::processReduction(
    mlir::Location currentLocation,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *reductionSymbols)
    const {
  return findRepeatableClause<ClauseTy::Reduction>(
      [&](const ClauseTy::Reduction *reductionClause,
          const Fortran::parser::CharBlock &) {
        ReductionProcessor rp;
        rp.addReductionDecl(currentLocation, converter, reductionClause->v,
                            reductionVars, reductionDeclSymbols,
                            reductionSymbols);
      });
}

bool ClauseProcessor::processSectionsReduction(
    mlir::Location currentLocation) const {
  return findRepeatableClause<ClauseTy::Reduction>(
      [&](const ClauseTy::Reduction *, const Fortran::parser::CharBlock &) {
        TODO(currentLocation, "OMPC_Reduction");
      });
}

bool ClauseProcessor::processTo(
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const {
  return findRepeatableClause<ClauseTy::To>(
      [&](const ClauseTy::To *toClause, const Fortran::parser::CharBlock &) {
        // Case: declare target to(func, var1, var2)...
        gatherFuncAndVarSyms(toClause->v,
                             mlir::omp::DeclareTargetCaptureClause::to, result);
      });
}

bool ClauseProcessor::processEnter(
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const {
  return findRepeatableClause<ClauseTy::Enter>(
      [&](const ClauseTy::Enter *enterClause,
          const Fortran::parser::CharBlock &) {
        // Case: declare target enter(func, var1, var2)...
        gatherFuncAndVarSyms(enterClause->v,
                             mlir::omp::DeclareTargetCaptureClause::enter,
                             result);
      });
}

bool ClauseProcessor::processUseDeviceAddr(
    llvm::SmallVectorImpl<mlir::Value> &operands,
    llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> &useDeviceSymbols)
    const {
  return findRepeatableClause<ClauseTy::UseDeviceAddr>(
      [&](const ClauseTy::UseDeviceAddr *devAddrClause,
          const Fortran::parser::CharBlock &) {
        addUseDeviceClause(converter, devAddrClause->v, operands,
                           useDeviceTypes, useDeviceLocs, useDeviceSymbols);
      });
}

bool ClauseProcessor::processUseDevicePtr(
    llvm::SmallVectorImpl<mlir::Value> &operands,
    llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> &useDeviceSymbols)
    const {
  return findRepeatableClause<ClauseTy::UseDevicePtr>(
      [&](const ClauseTy::UseDevicePtr *devPtrClause,
          const Fortran::parser::CharBlock &) {
        addUseDeviceClause(converter, devPtrClause->v, operands, useDeviceTypes,
                           useDeviceLocs, useDeviceSymbols);
      });
}
} // namespace omp
} // namespace lower
} // namespace Fortran
