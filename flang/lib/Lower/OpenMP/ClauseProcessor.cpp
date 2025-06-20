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
#include "Clauses.h"
#include "Utils.h"

#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/tools.h"
#include "llvm/Frontend/OpenMP/OMP.h.inc"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"

namespace Fortran {
namespace lower {
namespace omp {

/// Check for unsupported map operand types.
static void checkMapType(mlir::Location location, mlir::Type type) {
  if (auto refType = mlir::dyn_cast<fir::ReferenceType>(type))
    type = refType.getElementType();
  if (auto boxType = mlir::dyn_cast_or_null<fir::BoxType>(type))
    if (!mlir::isa<fir::PointerType>(boxType.getElementType()))
      TODO(location, "OMPD_target_data MapOperand BoxType");
}

static mlir::omp::ScheduleModifier
translateScheduleModifier(const omp::clause::Schedule::OrderingModifier &m) {
  switch (m) {
  case omp::clause::Schedule::OrderingModifier::Monotonic:
    return mlir::omp::ScheduleModifier::monotonic;
  case omp::clause::Schedule::OrderingModifier::Nonmonotonic:
    return mlir::omp::ScheduleModifier::nonmonotonic;
  }
  return mlir::omp::ScheduleModifier::none;
}

static mlir::omp::ScheduleModifier
getScheduleModifier(const omp::clause::Schedule &clause) {
  using Schedule = omp::clause::Schedule;
  const auto &modifier =
      std::get<std::optional<Schedule::OrderingModifier>>(clause.t);
  if (modifier)
    return translateScheduleModifier(*modifier);
  return mlir::omp::ScheduleModifier::none;
}

static mlir::omp::ScheduleModifier
getSimdModifier(const omp::clause::Schedule &clause) {
  using Schedule = omp::clause::Schedule;
  const auto &modifier =
      std::get<std::optional<Schedule::ChunkModifier>>(clause.t);
  if (modifier && *modifier == Schedule::ChunkModifier::Simd)
    return mlir::omp::ScheduleModifier::simd;
  return mlir::omp::ScheduleModifier::none;
}

static void
genAllocateClause(lower::AbstractConverter &converter,
                  const omp::clause::Allocate &clause,
                  llvm::SmallVectorImpl<mlir::Value> &allocatorOperands,
                  llvm::SmallVectorImpl<mlir::Value> &allocateOperands) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  lower::StatementContext stmtCtx;

  auto &objects = std::get<omp::ObjectList>(clause.t);

  using Allocate = omp::clause::Allocate;
  // ALIGN in this context is unimplemented
  if (std::get<std::optional<Allocate::AlignModifier>>(clause.t))
    TODO(currentLocation, "OmpAllocateClause ALIGN modifier");

  // Check if allocate clause has allocator specified. If so, add it
  // to list of allocators, otherwise, add default allocator to
  // list of allocators.
  using ComplexModifier = Allocate::AllocatorComplexModifier;
  if (auto &mod = std::get<std::optional<ComplexModifier>>(clause.t)) {
    mlir::Value operand = fir::getBase(converter.genExprValue(mod->v, stmtCtx));
    allocatorOperands.append(objects.size(), operand);
  } else {
    mlir::Value operand = firOpBuilder.createIntegerConstant(
        currentLocation, firOpBuilder.getI32Type(), 1);
    allocatorOperands.append(objects.size(), operand);
  }

  genObjectList(objects, converter, allocateOperands);
}

static mlir::omp::ClauseBindKindAttr
genBindKindAttr(fir::FirOpBuilder &firOpBuilder,
                const omp::clause::Bind &clause) {
  mlir::omp::ClauseBindKind bindKind;
  switch (clause.v) {
  case omp::clause::Bind::Binding::Teams:
    bindKind = mlir::omp::ClauseBindKind::Teams;
    break;
  case omp::clause::Bind::Binding::Parallel:
    bindKind = mlir::omp::ClauseBindKind::Parallel;
    break;
  case omp::clause::Bind::Binding::Thread:
    bindKind = mlir::omp::ClauseBindKind::Thread;
    break;
  }
  return mlir::omp::ClauseBindKindAttr::get(firOpBuilder.getContext(),
                                            bindKind);
}

static mlir::omp::ClauseProcBindKindAttr
genProcBindKindAttr(fir::FirOpBuilder &firOpBuilder,
                    const omp::clause::ProcBind &clause) {
  mlir::omp::ClauseProcBindKind procBindKind;
  switch (clause.v) {
  case omp::clause::ProcBind::AffinityPolicy::Master:
    procBindKind = mlir::omp::ClauseProcBindKind::Master;
    break;
  case omp::clause::ProcBind::AffinityPolicy::Close:
    procBindKind = mlir::omp::ClauseProcBindKind::Close;
    break;
  case omp::clause::ProcBind::AffinityPolicy::Spread:
    procBindKind = mlir::omp::ClauseProcBindKind::Spread;
    break;
  case omp::clause::ProcBind::AffinityPolicy::Primary:
    procBindKind = mlir::omp::ClauseProcBindKind::Primary;
    break;
  }
  return mlir::omp::ClauseProcBindKindAttr::get(firOpBuilder.getContext(),
                                                procBindKind);
}

static mlir::omp::ClauseTaskDependAttr
genDependKindAttr(lower::AbstractConverter &converter,
                  const omp::clause::DependenceType kind) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  mlir::omp::ClauseTaskDepend pbKind;
  switch (kind) {
  case omp::clause::DependenceType::In:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependin;
    break;
  case omp::clause::DependenceType::Out:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependout;
    break;
  case omp::clause::DependenceType::Inout:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependinout;
    break;
  case omp::clause::DependenceType::Mutexinoutset:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependmutexinoutset;
    break;
  case omp::clause::DependenceType::Inoutset:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependinoutset;
    break;
  case omp::clause::DependenceType::Depobj:
    TODO(currentLocation, "DEPOBJ dependence-type");
    break;
  case omp::clause::DependenceType::Sink:
  case omp::clause::DependenceType::Source:
    llvm_unreachable("unhandled parser task dependence type");
    break;
  }
  return mlir::omp::ClauseTaskDependAttr::get(firOpBuilder.getContext(),
                                              pbKind);
}

static mlir::Value
getIfClauseOperand(lower::AbstractConverter &converter,
                   const omp::clause::If &clause,
                   omp::clause::If::DirectiveNameModifier directiveName,
                   mlir::Location clauseLocation) {
  // Only consider the clause if it's intended for the given directive.
  auto &directive =
      std::get<std::optional<omp::clause::If::DirectiveNameModifier>>(clause.t);
  if (directive && directive.value() != directiveName)
    return nullptr;

  lower::StatementContext stmtCtx;
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Value ifVal = fir::getBase(
      converter.genExprValue(std::get<omp::SomeExpr>(clause.t), stmtCtx));
  return firOpBuilder.createConvert(clauseLocation, firOpBuilder.getI1Type(),
                                    ifVal);
}

static void addUseDeviceClause(
    lower::AbstractConverter &converter, const omp::ObjectList &objects,
    llvm::SmallVectorImpl<mlir::Value> &operands,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) {
  genObjectList(objects, converter, operands);
  for (mlir::Value &operand : operands)
    checkMapType(operand.getLoc(), operand.getType());

  for (const omp::Object &object : objects)
    useDeviceSyms.push_back(object.sym());
}

//===----------------------------------------------------------------------===//
// ClauseProcessor unique clauses
//===----------------------------------------------------------------------===//

bool ClauseProcessor::processBare(mlir::omp::BareClauseOps &result) const {
  return markClauseOccurrence<omp::clause::OmpxBare>(result.bare);
}

bool ClauseProcessor::processBind(mlir::omp::BindClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Bind>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    result.bindKind = genBindKindAttr(firOpBuilder, *clause);
    return true;
  }
  return false;
}

bool ClauseProcessor::processCancelDirectiveName(
    mlir::omp::CancelDirectiveNameClauseOps &result) const {
  using ConstructType = mlir::omp::ClauseCancellationConstructType;
  mlir::MLIRContext *context = &converter.getMLIRContext();

  ConstructType directive;
  if (auto *clause = findUniqueClause<omp::CancellationConstructType>()) {
    switch (clause->v) {
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_Parallel:
      directive = mlir::omp::ClauseCancellationConstructType::Parallel;
      break;
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_Loop:
      directive = mlir::omp::ClauseCancellationConstructType::Loop;
      break;
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_Sections:
      directive = mlir::omp::ClauseCancellationConstructType::Sections;
      break;
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_Taskgroup:
      directive = mlir::omp::ClauseCancellationConstructType::Taskgroup;
      break;
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_None:
      llvm_unreachable("OMP_CANCELLATION_CONSTRUCT_None");
      break;
    }
  } else {
    llvm_unreachable("cancel construct missing cancellation construct type");
  }

  result.cancelDirective =
      mlir::omp::ClauseCancellationConstructTypeAttr::get(context, directive);
  return true;
}

bool ClauseProcessor::processCollapse(
    mlir::Location currentLocation, lower::pft::Evaluation &eval,
    mlir::omp::LoopRelatedClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &iv) const {
  return collectLoopRelatedInfo(converter, currentLocation, eval, clauses,
                                result, iv);
}

bool ClauseProcessor::processDevice(lower::StatementContext &stmtCtx,
                                    mlir::omp::DeviceClauseOps &result) const {
  const parser::CharBlock *source = nullptr;
  if (auto *clause = findUniqueClause<omp::clause::Device>(&source)) {
    mlir::Location clauseLocation = converter.genLocation(*source);
    if (auto deviceModifier =
            std::get<std::optional<omp::clause::Device::DeviceModifier>>(
                clause->t)) {
      if (deviceModifier == omp::clause::Device::DeviceModifier::Ancestor) {
        TODO(clauseLocation, "OMPD_target Device Modifier Ancestor");
      }
    }
    const auto &deviceExpr = std::get<omp::SomeExpr>(clause->t);
    result.device = fir::getBase(converter.genExprValue(deviceExpr, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processDeviceType(
    mlir::omp::DeviceTypeClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::DeviceType>()) {
    // Case: declare target ... device_type(any | host | nohost)
    switch (clause->v) {
    case omp::clause::DeviceType::DeviceTypeDescription::Nohost:
      result.deviceType = mlir::omp::DeclareTargetDeviceType::nohost;
      break;
    case omp::clause::DeviceType::DeviceTypeDescription::Host:
      result.deviceType = mlir::omp::DeclareTargetDeviceType::host;
      break;
    case omp::clause::DeviceType::DeviceTypeDescription::Any:
      result.deviceType = mlir::omp::DeclareTargetDeviceType::any;
      break;
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processDistSchedule(
    lower::StatementContext &stmtCtx,
    mlir::omp::DistScheduleClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::DistSchedule>()) {
    result.distScheduleStatic = converter.getFirOpBuilder().getUnitAttr();
    const auto &chunkSize = std::get<std::optional<ExprTy>>(clause->t);
    if (chunkSize)
      result.distScheduleChunkSize =
          fir::getBase(converter.genExprValue(*chunkSize, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processExclusive(
    mlir::Location currentLocation,
    mlir::omp::ExclusiveClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Exclusive>()) {
    for (const Object &object : clause->v) {
      const semantics::Symbol *symbol = object.sym();
      mlir::Value symVal = converter.getSymbolAddress(*symbol);
      result.exclusiveVars.push_back(symVal);
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processFilter(lower::StatementContext &stmtCtx,
                                    mlir::omp::FilterClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Filter>()) {
    result.filteredThreadId =
        fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processFinal(lower::StatementContext &stmtCtx,
                                   mlir::omp::FinalClauseOps &result) const {
  const parser::CharBlock *source = nullptr;
  if (auto *clause = findUniqueClause<omp::clause::Final>(&source)) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    mlir::Location clauseLocation = converter.genLocation(*source);

    mlir::Value finalVal =
        fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    result.final = firOpBuilder.createConvert(
        clauseLocation, firOpBuilder.getI1Type(), finalVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processHint(mlir::omp::HintClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Hint>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    int64_t hintValue = *evaluate::ToInt64(clause->v);
    result.hint = firOpBuilder.getI64IntegerAttr(hintValue);
    return true;
  }
  return false;
}

bool ClauseProcessor::processInclusive(
    mlir::Location currentLocation,
    mlir::omp::InclusiveClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Inclusive>()) {
    for (const Object &object : clause->v) {
      const semantics::Symbol *symbol = object.sym();
      mlir::Value symVal = converter.getSymbolAddress(*symbol);
      result.inclusiveVars.push_back(symVal);
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processMergeable(
    mlir::omp::MergeableClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Mergeable>(result.mergeable);
}

bool ClauseProcessor::processNowait(mlir::omp::NowaitClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Nowait>(result.nowait);
}

bool ClauseProcessor::processNumTasks(
    lower::StatementContext &stmtCtx,
    mlir::omp::NumTasksClauseOps &result) const {
  using NumTasks = omp::clause::NumTasks;
  if (auto *clause = findUniqueClause<NumTasks>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    mlir::MLIRContext *context = firOpBuilder.getContext();
    const auto &modifier =
        std::get<std::optional<NumTasks::Prescriptiveness>>(clause->t);
    if (modifier && *modifier == NumTasks::Prescriptiveness::Strict) {
      result.numTasksMod = mlir::omp::ClauseNumTasksTypeAttr::get(
          context, mlir::omp::ClauseNumTasksType::Strict);
    }
    const auto &numtasksExpr = std::get<omp::SomeExpr>(clause->t);
    result.numTasks =
        fir::getBase(converter.genExprValue(numtasksExpr, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processNumTeams(
    lower::StatementContext &stmtCtx,
    mlir::omp::NumTeamsClauseOps &result) const {
  // TODO Get lower and upper bounds for num_teams when parser is updated to
  // accept both.
  if (auto *clause = findUniqueClause<omp::clause::NumTeams>()) {
    // The num_teams directive accepts a list of team lower/upper bounds.
    // This is an extension to support grid specification for ompx_bare.
    // Here, only expect a single element in the list.
    assert(clause->v.size() == 1);
    // auto lowerBound = std::get<std::optional<ExprTy>>(clause->v[0]->t);
    auto &upperBound = std::get<ExprTy>(clause->v[0].t);
    result.numTeamsUpper =
        fir::getBase(converter.genExprValue(upperBound, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processNumThreads(
    lower::StatementContext &stmtCtx,
    mlir::omp::NumThreadsClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::NumThreads>()) {
    // OMPIRBuilder expects `NUM_THREADS` clause as a `Value`.
    result.numThreads =
        fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processOrder(mlir::omp::OrderClauseOps &result) const {
  using Order = omp::clause::Order;
  if (auto *clause = findUniqueClause<Order>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    result.order = mlir::omp::ClauseOrderKindAttr::get(
        firOpBuilder.getContext(), mlir::omp::ClauseOrderKind::Concurrent);
    const auto &modifier =
        std::get<std::optional<Order::OrderModifier>>(clause->t);
    if (modifier && *modifier == Order::OrderModifier::Unconstrained) {
      result.orderMod = mlir::omp::OrderModifierAttr::get(
          firOpBuilder.getContext(), mlir::omp::OrderModifier::unconstrained);
    } else {
      // "If order-modifier is not unconstrained, the behavior is as if the
      // reproducible modifier is present."
      result.orderMod = mlir::omp::OrderModifierAttr::get(
          firOpBuilder.getContext(), mlir::omp::OrderModifier::reproducible);
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processOrdered(
    mlir::omp::OrderedClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Ordered>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    int64_t orderedClauseValue = 0l;
    if (clause->v.has_value())
      orderedClauseValue = *evaluate::ToInt64(*clause->v);
    result.ordered = firOpBuilder.getI64IntegerAttr(orderedClauseValue);
    return true;
  }
  return false;
}

bool ClauseProcessor::processPriority(
    lower::StatementContext &stmtCtx,
    mlir::omp::PriorityClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Priority>()) {
    result.priority = fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processDetach(mlir::omp::DetachClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Detach>()) {
    semantics::Symbol *sym = clause->v.sym();
    mlir::Value symVal = converter.getSymbolAddress(*sym);
    result.eventHandle = symVal;
    return true;
  }
  return false;
}

bool ClauseProcessor::processProcBind(
    mlir::omp::ProcBindClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::ProcBind>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    result.procBindKind = genProcBindKindAttr(firOpBuilder, *clause);
    return true;
  }
  return false;
}

bool ClauseProcessor::processSafelen(
    mlir::omp::SafelenClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Safelen>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    const std::optional<std::int64_t> safelenVal = evaluate::ToInt64(clause->v);
    result.safelen = firOpBuilder.getI64IntegerAttr(*safelenVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processSchedule(
    lower::StatementContext &stmtCtx,
    mlir::omp::ScheduleClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Schedule>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    mlir::MLIRContext *context = firOpBuilder.getContext();
    const auto &scheduleType = std::get<omp::clause::Schedule::Kind>(clause->t);

    mlir::omp::ClauseScheduleKind scheduleKind;
    switch (scheduleType) {
    case omp::clause::Schedule::Kind::Static:
      scheduleKind = mlir::omp::ClauseScheduleKind::Static;
      break;
    case omp::clause::Schedule::Kind::Dynamic:
      scheduleKind = mlir::omp::ClauseScheduleKind::Dynamic;
      break;
    case omp::clause::Schedule::Kind::Guided:
      scheduleKind = mlir::omp::ClauseScheduleKind::Guided;
      break;
    case omp::clause::Schedule::Kind::Auto:
      scheduleKind = mlir::omp::ClauseScheduleKind::Auto;
      break;
    case omp::clause::Schedule::Kind::Runtime:
      scheduleKind = mlir::omp::ClauseScheduleKind::Runtime;
      break;
    }

    result.scheduleKind =
        mlir::omp::ClauseScheduleKindAttr::get(context, scheduleKind);

    mlir::omp::ScheduleModifier scheduleMod = getScheduleModifier(*clause);
    if (scheduleMod != mlir::omp::ScheduleModifier::none)
      result.scheduleMod =
          mlir::omp::ScheduleModifierAttr::get(context, scheduleMod);

    if (getSimdModifier(*clause) != mlir::omp::ScheduleModifier::none)
      result.scheduleSimd = firOpBuilder.getUnitAttr();

    if (const auto &chunkExpr = std::get<omp::MaybeExpr>(clause->t))
      result.scheduleChunk =
          fir::getBase(converter.genExprValue(*chunkExpr, stmtCtx));

    return true;
  }
  return false;
}

bool ClauseProcessor::processSimdlen(
    mlir::omp::SimdlenClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Simdlen>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    const std::optional<std::int64_t> simdlenVal = evaluate::ToInt64(clause->v);
    result.simdlen = firOpBuilder.getI64IntegerAttr(*simdlenVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processThreadLimit(
    lower::StatementContext &stmtCtx,
    mlir::omp::ThreadLimitClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::ThreadLimit>()) {
    result.threadLimit =
        fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processUntied(mlir::omp::UntiedClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Untied>(result.untied);
}

//===----------------------------------------------------------------------===//
// ClauseProcessor repeatable clauses
//===----------------------------------------------------------------------===//
static llvm::StringMap<bool> getTargetFeatures(mlir::ModuleOp module) {
  llvm::StringMap<bool> featuresMap;
  llvm::SmallVector<llvm::StringRef> targetFeaturesVec;
  if (mlir::LLVM::TargetFeaturesAttr features =
          fir::getTargetFeatures(module)) {
    llvm::ArrayRef<mlir::StringAttr> featureAttrs = features.getFeatures();
    for (auto &featureAttr : featureAttrs) {
      llvm::StringRef featureKeyString = featureAttr.strref();
      featuresMap[featureKeyString.substr(1)] = (featureKeyString[0] == '+');
    }
  }
  return featuresMap;
}

static void
addAlignedClause(lower::AbstractConverter &converter,
                 const omp::clause::Aligned &clause,
                 llvm::SmallVectorImpl<mlir::Value> &alignedVars,
                 llvm::SmallVectorImpl<mlir::Attribute> &alignments) {
  using Aligned = omp::clause::Aligned;
  lower::StatementContext stmtCtx;
  mlir::IntegerAttr alignmentValueAttr;
  int64_t alignment = 0;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  if (auto &alignmentValueParserExpr =
          std::get<std::optional<Aligned::Alignment>>(clause.t)) {
    mlir::Value operand = fir::getBase(
        converter.genExprValue(*alignmentValueParserExpr, stmtCtx));
    alignment = *fir::getIntIfConstant(operand);
  } else {
    llvm::StringMap<bool> featuresMap = getTargetFeatures(builder.getModule());
    llvm::Triple triple = fir::getTargetTriple(builder.getModule());
    alignment =
        llvm::OpenMPIRBuilder::getOpenMPDefaultSimdAlign(triple, featuresMap);
  }

  // The default alignment for some targets is equal to 0.
  // Do not generate alignment assumption if alignment is less than or equal to
  // 0.
  if (alignment > 0) {
    // alignment value must be power of 2
    assert((alignment & (alignment - 1)) == 0 && "alignment is not power of 2");
    auto &objects = std::get<omp::ObjectList>(clause.t);
    if (!objects.empty())
      genObjectList(objects, converter, alignedVars);
    alignmentValueAttr = builder.getI64IntegerAttr(alignment);
    // All the list items in a aligned clause will have same alignment
    for (std::size_t i = 0; i < objects.size(); i++)
      alignments.push_back(alignmentValueAttr);
  }
}

bool ClauseProcessor::processAligned(
    mlir::omp::AlignedClauseOps &result) const {
  return findRepeatableClause<omp::clause::Aligned>(
      [&](const omp::clause::Aligned &clause, const parser::CharBlock &) {
        addAlignedClause(converter, clause, result.alignedVars,
                         result.alignments);
      });
}

bool ClauseProcessor::processAllocate(
    mlir::omp::AllocateClauseOps &result) const {
  return findRepeatableClause<omp::clause::Allocate>(
      [&](const omp::clause::Allocate &clause, const parser::CharBlock &) {
        genAllocateClause(converter, clause, result.allocatorVars,
                          result.allocateVars);
      });
}

bool ClauseProcessor::processCopyin() const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());
  auto checkAndCopyHostAssociateVar =
      [&](semantics::Symbol *sym,
          mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr) {
        assert(sym->has<semantics::HostAssocDetails>() &&
               "No host-association found");
        if (converter.isPresentShallowLookup(*sym))
          converter.copyHostAssociateVar(*sym, copyAssignIP);
      };
  bool hasCopyin = findRepeatableClause<omp::clause::Copyin>(
      [&](const omp::clause::Copyin &clause, const parser::CharBlock &) {
        for (const omp::Object &object : clause.v) {
          semantics::Symbol *sym = object.sym();
          assert(sym && "Expecting symbol");
          if (const auto *commonDetails =
                  sym->detailsIf<semantics::CommonBlockDetails>()) {
            for (const auto &mem : commonDetails->objects())
              checkAndCopyHostAssociateVar(&*mem, &insPt);
            break;
          }

          assert(sym->has<semantics::HostAssocDetails>() &&
                 "No host-association found");
          checkAndCopyHostAssociateVar(sym);
        }
      });

  // [OMP 5.0, 2.19.6.1] The copy is done after the team is formed and prior to
  // the execution of the associated structured block. Emit implicit barrier to
  // synchronize threads and avoid data races on propagation master's thread
  // values of threadprivate variables to local instances of that variables of
  // all other implicit threads.

  // All copies are inserted at either "insPt" (i.e. immediately before it),
  // or at some earlier point (as determined by "copyHostAssociateVar").
  // Unless the insertion point is given to "copyHostAssociateVar" explicitly,
  // it will not restore the builder's insertion point. Since the copies may be
  // inserted in any order (not following the execution order), make sure the
  // barrier is inserted following all of them.
  firOpBuilder.restoreInsertionPoint(insPt);
  if (hasCopyin)
    firOpBuilder.create<mlir::omp::BarrierOp>(converter.getCurrentLocation());
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
  llvm::ArrayRef<int64_t> getShape() const { return shape; }

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
  } else if (auto cty = mlir::dyn_cast<fir::ClassType>(ty)) {
    inBox = true;
    typeScan(cty.getEleTy());
  } else if (auto cty = mlir::dyn_cast<fir::CharacterType>(ty)) {
    charLen = cty.getLen();
  } else if (auto hty = mlir::dyn_cast<fir::HeapType>(ty)) {
    typeScan(hty.getEleTy());
  } else if (auto pty = mlir::dyn_cast<fir::PointerType>(ty)) {
    typeScan(pty.getEleTy());
  } else {
    // The scan ends when reaching any built-in, record or boxproc type.
    assert(ty.isIntOrIndexOrFloat() || mlir::isa<mlir::ComplexType>(ty) ||
           mlir::isa<fir::LogicalType>(ty) || mlir::isa<fir::RecordType>(ty) ||
           mlir::isa<fir::BoxProcType>(ty));
  }
}

// Create a function that performs a copy between two variables, compatible
// with their types and attributes.
static mlir::func::FuncOp
createCopyFunc(mlir::Location loc, lower::AbstractConverter &converter,
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
  fir::factory::setInternalLinkage(funcOp);
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
  auto declDst = builder.create<hlfir::DeclareOp>(
      loc, funcOp.getArgument(0), copyFuncName + "_dst", shape, typeparams,
      /*dummy_scope=*/nullptr, attrs);
  auto declSrc = builder.create<hlfir::DeclareOp>(
      loc, funcOp.getArgument(1), copyFuncName + "_src", shape, typeparams,
      /*dummy_scope=*/nullptr, attrs);
  converter.copyVar(loc, declDst.getBase(), declSrc.getBase(), varAttrs);
  builder.create<mlir::func::ReturnOp>(loc);
  return funcOp;
}

bool ClauseProcessor::processCopyprivate(
    mlir::Location currentLocation,
    mlir::omp::CopyprivateClauseOps &result) const {
  auto addCopyPrivateVar = [&](semantics::Symbol *sym) {
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

    result.copyprivateVars.push_back(cpVar);
    mlir::func::FuncOp funcOp =
        createCopyFunc(currentLocation, converter, cpVar.getType(), attrs);
    result.copyprivateSyms.push_back(mlir::SymbolRefAttr::get(funcOp));
  };

  bool hasCopyPrivate = findRepeatableClause<clause::Copyprivate>(
      [&](const clause::Copyprivate &clause, const parser::CharBlock &) {
        for (const Object &object : clause.v) {
          semantics::Symbol *sym = object.sym();
          if (const auto *commonDetails =
                  sym->detailsIf<semantics::CommonBlockDetails>()) {
            for (const auto &mem : commonDetails->objects())
              addCopyPrivateVar(&*mem);
            break;
          }
          addCopyPrivateVar(sym);
        }
      });

  return hasCopyPrivate;
}

template <typename T>
static bool isVectorSubscript(const evaluate::Expr<T> &expr) {
  if (std::optional<evaluate::DataRef> dataRef{evaluate::ExtractDataRef(expr)})
    if (const auto *arrayRef = std::get_if<evaluate::ArrayRef>(&dataRef->u))
      for (const evaluate::Subscript &subscript : arrayRef->subscript())
        if (std::holds_alternative<evaluate::IndirectSubscriptIntegerExpr>(
                subscript.u))
          if (subscript.Rank() > 0)
            return true;
  return false;
}

bool ClauseProcessor::processDefaultMap(lower::StatementContext &stmtCtx,
                                        DefaultMapsTy &result) const {
  auto process = [&](const omp::clause::Defaultmap &clause,
                     const parser::CharBlock &) {
    using Defmap = omp::clause::Defaultmap;
    clause::Defaultmap::VariableCategory variableCategory =
        Defmap::VariableCategory::All;
    // Variable Category is optional, if not specified defaults to all.
    // Multiples of the same category are illegal as are any other
    // defaultmaps being specified when a user specified all is in place,
    // however, this should be handled earlier during semantics.
    if (auto varCat =
            std::get<std::optional<Defmap::VariableCategory>>(clause.t))
      variableCategory = varCat.value();
    auto behaviour = std::get<Defmap::ImplicitBehavior>(clause.t);
    result[variableCategory] = behaviour;
  };
  return findRepeatableClause<omp::clause::Defaultmap>(process);
}

bool ClauseProcessor::processDepend(lower::SymMap &symMap,
                                    lower::StatementContext &stmtCtx,
                                    mlir::omp::DependClauseOps &result) const {
  auto process = [&](const omp::clause::Depend &clause,
                     const parser::CharBlock &) {
    using Depend = omp::clause::Depend;
    if (!std::holds_alternative<Depend::TaskDep>(clause.u)) {
      TODO(converter.getCurrentLocation(),
           "DEPEND clause with SINK or SOURCE is not supported yet");
    }
    auto &taskDep = std::get<Depend::TaskDep>(clause.u);
    auto depType = std::get<clause::DependenceType>(taskDep.t);
    auto &objects = std::get<omp::ObjectList>(taskDep.t);
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();

    if (std::get<std::optional<omp::clause::Iterator>>(taskDep.t)) {
      TODO(converter.getCurrentLocation(),
           "Support for iterator modifiers is not implemented yet");
    }
    mlir::omp::ClauseTaskDependAttr dependTypeOperand =
        genDependKindAttr(converter, depType);
    result.dependKinds.append(objects.size(), dependTypeOperand);

    for (const omp::Object &object : objects) {
      assert(object.ref() && "Expecting designator");
      mlir::Value dependVar;
      SomeExpr expr = *object.ref();

      if (evaluate::IsArrayElement(expr) || evaluate::ExtractSubstring(expr)) {
        // Array Section or character (sub)string
        if (isVectorSubscript(expr)) {
          // OpenMP needs the address of the first indexed element (required by
          // the standard to be the lowest index) to identify the dependency. We
          // don't need an accurate length for the array section because the
          // OpenMP standard forbids overlapping array sections.
          dependVar = genVectorSubscriptedDesignatorFirstElementAddress(
              converter.getCurrentLocation(), converter, expr, symMap, stmtCtx);
        } else {
          // Ordinary array section e.g. A(1:512:2)
          hlfir::EntityWithAttributes entity = convertExprToHLFIR(
              converter.getCurrentLocation(), converter, expr, symMap, stmtCtx);
          dependVar = entity.getBase();
        }
      } else if (evaluate::isStructureComponent(expr) ||
                 evaluate::ExtractComplexPart(expr)) {
        SomeExpr expr = *object.ref();
        hlfir::EntityWithAttributes entity = convertExprToHLFIR(
            converter.getCurrentLocation(), converter, expr, symMap, stmtCtx);
        dependVar = entity.getBase();
      } else {
        semantics::Symbol *sym = object.sym();
        dependVar = converter.getSymbolAddress(*sym);
      }

      // If we pass a mutable box e.g. !fir.ref<!fir.box<!fir.heap<...>>> then
      // the runtime will use the address of the box not the address of the
      // data. Flang generates a lot of memcpys between different box
      // allocations so this is not a reliable way to identify the dependency.
      if (auto ref = mlir::dyn_cast<fir::ReferenceType>(dependVar.getType()))
        if (fir::isa_box_type(ref.getElementType()))
          dependVar = builder.create<fir::LoadOp>(
              converter.getCurrentLocation(), dependVar);

      // The openmp dialect doesn't know what to do with boxes (and it would
      // break layering to teach it about them). The dependency variable can be
      // a box because it was an array section or because the original symbol
      // was mapped to a box.
      // Getting the address of the box data is okay because all the runtime
      // ultimately cares about is the base address of the array.
      if (fir::isa_box_type(dependVar.getType()))
        dependVar = builder.create<fir::BoxAddrOp>(
            converter.getCurrentLocation(), dependVar);

      result.dependVars.push_back(dependVar);
    }
  };

  return findRepeatableClause<omp::clause::Depend>(process);
}

bool ClauseProcessor::processGrainsize(
    lower::StatementContext &stmtCtx,
    mlir::omp::GrainsizeClauseOps &result) const {
  using Grainsize = omp::clause::Grainsize;
  if (auto *clause = findUniqueClause<Grainsize>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    mlir::MLIRContext *context = firOpBuilder.getContext();
    const auto &modifier =
        std::get<std::optional<Grainsize::Prescriptiveness>>(clause->t);
    if (modifier && *modifier == Grainsize::Prescriptiveness::Strict) {
      result.grainsizeMod = mlir::omp::ClauseGrainsizeTypeAttr::get(
          context, mlir::omp::ClauseGrainsizeType::Strict);
    }
    const auto &grainsizeExpr = std::get<omp::SomeExpr>(clause->t);
    result.grainsize =
        fir::getBase(converter.genExprValue(grainsizeExpr, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processHasDeviceAddr(
    lower::StatementContext &stmtCtx, mlir::omp::HasDeviceAddrClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &hasDeviceSyms) const {
  // For HAS_DEVICE_ADDR objects, implicitly map the top-level entities.
  // Their address (or the whole descriptor, if the entity had one) will be
  // passed to the target region.
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;
  bool clauseFound = findRepeatableClause<omp::clause::HasDeviceAddr>(
      [&](const omp::clause::HasDeviceAddr &clause,
          const parser::CharBlock &source) {
        mlir::Location location = converter.genLocation(source);
        llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
        omp::ObjectList baseObjects;
        llvm::transform(clause.v, std::back_inserter(baseObjects),
                        [&](const omp::Object &object) {
                          if (auto maybeBase = getBaseObject(object, semaCtx))
                            return *maybeBase;
                          return object;
                        });
        processMapObjects(stmtCtx, location, baseObjects, mapTypeBits,
                          parentMemberIndices, result.hasDeviceAddrVars,
                          hasDeviceSyms);
      });

  insertChildMapInfoIntoParent(converter, semaCtx, stmtCtx, parentMemberIndices,
                               result.hasDeviceAddrVars, hasDeviceSyms);
  return clauseFound;
}

bool ClauseProcessor::processIf(
    omp::clause::If::DirectiveNameModifier directiveName,
    mlir::omp::IfClauseOps &result) const {
  bool found = false;
  findRepeatableClause<omp::clause::If>([&](const omp::clause::If &clause,
                                            const parser::CharBlock &source) {
    mlir::Location clauseLocation = converter.genLocation(source);
    mlir::Value operand =
        getIfClauseOperand(converter, clause, directiveName, clauseLocation);
    // Assume that, at most, a single 'if' clause will be applicable to the
    // given directive.
    if (operand) {
      result.ifExpr = operand;
      found = true;
    }
  });
  return found;
}
bool ClauseProcessor::processInReduction(
    mlir::Location currentLocation, mlir::omp::InReductionClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &outReductionSyms) const {
  return findRepeatableClause<omp::clause::InReduction>(
      [&](const omp::clause::InReduction &clause, const parser::CharBlock &) {
        llvm::SmallVector<mlir::Value> inReductionVars;
        llvm::SmallVector<bool> inReduceVarByRef;
        llvm::SmallVector<mlir::Attribute> inReductionDeclSymbols;
        llvm::SmallVector<const semantics::Symbol *> inReductionSyms;
        ReductionProcessor rp;
        rp.processReductionArguments<omp::clause::InReduction>(
            currentLocation, converter, clause, inReductionVars,
            inReduceVarByRef, inReductionDeclSymbols, inReductionSyms);

        // Copy local lists into the output.
        llvm::copy(inReductionVars, std::back_inserter(result.inReductionVars));
        llvm::copy(inReduceVarByRef,
                   std::back_inserter(result.inReductionByref));
        llvm::copy(inReductionDeclSymbols,
                   std::back_inserter(result.inReductionSyms));
        llvm::copy(inReductionSyms, std::back_inserter(outReductionSyms));
      });
}

bool ClauseProcessor::processIsDevicePtr(
    mlir::omp::IsDevicePtrClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &isDeviceSyms) const {
  return findRepeatableClause<omp::clause::IsDevicePtr>(
      [&](const omp::clause::IsDevicePtr &devPtrClause,
          const parser::CharBlock &) {
        addUseDeviceClause(converter, devPtrClause.v, result.isDevicePtrVars,
                           isDeviceSyms);
      });
}

bool ClauseProcessor::processLinear(mlir::omp::LinearClauseOps &result) const {
  lower::StatementContext stmtCtx;
  return findRepeatableClause<
      omp::clause::Linear>([&](const omp::clause::Linear &clause,
                               const parser::CharBlock &) {
    auto &objects = std::get<omp::ObjectList>(clause.t);
    for (const omp::Object &object : objects) {
      semantics::Symbol *sym = object.sym();
      const mlir::Value variable = converter.getSymbolAddress(*sym);
      result.linearVars.push_back(variable);
    }
    if (objects.size()) {
      if (auto &mod =
              std::get<std::optional<omp::clause::Linear::StepComplexModifier>>(
                  clause.t)) {
        mlir::Value operand =
            fir::getBase(converter.genExprValue(toEvExpr(*mod), stmtCtx));
        result.linearStepVars.append(objects.size(), operand);
      } else if (std::get<std::optional<omp::clause::Linear::LinearModifier>>(
                     clause.t)) {
        mlir::Location currentLocation = converter.getCurrentLocation();
        TODO(currentLocation, "Linear modifiers not yet implemented");
      } else {
        // If nothing is present, add the default step of 1.
        fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
        mlir::Location currentLocation = converter.getCurrentLocation();
        mlir::Value operand = firOpBuilder.createIntegerConstant(
            currentLocation, firOpBuilder.getI32Type(), 1);
        result.linearStepVars.append(objects.size(), operand);
      }
    }
  });
}

bool ClauseProcessor::processLink(
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const {
  return findRepeatableClause<omp::clause::Link>(
      [&](const omp::clause::Link &clause, const parser::CharBlock &) {
        // Case: declare target link(var1, var2)...
        gatherFuncAndVarSyms(
            clause.v, mlir::omp::DeclareTargetCaptureClause::link, result);
      });
}

void ClauseProcessor::processMapObjects(
    lower::StatementContext &stmtCtx, mlir::Location clauseLocation,
    const omp::ObjectList &objects,
    llvm::omp::OpenMPOffloadMappingFlags mapTypeBits,
    std::map<Object, OmpMapParentAndMemberData> &parentMemberIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapVars,
    llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms,
    llvm::StringRef mapperIdNameRef) const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  auto getDefaultMapperID = [&](const omp::Object &object,
                                std::string &mapperIdName) {
    if (!mlir::isa<mlir::omp::DeclareMapperOp>(
            firOpBuilder.getRegion().getParentOp())) {
      const semantics::DerivedTypeSpec *typeSpec = nullptr;

      if (object.sym()->owner().IsDerivedType())
        typeSpec = object.sym()->owner().derivedTypeSpec();
      else if (object.sym()->GetType() &&
               object.sym()->GetType()->category() ==
                   semantics::DeclTypeSpec::TypeDerived)
        typeSpec = &object.sym()->GetType()->derivedTypeSpec();

      if (typeSpec) {
        mapperIdName =
            typeSpec->name().ToString() + llvm::omp::OmpDefaultMapperName;
        if (auto *sym = converter.getCurrentScope().FindSymbol(mapperIdName))
          mapperIdName = converter.mangleName(mapperIdName, sym->owner());
      }
    }
  };

  // Create the mapper symbol from its name, if specified.
  mlir::FlatSymbolRefAttr mapperId;
  if (!mapperIdNameRef.empty() && !objects.empty() &&
      mapperIdNameRef != "__implicit_mapper") {
    std::string mapperIdName = mapperIdNameRef.str();
    const omp::Object &object = objects.front();
    if (mapperIdNameRef == "default")
      getDefaultMapperID(object, mapperIdName);
    assert(converter.getModuleOp().lookupSymbol(mapperIdName) &&
           "mapper not found");
    mapperId =
        mlir::FlatSymbolRefAttr::get(&converter.getMLIRContext(), mapperIdName);
  }

  for (const omp::Object &object : objects) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    std::optional<omp::Object> parentObj;

    fir::factory::AddrAndBoundsInfo info =
        lower::gatherDataOperandAddrAndBounds<mlir::omp::MapBoundsOp,
                                              mlir::omp::MapBoundsType>(
            converter, firOpBuilder, semaCtx, stmtCtx, *object.sym(),
            object.ref(), clauseLocation, asFortran, bounds,
            treatIndexAsSection);

    mlir::Value baseOp = info.rawInput;
    if (object.sym()->owner().IsDerivedType()) {
      omp::ObjectList objectList = gatherObjectsOf(object, semaCtx);
      assert(!objectList.empty() &&
             "could not find parent objects of derived type member");
      parentObj = objectList[0];
      parentMemberIndices.emplace(parentObj.value(),
                                  OmpMapParentAndMemberData{});

      if (isMemberOrParentAllocatableOrPointer(object, semaCtx)) {
        llvm::SmallVector<int64_t> indices;
        generateMemberPlacementIndices(object, indices, semaCtx);
        baseOp = createParentSymAndGenIntermediateMaps(
            clauseLocation, converter, semaCtx, stmtCtx, objectList, indices,
            parentMemberIndices[parentObj.value()], asFortran.str(),
            mapTypeBits);
      }
    }

    if (mapperIdNameRef == "__implicit_mapper") {
      std::string mapperIdName;
      getDefaultMapperID(object, mapperIdName);
      mapperId = converter.getModuleOp().lookupSymbol(mapperIdName)
                     ? mlir::FlatSymbolRefAttr::get(&converter.getMLIRContext(),
                                                    mapperIdName)
                     : mlir::FlatSymbolRefAttr();
    }

    // Explicit map captures are captured ByRef by default,
    // optimisation passes may alter this to ByCopy or other capture
    // types to optimise
    auto location = mlir::NameLoc::get(
        mlir::StringAttr::get(firOpBuilder.getContext(), asFortran.str()),
        baseOp.getLoc());
    mlir::omp::MapInfoOp mapOp = createMapInfoOp(
        firOpBuilder, location, baseOp,
        /*varPtrPtr=*/mlir::Value{}, asFortran.str(), bounds,
        /*members=*/{}, /*membersIndex=*/mlir::ArrayAttr{},
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            mapTypeBits),
        mlir::omp::VariableCaptureKind::ByRef, baseOp.getType(),
        /*partialMap=*/false, mapperId);

    if (parentObj.has_value()) {
      parentMemberIndices[parentObj.value()].addChildIndexAndMapToParent(
          object, mapOp, semaCtx);
    } else {
      mapVars.push_back(mapOp);
      mapSyms.push_back(object.sym());
    }
  }
}

bool ClauseProcessor::processMap(
    mlir::Location currentLocation, lower::StatementContext &stmtCtx,
    mlir::omp::MapClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> *mapSyms) const {
  // We always require tracking of symbols, even if the caller does not,
  // so we create an optionally used local set of symbols when the mapSyms
  // argument is not present.
  llvm::SmallVector<const semantics::Symbol *> localMapSyms;
  llvm::SmallVectorImpl<const semantics::Symbol *> *ptrMapSyms =
      mapSyms ? mapSyms : &localMapSyms;
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;

  auto process = [&](const omp::clause::Map &clause,
                     const parser::CharBlock &source) {
    using Map = omp::clause::Map;
    mlir::Location clauseLocation = converter.genLocation(source);
    const auto &[mapType, typeMods, mappers, iterator, objects] = clause.t;
    llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE;
    std::string mapperIdName = "__implicit_mapper";
    // If the map type is specified, then process it else Tofrom is the
    // default.
    Map::MapType type = mapType.value_or(Map::MapType::Tofrom);
    switch (type) {
    case Map::MapType::To:
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
      break;
    case Map::MapType::From:
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
      break;
    case Map::MapType::Tofrom:
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
                     llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
      break;
    case Map::MapType::Alloc:
    case Map::MapType::Release:
      // alloc and release is the default map_type for the Target Data
      // Ops, i.e. if no bits for map_type is supplied then alloc/release
      // is implicitly assumed based on the target directive. Default
      // value for Target Data and Enter Data is alloc and for Exit Data
      // it is release.
      break;
    case Map::MapType::Delete:
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_DELETE;
    }

    if (typeMods) {
      // TODO: Still requires "self" modifier, an OpenMP 6.0+ feature
      if (llvm::is_contained(*typeMods, Map::MapTypeModifier::Always))
        mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_ALWAYS;
      if (llvm::is_contained(*typeMods, Map::MapTypeModifier::Present))
        mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PRESENT;
      if (llvm::is_contained(*typeMods, Map::MapTypeModifier::Close))
        mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_CLOSE;
      if (llvm::is_contained(*typeMods, Map::MapTypeModifier::OmpxHold))
        mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_OMPX_HOLD;
    }

    if (iterator) {
      TODO(currentLocation,
           "Support for iterator modifiers is not implemented yet");
    }
    if (mappers) {
      assert(mappers->size() == 1 && "more than one mapper");
      mapperIdName = mappers->front().v.id().symbol->name().ToString();
      if (mapperIdName != "default")
        mapperIdName = converter.mangleName(
            mapperIdName, mappers->front().v.id().symbol->owner());
    }

    processMapObjects(stmtCtx, clauseLocation,
                      std::get<omp::ObjectList>(clause.t), mapTypeBits,
                      parentMemberIndices, result.mapVars, *ptrMapSyms,
                      mapperIdName);
  };

  bool clauseFound = findRepeatableClause<omp::clause::Map>(process);
  insertChildMapInfoIntoParent(converter, semaCtx, stmtCtx, parentMemberIndices,
                               result.mapVars, *ptrMapSyms);

  return clauseFound;
}

bool ClauseProcessor::processMotionClauses(lower::StatementContext &stmtCtx,
                                           mlir::omp::MapClauseOps &result) {
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;
  llvm::SmallVector<const semantics::Symbol *> mapSymbols;

  auto callbackFn = [&](const auto &clause, const parser::CharBlock &source) {
    mlir::Location clauseLocation = converter.genLocation(source);
    const auto &[expectation, mapper, iterator, objects] = clause.t;

    // TODO Support motion modifiers: mapper, iterator.
    if (mapper) {
      TODO(clauseLocation, "Mapper modifier is not supported yet");
    } else if (iterator) {
      TODO(clauseLocation, "Iterator modifier is not supported yet");
    }

    llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
        std::is_same_v<llvm::remove_cvref_t<decltype(clause)>, omp::clause::To>
            ? llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO
            : llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
    if (expectation && *expectation == omp::clause::To::Expectation::Present)
      mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PRESENT;
    processMapObjects(stmtCtx, clauseLocation, objects, mapTypeBits,
                      parentMemberIndices, result.mapVars, mapSymbols);
  };

  bool clauseFound = findRepeatableClause<omp::clause::To>(callbackFn);
  clauseFound =
      findRepeatableClause<omp::clause::From>(callbackFn) || clauseFound;

  insertChildMapInfoIntoParent(converter, semaCtx, stmtCtx, parentMemberIndices,
                               result.mapVars, mapSymbols);

  return clauseFound;
}

bool ClauseProcessor::processNontemporal(
    mlir::omp::NontemporalClauseOps &result) const {
  return findRepeatableClause<omp::clause::Nontemporal>(
      [&](const omp::clause::Nontemporal &clause, const parser::CharBlock &) {
        for (const Object &object : clause.v) {
          semantics::Symbol *sym = object.sym();
          mlir::Value symVal = converter.getSymbolAddress(*sym);
          result.nontemporalVars.push_back(symVal);
        }
      });
}

bool ClauseProcessor::processReduction(
    mlir::Location currentLocation, mlir::omp::ReductionClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &outReductionSyms) const {
  return findRepeatableClause<omp::clause::Reduction>(
      [&](const omp::clause::Reduction &clause, const parser::CharBlock &) {
        llvm::SmallVector<mlir::Value> reductionVars;
        llvm::SmallVector<bool> reduceVarByRef;
        llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;
        llvm::SmallVector<const semantics::Symbol *> reductionSyms;
        ReductionProcessor rp;
        rp.processReductionArguments<omp::clause::Reduction>(
            currentLocation, converter, clause, reductionVars, reduceVarByRef,
            reductionDeclSymbols, reductionSyms, &result.reductionMod);
        // Copy local lists into the output.
        llvm::copy(reductionVars, std::back_inserter(result.reductionVars));
        llvm::copy(reduceVarByRef, std::back_inserter(result.reductionByref));
        llvm::copy(reductionDeclSymbols,
                   std::back_inserter(result.reductionSyms));
        llvm::copy(reductionSyms, std::back_inserter(outReductionSyms));
      });
}

bool ClauseProcessor::processTaskReduction(
    mlir::Location currentLocation, mlir::omp::TaskReductionClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &outReductionSyms) const {
  return findRepeatableClause<omp::clause::TaskReduction>(
      [&](const omp::clause::TaskReduction &clause, const parser::CharBlock &) {
        llvm::SmallVector<mlir::Value> taskReductionVars;
        llvm::SmallVector<bool> TaskReduceVarByRef;
        llvm::SmallVector<mlir::Attribute> TaskReductionDeclSymbols;
        llvm::SmallVector<const semantics::Symbol *> TaskReductionSyms;
        ReductionProcessor rp;
        rp.processReductionArguments<omp::clause::TaskReduction>(
            currentLocation, converter, clause, taskReductionVars,
            TaskReduceVarByRef, TaskReductionDeclSymbols, TaskReductionSyms);
        // Copy local lists into the output.
        llvm::copy(taskReductionVars,
                   std::back_inserter(result.taskReductionVars));
        llvm::copy(TaskReduceVarByRef,
                   std::back_inserter(result.taskReductionByref));
        llvm::copy(TaskReductionDeclSymbols,
                   std::back_inserter(result.taskReductionSyms));
        llvm::copy(TaskReductionSyms, std::back_inserter(outReductionSyms));
      });
}

bool ClauseProcessor::processTo(
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const {
  return findRepeatableClause<omp::clause::To>(
      [&](const omp::clause::To &clause, const parser::CharBlock &) {
        // Case: declare target to(func, var1, var2)...
        gatherFuncAndVarSyms(std::get<ObjectList>(clause.t),
                             mlir::omp::DeclareTargetCaptureClause::to, result);
      });
}

bool ClauseProcessor::processEnter(
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const {
  return findRepeatableClause<omp::clause::Enter>(
      [&](const omp::clause::Enter &clause, const parser::CharBlock &) {
        // Case: declare target enter(func, var1, var2)...
        gatherFuncAndVarSyms(
            clause.v, mlir::omp::DeclareTargetCaptureClause::enter, result);
      });
}

bool ClauseProcessor::processUseDeviceAddr(
    lower::StatementContext &stmtCtx, mlir::omp::UseDeviceAddrClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const {
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;
  bool clauseFound = findRepeatableClause<omp::clause::UseDeviceAddr>(
      [&](const omp::clause::UseDeviceAddr &clause,
          const parser::CharBlock &source) {
        mlir::Location location = converter.genLocation(source);
        llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_RETURN_PARAM;
        processMapObjects(stmtCtx, location, clause.v, mapTypeBits,
                          parentMemberIndices, result.useDeviceAddrVars,
                          useDeviceSyms);
      });

  insertChildMapInfoIntoParent(converter, semaCtx, stmtCtx, parentMemberIndices,
                               result.useDeviceAddrVars, useDeviceSyms);
  return clauseFound;
}

bool ClauseProcessor::processUseDevicePtr(
    lower::StatementContext &stmtCtx, mlir::omp::UseDevicePtrClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const {
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;

  bool clauseFound = findRepeatableClause<omp::clause::UseDevicePtr>(
      [&](const omp::clause::UseDevicePtr &clause,
          const parser::CharBlock &source) {
        mlir::Location location = converter.genLocation(source);
        llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_RETURN_PARAM;
        processMapObjects(stmtCtx, location, clause.v, mapTypeBits,
                          parentMemberIndices, result.useDevicePtrVars,
                          useDeviceSyms);
      });

  insertChildMapInfoIntoParent(converter, semaCtx, stmtCtx, parentMemberIndices,
                               result.useDevicePtrVars, useDeviceSyms);
  return clauseFound;
}

} // namespace omp
} // namespace lower
} // namespace Fortran
