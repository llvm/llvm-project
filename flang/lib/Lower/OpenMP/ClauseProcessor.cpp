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

#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/tools.h"
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
  using SimpleModifier = Allocate::AllocatorSimpleModifier;
  using ComplexModifier = Allocate::AllocatorComplexModifier;
  if (auto &mod = std::get<std::optional<SimpleModifier>>(clause.t)) {
    mlir::Value operand = fir::getBase(converter.genExprValue(*mod, stmtCtx));
    allocatorOperands.append(objects.size(), operand);
  } else if (auto &mod = std::get<std::optional<ComplexModifier>>(clause.t)) {
    mlir::Value operand = fir::getBase(converter.genExprValue(mod->v, stmtCtx));
    allocatorOperands.append(objects.size(), operand);
  } else {
    mlir::Value operand = firOpBuilder.createIntegerConstant(
        currentLocation, firOpBuilder.getI32Type(), 1);
    allocatorOperands.append(objects.size(), operand);
  }

  genObjectList(objects, converter, allocateOperands);
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
genDependKindAttr(fir::FirOpBuilder &firOpBuilder,
                  const omp::clause::Depend::TaskDependenceType kind) {
  mlir::omp::ClauseTaskDepend pbKind;
  switch (kind) {
  case omp::clause::Depend::TaskDependenceType::In:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependin;
    break;
  case omp::clause::Depend::TaskDependenceType::Out:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependout;
    break;
  case omp::clause::Depend::TaskDependenceType::Inout:
    pbKind = mlir::omp::ClauseTaskDepend::taskdependinout;
    break;
  case omp::clause::Depend::TaskDependenceType::Mutexinoutset:
  case omp::clause::Depend::TaskDependenceType::Inoutset:
  case omp::clause::Depend::TaskDependenceType::Depobj:
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
    llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) {
  genObjectList(objects, converter, operands);
  for (mlir::Value &operand : operands) {
    checkMapType(operand.getLoc(), operand.getType());
    useDeviceTypes.push_back(operand.getType());
    useDeviceLocs.push_back(operand.getLoc());
  }
  for (const omp::Object &object : objects)
    useDeviceSyms.push_back(object.sym());
}

static void convertLoopBounds(lower::AbstractConverter &converter,
                              mlir::Location loc,
                              mlir::omp::CollapseClauseOps &result,
                              std::size_t loopVarTypeSize) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  // The types of lower bound, upper bound, and step are converted into the
  // type of the loop variable if necessary.
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  for (unsigned it = 0; it < (unsigned)result.loopLBVar.size(); it++) {
    result.loopLBVar[it] =
        firOpBuilder.createConvert(loc, loopVarType, result.loopLBVar[it]);
    result.loopUBVar[it] =
        firOpBuilder.createConvert(loc, loopVarType, result.loopUBVar[it]);
    result.loopStepVar[it] =
        firOpBuilder.createConvert(loc, loopVarType, result.loopStepVar[it]);
  }
}

//===----------------------------------------------------------------------===//
// ClauseProcessor unique clauses
//===----------------------------------------------------------------------===//

bool ClauseProcessor::processCollapse(
    mlir::Location currentLocation, lower::pft::Evaluation &eval,
    mlir::omp::CollapseClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &iv) const {
  bool found = false;
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Collect the loops to collapse.
  lower::pft::Evaluation *doConstructEval = &eval.getFirstNestedEvaluation();
  if (doConstructEval->getIf<parser::DoConstruct>()->IsDoConcurrent()) {
    TODO(currentLocation, "Do Concurrent in Worksharing loop construct");
  }

  std::int64_t collapseValue = 1l;
  if (auto *clause = findUniqueClause<omp::clause::Collapse>()) {
    collapseValue = evaluate::ToInt64(clause->v).value();
    found = true;
  }

  std::size_t loopVarTypeSize = 0;
  do {
    lower::pft::Evaluation *doLoop =
        &doConstructEval->getFirstNestedEvaluation();
    auto *doStmt = doLoop->getIf<parser::NonLabelDoStmt>();
    assert(doStmt && "Expected do loop to be in the nested evaluation");
    const auto &loopControl =
        std::get<std::optional<parser::LoopControl>>(doStmt->t);
    const parser::LoopControl::Bounds *bounds =
        std::get_if<parser::LoopControl::Bounds>(&loopControl->u);
    assert(bounds && "Expected bounds for worksharing do loop");
    lower::StatementContext stmtCtx;
    result.loopLBVar.push_back(fir::getBase(
        converter.genExprValue(*semantics::GetExpr(bounds->lower), stmtCtx)));
    result.loopUBVar.push_back(fir::getBase(
        converter.genExprValue(*semantics::GetExpr(bounds->upper), stmtCtx)));
    if (bounds->step) {
      result.loopStepVar.push_back(fir::getBase(
          converter.genExprValue(*semantics::GetExpr(bounds->step), stmtCtx)));
    } else { // If `step` is not present, assume it as `1`.
      result.loopStepVar.push_back(firOpBuilder.createIntegerConstant(
          currentLocation, firOpBuilder.getIntegerType(32), 1));
    }
    iv.push_back(bounds->name.thing.symbol);
    loopVarTypeSize = std::max(loopVarTypeSize,
                               bounds->name.thing.symbol->GetUltimate().size());
    collapseValue--;
    doConstructEval =
        &*std::next(doConstructEval->getNestedEvaluations().begin());
  } while (collapseValue > 0);

  convertLoopBounds(converter, currentLocation, result, loopVarTypeSize);

  return found;
}

bool ClauseProcessor::processDefault() const {
  if (auto *clause = findUniqueClause<omp::clause::Default>()) {
    // Private, Firstprivate, Shared, None
    switch (clause->v) {
    case omp::clause::Default::DataSharingAttribute::Shared:
    case omp::clause::Default::DataSharingAttribute::None:
      // Default clause with shared or none do not require any handling since
      // Shared is the default behavior in the IR and None is only required
      // for semantic checks.
      break;
    case omp::clause::Default::DataSharingAttribute::Private:
      // TODO Support default(private)
      break;
    case omp::clause::Default::DataSharingAttribute::Firstprivate:
      // TODO Support default(firstprivate)
      break;
    }
    return true;
  }
  return false;
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
    result.deviceVar =
        fir::getBase(converter.genExprValue(deviceExpr, stmtCtx));
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
    result.distScheduleStaticAttr = converter.getFirOpBuilder().getUnitAttr();
    const auto &chunkSize = std::get<std::optional<ExprTy>>(clause->t);
    if (chunkSize)
      result.distScheduleChunkSizeVar =
          fir::getBase(converter.genExprValue(*chunkSize, stmtCtx));
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
    result.finalVar = firOpBuilder.createConvert(
        clauseLocation, firOpBuilder.getI1Type(), finalVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processHint(mlir::omp::HintClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Hint>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    int64_t hintValue = *evaluate::ToInt64(clause->v);
    result.hintAttr = firOpBuilder.getI64IntegerAttr(hintValue);
    return true;
  }
  return false;
}

bool ClauseProcessor::processMergeable(
    mlir::omp::MergeableClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Mergeable>(result.mergeableAttr);
}

bool ClauseProcessor::processNowait(mlir::omp::NowaitClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Nowait>(result.nowaitAttr);
}

bool ClauseProcessor::processNumTeams(
    lower::StatementContext &stmtCtx,
    mlir::omp::NumTeamsClauseOps &result) const {
  // TODO Get lower and upper bounds for num_teams when parser is updated to
  // accept both.
  if (auto *clause = findUniqueClause<omp::clause::NumTeams>()) {
    // auto lowerBound = std::get<std::optional<ExprTy>>(clause->t);
    auto &upperBound = std::get<ExprTy>(clause->t);
    result.numTeamsUpperVar =
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
    result.numThreadsVar =
        fir::getBase(converter.genExprValue(clause->v, stmtCtx));
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
    result.orderedAttr = firOpBuilder.getI64IntegerAttr(orderedClauseValue);
    return true;
  }
  return false;
}

bool ClauseProcessor::processPriority(
    lower::StatementContext &stmtCtx,
    mlir::omp::PriorityClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Priority>()) {
    result.priorityVar =
        fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processProcBind(
    mlir::omp::ProcBindClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::ProcBind>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    result.procBindKindAttr = genProcBindKindAttr(firOpBuilder, *clause);
    return true;
  }
  return false;
}

bool ClauseProcessor::processSafelen(
    mlir::omp::SafelenClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Safelen>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    const std::optional<std::int64_t> safelenVal = evaluate::ToInt64(clause->v);
    result.safelenAttr = firOpBuilder.getI64IntegerAttr(*safelenVal);
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

    result.scheduleValAttr =
        mlir::omp::ClauseScheduleKindAttr::get(context, scheduleKind);

    mlir::omp::ScheduleModifier scheduleModifier = getScheduleModifier(*clause);
    if (scheduleModifier != mlir::omp::ScheduleModifier::none)
      result.scheduleModAttr =
          mlir::omp::ScheduleModifierAttr::get(context, scheduleModifier);

    if (getSimdModifier(*clause) != mlir::omp::ScheduleModifier::none)
      result.scheduleSimdAttr = firOpBuilder.getUnitAttr();

    if (const auto &chunkExpr = std::get<omp::MaybeExpr>(clause->t))
      result.scheduleChunkVar =
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
    result.simdlenAttr = firOpBuilder.getI64IntegerAttr(*simdlenVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processThreadLimit(
    lower::StatementContext &stmtCtx,
    mlir::omp::ThreadLimitClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::ThreadLimit>()) {
    result.threadLimitVar =
        fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processUntied(mlir::omp::UntiedClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Untied>(result.untiedAttr);
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
                 llvm::SmallVectorImpl<mlir::Attribute> &alignmentAttrs) {
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
    auto &objects = std::get<omp::ObjectList>(clause.t);
    if (!objects.empty())
      genObjectList(objects, converter, alignedVars);
    alignmentValueAttr = builder.getI64IntegerAttr(alignment);
    // All the list items in a aligned clause will have same alignment
    for (std::size_t i = 0; i < objects.size(); i++)
      alignmentAttrs.push_back(alignmentValueAttr);
  }
}

bool ClauseProcessor::processAligned(
    mlir::omp::AlignedClauseOps &result) const {
  return findRepeatableClause<omp::clause::Aligned>(
      [&](const omp::clause::Aligned &clause, const parser::CharBlock &) {
        addAlignedClause(converter, clause, result.alignedVars,
                         result.alignmentAttrs);
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
          if (semantics::IsAllocatableOrObjectPointer(&sym->GetUltimate()))
            TODO(converter.getCurrentLocation(),
                 "pointer or allocatable variables in Copyin clause");
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
  converter.copyVar(loc, declDst.getBase(), declSrc.getBase());
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
    result.copyprivateFuncs.push_back(mlir::SymbolRefAttr::get(funcOp));
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

bool ClauseProcessor::processDepend(mlir::omp::DependClauseOps &result) const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  return findRepeatableClause<omp::clause::Depend>(
      [&](const omp::clause::Depend &clause, const parser::CharBlock &) {
        using Depend = omp::clause::Depend;
        assert(std::holds_alternative<Depend::WithLocators>(clause.u) &&
               "Only the modern form is handled at the moment");
        auto &modern = std::get<Depend::WithLocators>(clause.u);
        auto kind = std::get<Depend::TaskDependenceType>(modern.t);
        auto &objects = std::get<omp::ObjectList>(modern.t);

        mlir::omp::ClauseTaskDependAttr dependTypeOperand =
            genDependKindAttr(firOpBuilder, kind);
        result.dependTypeAttrs.append(objects.size(), dependTypeOperand);

        for (const omp::Object &object : objects) {
          assert(object.ref() && "Expecting designator");

          if (evaluate::ExtractSubstring(*object.ref())) {
            TODO(converter.getCurrentLocation(),
                 "substring not supported for task depend");
          } else if (evaluate::IsArrayElement(*object.ref())) {
            TODO(converter.getCurrentLocation(),
                 "array sections not supported for task depend");
          }

          semantics::Symbol *sym = object.sym();
          const mlir::Value variable = converter.getSymbolAddress(*sym);
          result.dependVars.push_back(variable);
        }
      });
}

bool ClauseProcessor::processHasDeviceAddr(
    mlir::omp::HasDeviceAddrClauseOps &result,
    llvm::SmallVectorImpl<mlir::Type> &isDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &isDeviceLocs,
    llvm::SmallVectorImpl<const semantics::Symbol *> &isDeviceSymbols) const {
  return findRepeatableClause<omp::clause::HasDeviceAddr>(
      [&](const omp::clause::HasDeviceAddr &devAddrClause,
          const parser::CharBlock &) {
        addUseDeviceClause(converter, devAddrClause.v, result.hasDeviceAddrVars,
                           isDeviceTypes, isDeviceLocs, isDeviceSymbols);
      });
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
      result.ifVar = operand;
      found = true;
    }
  });
  return found;
}

bool ClauseProcessor::processIsDevicePtr(
    mlir::omp::IsDevicePtrClauseOps &result,
    llvm::SmallVectorImpl<mlir::Type> &isDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &isDeviceLocs,
    llvm::SmallVectorImpl<const semantics::Symbol *> &isDeviceSymbols) const {
  return findRepeatableClause<omp::clause::IsDevicePtr>(
      [&](const omp::clause::IsDevicePtr &devPtrClause,
          const parser::CharBlock &) {
        addUseDeviceClause(converter, devPtrClause.v, result.isDevicePtrVars,
                           isDeviceTypes, isDeviceLocs, isDeviceSymbols);
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

bool ClauseProcessor::processMap(
    mlir::Location currentLocation, lower::StatementContext &stmtCtx,
    mlir::omp::MapClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> *mapSyms,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes) const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  // We always require tracking of symbols, even if the caller does not,
  // so we create an optionally used local set of symbols when the mapSyms
  // argument is not present.
  llvm::SmallVector<const semantics::Symbol *> localMapSyms;
  llvm::SmallVectorImpl<const semantics::Symbol *> *ptrMapSyms =
      mapSyms ? mapSyms : &localMapSyms;
  std::map<const semantics::Symbol *,
           llvm::SmallVector<OmpMapMemberIndicesData>>
      parentMemberIndices;

  bool clauseFound = findRepeatableClause<omp::clause::Map>(
      [&](const omp::clause::Map &clause, const parser::CharBlock &source) {
        using Map = omp::clause::Map;
        mlir::Location clauseLocation = converter.genLocation(source);
        const auto &mapType = std::get<std::optional<Map::MapType>>(clause.t);
        llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE;
        // If the map type is specified, then process it else Tofrom is the
        // default.
        if (mapType) {
          switch (*mapType) {
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

          auto &modTypeMods =
              std::get<std::optional<Map::MapTypeModifiers>>(clause.t);
          if (modTypeMods) {
            if (llvm::is_contained(*modTypeMods, Map::MapTypeModifier::Always))
              mapTypeBits |=
                  llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_ALWAYS;
          }
        } else {
          mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
                         llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
        }

        for (const omp::Object &object : std::get<omp::ObjectList>(clause.t)) {
          llvm::SmallVector<mlir::Value> bounds;
          std::stringstream asFortran;

          lower::AddrAndBoundsInfo info =
              lower::gatherDataOperandAddrAndBounds<mlir::omp::MapBoundsOp,
                                                    mlir::omp::MapBoundsType>(
                  converter, firOpBuilder, semaCtx, stmtCtx, *object.sym(),
                  object.ref(), clauseLocation, asFortran, bounds,
                  treatIndexAsSection);

          auto origSymbol = converter.getSymbolAddress(*object.sym());
          mlir::Value symAddr = info.addr;
          if (origSymbol && fir::isTypeWithDescriptor(origSymbol.getType()))
            symAddr = origSymbol;

          // Explicit map captures are captured ByRef by default,
          // optimisation passes may alter this to ByCopy or other capture
          // types to optimise
          auto location = mlir::NameLoc::get(
              mlir::StringAttr::get(firOpBuilder.getContext(), asFortran.str()),
              symAddr.getLoc());
          mlir::omp::MapInfoOp mapOp = createMapInfoOp(
              firOpBuilder, location, symAddr,
              /*varPtrPtr=*/mlir::Value{}, asFortran.str(), bounds,
              /*members=*/{}, /*membersIndex=*/mlir::DenseIntElementsAttr{},
              static_cast<
                  std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                  mapTypeBits),
              mlir::omp::VariableCaptureKind::ByRef, symAddr.getType());

          if (object.sym()->owner().IsDerivedType()) {
            addChildIndexAndMapToParent(object, parentMemberIndices, mapOp,
                                        semaCtx);
          } else {
            result.mapVars.push_back(mapOp);
            ptrMapSyms->push_back(object.sym());
            if (mapSymTypes)
              mapSymTypes->push_back(symAddr.getType());
            if (mapSymLocs)
              mapSymLocs->push_back(symAddr.getLoc());
          }
        }
      });

  insertChildMapInfoIntoParent(converter, parentMemberIndices, result.mapVars,
                               *ptrMapSyms, mapSymTypes, mapSymLocs);

  return clauseFound;
}

bool ClauseProcessor::processReduction(
    mlir::Location currentLocation, mlir::omp::ReductionClauseOps &result,
    llvm::SmallVectorImpl<mlir::Type> *outReductionTypes,
    llvm::SmallVectorImpl<const semantics::Symbol *> *outReductionSyms) const {
  return findRepeatableClause<omp::clause::Reduction>(
      [&](const omp::clause::Reduction &clause, const parser::CharBlock &) {
        llvm::SmallVector<mlir::Value> reductionVars;
        llvm::SmallVector<bool> reduceVarByRef;
        llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;
        llvm::SmallVector<const semantics::Symbol *> reductionSyms;
        ReductionProcessor rp;
        rp.addDeclareReduction(
            currentLocation, converter, clause, reductionVars, reduceVarByRef,
            reductionDeclSymbols, outReductionSyms ? &reductionSyms : nullptr);

        // Copy local lists into the output.
        llvm::copy(reductionVars, std::back_inserter(result.reductionVars));
        llvm::copy(reduceVarByRef, std::back_inserter(result.reduceVarByRef));
        llvm::copy(reductionDeclSymbols,
                   std::back_inserter(result.reductionDeclSymbols));

        if (outReductionTypes) {
          outReductionTypes->reserve(outReductionTypes->size() +
                                     reductionVars.size());
          llvm::transform(reductionVars, std::back_inserter(*outReductionTypes),
                          [](mlir::Value v) { return v.getType(); });
        }

        if (outReductionSyms)
          llvm::copy(reductionSyms, std::back_inserter(*outReductionSyms));
      });
}

bool ClauseProcessor::processSectionsReduction(
    mlir::Location currentLocation, mlir::omp::ReductionClauseOps &) const {
  return findRepeatableClause<omp::clause::Reduction>(
      [&](const omp::clause::Reduction &, const parser::CharBlock &) {
        TODO(currentLocation, "OMPC_Reduction");
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
    mlir::omp::UseDeviceClauseOps &result,
    llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const {
  return findRepeatableClause<omp::clause::UseDeviceAddr>(
      [&](const omp::clause::UseDeviceAddr &clause, const parser::CharBlock &) {
        addUseDeviceClause(converter, clause.v, result.useDeviceAddrVars,
                           useDeviceTypes, useDeviceLocs, useDeviceSyms);
      });
}

bool ClauseProcessor::processUseDevicePtr(
    mlir::omp::UseDeviceClauseOps &result,
    llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const {
  return findRepeatableClause<omp::clause::UseDevicePtr>(
      [&](const omp::clause::UseDevicePtr &clause, const parser::CharBlock &) {
        addUseDeviceClause(converter, clause.v, result.useDevicePtrVars,
                           useDeviceTypes, useDeviceLocs, useDeviceSyms);
      });
}

} // namespace omp
} // namespace lower
} // namespace Fortran
