//===-- ClauseProcessor.cpp -------------------------------------*- C++ -*-===//
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

#include "ClauseProcessor.h"
#include "Utils.h"

#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/ReductionProcessor.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/tools.h"
#include "flang/Utils/OpenMP.h"
#include "llvm/Frontend/OpenMP/OMP.h.inc"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"

namespace Fortran {
namespace lower {
namespace omp {

using ReductionModifier =
    Fortran::lower::omp::clause::Reduction::ReductionModifier;

aiir::omp::ReductionModifier translateReductionModifier(ReductionModifier mod) {
  switch (mod) {
  case ReductionModifier::Default:
    return aiir::omp::ReductionModifier::defaultmod;
  case ReductionModifier::Inscan:
    return aiir::omp::ReductionModifier::inscan;
  case ReductionModifier::Task:
    return aiir::omp::ReductionModifier::task;
  }
  return aiir::omp::ReductionModifier::defaultmod;
}

static aiir::omp::ScheduleModifier
translateScheduleModifier(const omp::clause::Schedule::OrderingModifier &m) {
  switch (m) {
  case omp::clause::Schedule::OrderingModifier::Monotonic:
    return aiir::omp::ScheduleModifier::monotonic;
  case omp::clause::Schedule::OrderingModifier::Nonmonotonic:
    return aiir::omp::ScheduleModifier::nonmonotonic;
  }
  return aiir::omp::ScheduleModifier::none;
}

static aiir::omp::ScheduleModifier
getScheduleModifier(const omp::clause::Schedule &clause) {
  using Schedule = omp::clause::Schedule;
  const auto &modifier =
      std::get<std::optional<Schedule::OrderingModifier>>(clause.t);
  if (modifier)
    return translateScheduleModifier(*modifier);
  return aiir::omp::ScheduleModifier::none;
}

static aiir::omp::ScheduleModifier
getSimdModifier(const omp::clause::Schedule &clause) {
  using Schedule = omp::clause::Schedule;
  const auto &modifier =
      std::get<std::optional<Schedule::ChunkModifier>>(clause.t);
  if (modifier && *modifier == Schedule::ChunkModifier::Simd)
    return aiir::omp::ScheduleModifier::simd;
  return aiir::omp::ScheduleModifier::none;
}

static void
genAllocateClause(lower::AbstractConverter &converter,
                  const omp::clause::Allocate &clause,
                  llvm::SmallVectorImpl<aiir::Value> &allocatorOperands,
                  llvm::SmallVectorImpl<aiir::Value> &allocateOperands) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  aiir::Location currentLocation = converter.getCurrentLocation();
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
    aiir::Value operand = fir::getBase(converter.genExprValue(mod->v, stmtCtx));
    allocatorOperands.append(objects.size(), operand);
  } else {
    aiir::Value operand = firOpBuilder.createIntegerConstant(
        currentLocation, firOpBuilder.getI32Type(), 1);
    allocatorOperands.append(objects.size(), operand);
  }

  genObjectList(objects, converter, allocateOperands);
}

static aiir::omp::ClauseBindKindAttr
genBindKindAttr(fir::FirOpBuilder &firOpBuilder,
                const omp::clause::Bind &clause) {
  aiir::omp::ClauseBindKind bindKind;
  switch (clause.v) {
  case omp::clause::Bind::Binding::Teams:
    bindKind = aiir::omp::ClauseBindKind::Teams;
    break;
  case omp::clause::Bind::Binding::Parallel:
    bindKind = aiir::omp::ClauseBindKind::Parallel;
    break;
  case omp::clause::Bind::Binding::Thread:
    bindKind = aiir::omp::ClauseBindKind::Thread;
    break;
  }
  return aiir::omp::ClauseBindKindAttr::get(firOpBuilder.getContext(),
                                            bindKind);
}

static aiir::omp::ClauseProcBindKindAttr
genProcBindKindAttr(fir::FirOpBuilder &firOpBuilder,
                    const omp::clause::ProcBind &clause) {
  aiir::omp::ClauseProcBindKind procBindKind;
  switch (clause.v) {
  case omp::clause::ProcBind::AffinityPolicy::Master:
    procBindKind = aiir::omp::ClauseProcBindKind::Master;
    break;
  case omp::clause::ProcBind::AffinityPolicy::Close:
    procBindKind = aiir::omp::ClauseProcBindKind::Close;
    break;
  case omp::clause::ProcBind::AffinityPolicy::Spread:
    procBindKind = aiir::omp::ClauseProcBindKind::Spread;
    break;
  case omp::clause::ProcBind::AffinityPolicy::Primary:
    procBindKind = aiir::omp::ClauseProcBindKind::Primary;
    break;
  }
  return aiir::omp::ClauseProcBindKindAttr::get(firOpBuilder.getContext(),
                                                procBindKind);
}

static aiir::omp::ClauseTaskDependAttr
genDependKindAttr(lower::AbstractConverter &converter,
                  const omp::clause::DependenceType kind) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  aiir::Location currentLocation = converter.getCurrentLocation();

  aiir::omp::ClauseTaskDepend pbKind;
  switch (kind) {
  case omp::clause::DependenceType::In:
    pbKind = aiir::omp::ClauseTaskDepend::taskdependin;
    break;
  case omp::clause::DependenceType::Out:
    pbKind = aiir::omp::ClauseTaskDepend::taskdependout;
    break;
  case omp::clause::DependenceType::Inout:
    pbKind = aiir::omp::ClauseTaskDepend::taskdependinout;
    break;
  case omp::clause::DependenceType::Mutexinoutset:
    pbKind = aiir::omp::ClauseTaskDepend::taskdependmutexinoutset;
    break;
  case omp::clause::DependenceType::Inoutset:
    pbKind = aiir::omp::ClauseTaskDepend::taskdependinoutset;
    break;
  case omp::clause::DependenceType::Depobj:
    TODO(currentLocation, "DEPOBJ dependence-type");
    break;
  case omp::clause::DependenceType::Sink:
  case omp::clause::DependenceType::Source:
    llvm_unreachable("unhandled parser task dependence type");
    break;
  }
  return aiir::omp::ClauseTaskDependAttr::get(firOpBuilder.getContext(),
                                              pbKind);
}

static aiir::Value
getIfClauseOperand(lower::AbstractConverter &converter,
                   const omp::clause::If &clause,
                   omp::clause::If::DirectiveNameModifier directiveName,
                   aiir::Location clauseLocation) {
  // Only consider the clause if it's intended for the given directive.
  auto &directive =
      std::get<std::optional<omp::clause::If::DirectiveNameModifier>>(clause.t);
  if (directive && directive.value() != directiveName)
    return nullptr;

  lower::StatementContext stmtCtx;
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  aiir::Value ifVal = fir::getBase(
      converter.genExprValue(std::get<omp::SomeExpr>(clause.t), stmtCtx));
  return firOpBuilder.createConvert(clauseLocation, firOpBuilder.getI1Type(),
                                    ifVal);
}

template <typename SomeType, typename IteratorSpecT>
static IteratorRange lowerIteratorRange(
    Fortran::lower::AbstractConverter &converter, const IteratorSpecT &itSpec,
    Fortran::lower::StatementContext &stmtCtx, aiir::Location loc) {
  auto &builder = converter.getFirOpBuilder();

  using IdTy =
      Fortran::lower::omp::IdTyTemplate<Fortran::evaluate::Expr<SomeType>>;
  using ExprTy = Fortran::evaluate::Expr<SomeType>;

  using ObjTy = tomp::type::ObjectT<IdTy, ExprTy>;
  using RangeTy = tomp::type::RangeT<ExprTy>;

  const ObjTy &ivObj = std::get<1>(itSpec.t);
  const RangeTy &range = std::get<2>(itSpec.t);

  IteratorRange r;
  r.ivSym = ivObj.sym();
  assert(r.ivSym && "expected iterator induction symbol");

  const auto &lbExpr = std::get<0>(range.t);
  const auto &ubExpr = std::get<1>(range.t);
  const auto &stExpr = std::get<2>(range.t);

  aiir::Value lbVal =
      fir::getBase(converter.genExprValue(toEvExpr(lbExpr), stmtCtx));
  aiir::Value ubVal =
      fir::getBase(converter.genExprValue(toEvExpr(ubExpr), stmtCtx));

  auto toIndex = [](fir::FirOpBuilder &builder, aiir::Location loc,
                    aiir::Value v) -> aiir::Value {
    if (v.getType().isIndex())
      return v;
    return fir::ConvertOp::create(builder, loc, builder.getIndexType(), v);
  };

  r.lb = toIndex(builder, loc, lbVal);
  r.ub = toIndex(builder, loc, ubVal);

  if (stExpr) {
    aiir::Value stVal =
        fir::getBase(converter.genExprValue(toEvExpr(*stExpr), stmtCtx));
    r.step = toIndex(builder, loc, stVal);
  } else {
    r.step = aiir::arith::ConstantIndexOp::create(builder, loc, 1);
  }

  return r;
}

template <typename BodyFn>
static aiir::Value buildIteratorOp(Fortran::lower::AbstractConverter &converter,
                                   aiir::Location loc, aiir::Type iterTy,
                                   llvm::ArrayRef<IteratorRange> ranges,
                                   BodyFn &&bodyGen) {

  auto &builder = converter.getFirOpBuilder();

  llvm::SmallVector<aiir::Value> lbs, ubs, steps;
  lbs.reserve(ranges.size());
  ubs.reserve(ranges.size());
  steps.reserve(ranges.size());
  for (auto &r : ranges) {
    lbs.push_back(r.lb);
    ubs.push_back(r.ub);
    steps.push_back(r.step);
  }

  auto itOp = aiir::omp::IteratorOp::create(
      builder, loc, iterTy, aiir::ValueRange{lbs}, aiir::ValueRange{ubs},
      aiir::ValueRange{steps});

  aiir::OpBuilder::InsertionGuard guard(builder);

  aiir::Region &reg = itOp.getRegion();
  aiir::Block *body = builder.createBlock(&reg);

  llvm::SmallVector<aiir::Value> ivs;
  ivs.reserve(ranges.size());
  for (size_t i = 0; i < ranges.size(); ++i)
    ivs.push_back(body->addArgument(builder.getIndexType(), loc));

  Fortran::lower::SymMap &symMap = converter.getSymbolMap();
  Fortran::lower::SymMapScope scope(symMap);
  for (size_t i = 0; i < ranges.size(); ++i) {
    aiir::Value ivVal = ivs[i];
    aiir::Type ivTy = converter.genType(*ranges[i].ivSym);
    if (ivVal.getType() != ivTy)
      ivVal = fir::ConvertOp::create(builder, loc, ivTy, ivVal);
    symMap.addSymbol(*ranges[i].ivSym, ivVal, /*force=*/true);
  }

  aiir::omp::YieldOp::create(builder, loc, bodyGen(builder, loc, ivs));

  return itOp.getResult();
}

template <typename ClauseTuple>
static void collectIteratorIVs(
    const ClauseTuple &clause, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::StatementContext &stmtCtx,
    llvm::SmallVectorImpl<IteratorRange> &iteratorRanges,
    llvm::SmallPtrSetImpl<const Fortran::semantics::Symbol *> &ivSyms) {
  auto &iteratorModifier =
      std::get<std::optional<omp::clause::Iterator>>(clause.t);
  if (!iteratorModifier.has_value())
    return;

  aiir::Location clauseLocation = converter.getCurrentLocation();
  const auto &iteratorModifierSpecs = *iteratorModifier;
  iteratorRanges.reserve(iteratorModifierSpecs.size());
  for (const auto &itSpec : iteratorModifierSpecs)
    iteratorRanges.push_back(lowerIteratorRange<Fortran::evaluate::SomeType>(
        converter, itSpec, stmtCtx, clauseLocation));

  for (const IteratorRange &r : iteratorRanges)
    ivSyms.insert(&r.ivSym->GetUltimate());
}

//===----------------------------------------------------------------------===//
// ClauseProcessor unique clauses
//===----------------------------------------------------------------------===//

bool ClauseProcessor::processBare(aiir::omp::BareClauseOps &result) const {
  return markClauseOccurrence<omp::clause::OmpxBare>(result.bare);
}

bool ClauseProcessor::processBind(aiir::omp::BindClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Bind>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    result.bindKind = genBindKindAttr(firOpBuilder, *clause);
    return true;
  }
  return false;
}

bool ClauseProcessor::processCancelDirectiveName(
    aiir::omp::CancelDirectiveNameClauseOps &result) const {
  using ConstructType = aiir::omp::ClauseCancellationConstructType;
  aiir::AIIRContext *context = &converter.getAIIRContext();

  ConstructType directive;
  if (auto *clause = findUniqueClause<omp::CancellationConstructType>()) {
    switch (clause->v) {
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_Parallel:
      directive = aiir::omp::ClauseCancellationConstructType::Parallel;
      break;
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_Loop:
      directive = aiir::omp::ClauseCancellationConstructType::Loop;
      break;
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_Sections:
      directive = aiir::omp::ClauseCancellationConstructType::Sections;
      break;
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_Taskgroup:
      directive = aiir::omp::ClauseCancellationConstructType::Taskgroup;
      break;
    case llvm::omp::OMP_CANCELLATION_CONSTRUCT_None:
      llvm_unreachable("OMP_CANCELLATION_CONSTRUCT_None");
      break;
    }
  } else {
    llvm_unreachable("cancel construct missing cancellation construct type");
  }

  result.cancelDirective =
      aiir::omp::ClauseCancellationConstructTypeAttr::get(context, directive);
  return true;
}

bool ClauseProcessor::processCollapse(
    aiir::Location currentLocation, lower::pft::Evaluation &eval,
    aiir::omp::LoopRelatedClauseOps &loopResult,
    aiir::omp::CollapseClauseOps &collapseResult,
    llvm::SmallVectorImpl<const semantics::Symbol *> &iv) const {

  int64_t numCollapse = collectLoopRelatedInfo(converter, currentLocation, eval,
                                               getNestedDoConstruct(eval),
                                               clauses, loopResult, iv);
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  collapseResult.collapseNumLoops = firOpBuilder.getI64IntegerAttr(numCollapse);
  return numCollapse > 1;
}

bool ClauseProcessor::processDevice(lower::StatementContext &stmtCtx,
                                    aiir::omp::DeviceClauseOps &result) const {
  const parser::CharBlock *source = nullptr;
  if (auto *clause = findUniqueClause<omp::clause::Device>(&source)) {
    aiir::Location clauseLocation = converter.genLocation(*source);
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
    aiir::omp::DeviceTypeClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::DeviceType>()) {
    // Case: declare target ... device_type(any | host | nohost)
    switch (clause->v) {
    case omp::clause::DeviceType::DeviceTypeDescription::Nohost:
      result.deviceType = aiir::omp::DeclareTargetDeviceType::nohost;
      break;
    case omp::clause::DeviceType::DeviceTypeDescription::Host:
      result.deviceType = aiir::omp::DeclareTargetDeviceType::host;
      break;
    case omp::clause::DeviceType::DeviceTypeDescription::Any:
      result.deviceType = aiir::omp::DeclareTargetDeviceType::any;
      break;
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processDistSchedule(
    lower::StatementContext &stmtCtx,
    aiir::omp::DistScheduleClauseOps &result) const {
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
    aiir::Location currentLocation,
    aiir::omp::ExclusiveClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Exclusive>()) {
    for (const Object &object : clause->v) {
      const semantics::Symbol *symbol = object.sym();
      aiir::Value symVal = converter.getSymbolAddress(*symbol);
      result.exclusiveVars.push_back(symVal);
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processFilter(lower::StatementContext &stmtCtx,
                                    aiir::omp::FilterClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Filter>()) {
    result.filteredThreadId =
        fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processFinal(lower::StatementContext &stmtCtx,
                                   aiir::omp::FinalClauseOps &result) const {
  const parser::CharBlock *source = nullptr;
  if (auto *clause = findUniqueClause<omp::clause::Final>(&source)) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    aiir::Location clauseLocation = converter.genLocation(*source);

    aiir::Value finalVal =
        fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    result.final = firOpBuilder.createConvert(
        clauseLocation, firOpBuilder.getI1Type(), finalVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processHint(aiir::omp::HintClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Hint>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    int64_t hintValue = *evaluate::ToInt64(clause->v);
    result.hint = firOpBuilder.getI64IntegerAttr(hintValue);
    return true;
  }
  return false;
}

bool ClauseProcessor::processInclusive(
    aiir::Location currentLocation,
    aiir::omp::InclusiveClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Inclusive>()) {
    for (const Object &object : clause->v) {
      const semantics::Symbol *symbol = object.sym();
      aiir::Value symVal = converter.getSymbolAddress(*symbol);
      result.inclusiveVars.push_back(symVal);
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processInitializer(
    lower::SymMap &symMap,
    ReductionProcessor::GenInitValueCBTy &genInitValueCB) const {
  if (auto *clause = findUniqueClause<omp::clause::Initializer>()) {
    genInitValueCB = [&, clause](fir::FirOpBuilder &builder, aiir::Location loc,
                                 aiir::Type type, aiir::Value ompOrig) {
      lower::SymMapScope scope(symMap);
      aiir::Value ompPrivVar;
      const StylizedInstance &inst = clause->v.front();

      for (const Object &object :
           std::get<StylizedInstance::Variables>(inst.t)) {
        aiir::Value addr;
        aiir::Type ompOrigType = ompOrig.getType();
        // Check for unsupported dynamic-length character reductions
        aiir::Type unwrappedType = fir::unwrapRefType(ompOrigType);
        if (aiir::isa<fir::BoxCharType>(unwrappedType)) {
          TODO(loc, "OpenMP reduction allocation for dynamic length character");
        }
        if (auto charTy = aiir::dyn_cast<fir::CharacterType>(unwrappedType)) {
          if (!charTy.hasConstantLen()) {
            TODO(loc,
                 "OpenMP reduction allocation for dynamic length character");
          }
        }
        // If ompOrig is already a reference, we can use it directly
        if (fir::isa_ref_type(ompOrigType)) {
          addr = ompOrig;
        } else {
          addr = builder.createTemporary(loc, ompOrigType);
          fir::StoreOp::create(builder, loc, ompOrig, addr);
        }
        fir::FortranVariableFlagsEnum extraFlags = {};
        fir::FortranVariableFlagsAttr attributes =
            Fortran::lower::translateSymbolAttributes(
                builder.getContext(), *object.sym(), extraFlags);
        std::string name = object.sym()->name().ToString();
        // Get length parameters for types that need them (e.g., characters).
        // Note: DeclareOp requires exactly one type parameter for non-boxed
        // characters, unlike EmboxOp which doesn't allow them for constant-len.
        llvm::SmallVector<aiir::Value> typeParams;
        if (hlfir::isFortranEntity(addr)) {
          hlfir::genLengthParameters(loc, builder, hlfir::Entity{addr},
                                     typeParams);
        }
        auto declareOp = hlfir::DeclareOp::create(builder, loc, addr, name,
                                                  nullptr, typeParams, nullptr,
                                                  nullptr, 0, attributes);
        if (name == "omp_priv")
          ompPrivVar = declareOp.getResult(0);
        symMap.addVariableDefinition(*object.sym(), declareOp);
      }

      // Lower the expression/function call
      lower::StatementContext stmtCtx;
      const semantics::SomeExpr &initExpr =
          std::get<StylizedInstance::Instance>(inst.t);
      aiir::Value result = common::visit(
          common::visitors{
              [&](const evaluate::ProcedureRef &procRef) -> aiir::Value {
                convertCallToHLFIR(loc, converter, procRef, std::nullopt,
                                   symMap, stmtCtx);
                auto privVal = fir::LoadOp::create(builder, loc, ompPrivVar);
                return privVal;
              },
              [&](const auto &expr) -> aiir::Value {
                aiir::Value exprResult = fir::getBase(convertExprToValue(
                    loc, converter, initExpr, symMap, stmtCtx));
                // Conversion can either give a value or a refrence to a value,
                // we need to return the reduction type, so an optional load may
                // be generated.
                if (auto refType = llvm::dyn_cast<fir::ReferenceType>(
                        exprResult.getType()))
                  if (ompPrivVar.getType() == refType)
                    exprResult = fir::LoadOp::create(builder, loc, exprResult);
                return exprResult;
              }},
          initExpr.u);
      stmtCtx.finalizeAndPop();
      return result;
    };
    return true;
  }
  TODO(converter.getCurrentLocation(),
       "declare reduction without an initializer clause is not yet "
       "supported");
}

bool ClauseProcessor::processMergeable(
    aiir::omp::MergeableClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Mergeable>(result.mergeable);
}

bool ClauseProcessor::processNogroup(
    aiir::omp::NogroupClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Nogroup>(result.nogroup);
}

bool ClauseProcessor::processNowait(aiir::omp::NowaitClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Nowait>(result.nowait);
}

bool ClauseProcessor::processNumTasks(
    lower::StatementContext &stmtCtx,
    aiir::omp::NumTasksClauseOps &result) const {
  using NumTasks = omp::clause::NumTasks;
  if (auto *clause = findUniqueClause<NumTasks>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    aiir::AIIRContext *context = firOpBuilder.getContext();
    const auto &modifier =
        std::get<std::optional<NumTasks::Prescriptiveness>>(clause->t);
    if (modifier && *modifier == NumTasks::Prescriptiveness::Strict) {
      result.numTasksMod = aiir::omp::ClauseNumTasksTypeAttr::get(
          context, aiir::omp::ClauseNumTasksType::Strict);
    }
    const auto &numtasksExpr = std::get<omp::SomeExpr>(clause->t);
    result.numTasks =
        fir::getBase(converter.genExprValue(numtasksExpr, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processSizes(StatementContext &stmtCtx,
                                   aiir::omp::SizesClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Sizes>()) {
    result.sizes.reserve(clause->v.size());
    for (const ExprTy &vv : clause->v)
      result.sizes.push_back(fir::getBase(converter.genExprValue(vv, stmtCtx)));

    return true;
  }

  return false;
}

bool ClauseProcessor::processLooprange(StatementContext &stmtCtx,
                                       aiir::omp::LooprangeClauseOps &result,
                                       int64_t &count) const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  if (auto *clause = findUniqueClause<omp::clause::Looprange>()) {
    int64_t first = evaluate::ToInt64(std::get<0>(clause->t)).value();
    count = evaluate::ToInt64(std::get<1>(clause->t)).value();
    result.first = firOpBuilder.getI64IntegerAttr(first);
    result.count = firOpBuilder.getI64IntegerAttr(count);
    return true;
  }

  return false;
}

bool ClauseProcessor::processNumTeams(
    lower::StatementContext &stmtCtx,
    aiir::omp::NumTeamsClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::NumTeams>()) {
    // Structure: {LB?, [UB]} - single optional lower bound, list of upper
    // bounds
    auto &lowerBound = std::get<std::optional<ExprTy>>(clause->t);
    auto &upperBounds =
        std::get<omp::clause::NumTeams::UpperBoundList>(clause->t);
    assert(!upperBounds.empty());

    // Extract optional lower bound
    if (lowerBound) {
      result.numTeamsLower =
          fir::getBase(converter.genExprValue(*lowerBound, stmtCtx));
    }

    // Extract all upper bounds
    result.numTeamsUpperVars.reserve(upperBounds.size());
    for (const auto &ub : upperBounds) {
      result.numTeamsUpperVars.push_back(
          fir::getBase(converter.genExprValue(ub, stmtCtx)));
    }

    return true;
  }
  return false;
}

bool ClauseProcessor::processNumThreads(
    lower::StatementContext &stmtCtx,
    aiir::omp::NumThreadsClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::NumThreads>()) {
    // OMPIRBuilder expects `NUM_THREADS` clause as a list of Values.
    for (const ExprTy &expr : clause->v) {
      result.numThreadsVars.push_back(
          fir::getBase(converter.genExprValue(expr, stmtCtx)));
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processOrder(aiir::omp::OrderClauseOps &result) const {
  using Order = omp::clause::Order;
  if (auto *clause = findUniqueClause<Order>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    result.order = aiir::omp::ClauseOrderKindAttr::get(
        firOpBuilder.getContext(), aiir::omp::ClauseOrderKind::Concurrent);
    const auto &modifier =
        std::get<std::optional<Order::OrderModifier>>(clause->t);
    if (modifier && *modifier == Order::OrderModifier::Unconstrained) {
      result.orderMod = aiir::omp::OrderModifierAttr::get(
          firOpBuilder.getContext(), aiir::omp::OrderModifier::unconstrained);
    } else {
      // "If order-modifier is not unconstrained, the behavior is as if the
      // reproducible modifier is present."
      result.orderMod = aiir::omp::OrderModifierAttr::get(
          firOpBuilder.getContext(), aiir::omp::OrderModifier::reproducible);
    }
    return true;
  }
  return false;
}

bool ClauseProcessor::processOrdered(
    aiir::omp::OrderedClauseOps &result) const {
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
    aiir::omp::PriorityClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Priority>()) {
    result.priority = fir::getBase(converter.genExprValue(clause->v, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processDetach(aiir::omp::DetachClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Detach>()) {
    semantics::Symbol *sym = clause->v.sym();
    aiir::Value symVal = converter.getSymbolAddress(*sym);
    result.eventHandle = symVal;
    return true;
  }
  return false;
}

bool ClauseProcessor::processProcBind(
    aiir::omp::ProcBindClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::ProcBind>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    result.procBindKind = genProcBindKindAttr(firOpBuilder, *clause);
    return true;
  }
  return false;
}

bool ClauseProcessor::processTileSizes(
    lower::pft::Evaluation &eval, aiir::omp::LoopNestOperands &result) const {
  auto *ompCons{eval.getIf<parser::OpenMPConstruct>()};
  collectTileSizesFromOpenMPConstruct(ompCons, result.tileSizes, semaCtx);
  return !result.tileSizes.empty();
}

bool ClauseProcessor::processSafelen(
    aiir::omp::SafelenClauseOps &result) const {
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
    aiir::omp::ScheduleClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Schedule>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    aiir::AIIRContext *context = firOpBuilder.getContext();
    const auto &scheduleType = std::get<omp::clause::Schedule::Kind>(clause->t);

    aiir::omp::ClauseScheduleKind scheduleKind;
    switch (scheduleType) {
    case omp::clause::Schedule::Kind::Static:
      scheduleKind = aiir::omp::ClauseScheduleKind::Static;
      break;
    case omp::clause::Schedule::Kind::Dynamic:
      scheduleKind = aiir::omp::ClauseScheduleKind::Dynamic;
      break;
    case omp::clause::Schedule::Kind::Guided:
      scheduleKind = aiir::omp::ClauseScheduleKind::Guided;
      break;
    case omp::clause::Schedule::Kind::Auto:
      scheduleKind = aiir::omp::ClauseScheduleKind::Auto;
      break;
    case omp::clause::Schedule::Kind::Runtime:
      scheduleKind = aiir::omp::ClauseScheduleKind::Runtime;
      break;
    }

    result.scheduleKind =
        aiir::omp::ClauseScheduleKindAttr::get(context, scheduleKind);

    aiir::omp::ScheduleModifier scheduleMod = getScheduleModifier(*clause);
    if (scheduleMod != aiir::omp::ScheduleModifier::none)
      result.scheduleMod =
          aiir::omp::ScheduleModifierAttr::get(context, scheduleMod);

    if (getSimdModifier(*clause) != aiir::omp::ScheduleModifier::none)
      result.scheduleSimd = firOpBuilder.getUnitAttr();

    if (const auto &chunkExpr = std::get<omp::MaybeExpr>(clause->t))
      result.scheduleChunk =
          fir::getBase(converter.genExprValue(*chunkExpr, stmtCtx));

    return true;
  }
  return false;
}

bool ClauseProcessor::processSimdlen(
    aiir::omp::SimdlenClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::Simdlen>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    const std::optional<std::int64_t> simdlenVal = evaluate::ToInt64(clause->v);
    result.simdlen = firOpBuilder.getI64IntegerAttr(*simdlenVal);
    return true;
  }
  return false;
}

bool ClauseProcessor::processSimd(
    aiir::omp::OrderedRegionOperands &result) const {
  return markClauseOccurrence<omp::clause::Simd>(result.parLevelSimd);
}

bool ClauseProcessor::processThreadLimit(
    lower::StatementContext &stmtCtx,
    aiir::omp::ThreadLimitClauseOps &result) const {
  if (auto *clause = findUniqueClause<omp::clause::ThreadLimit>()) {
    result.threadLimitVars.reserve(clause->v.size());
    for (const ExprTy &vv : clause->v)
      result.threadLimitVars.push_back(
          fir::getBase(converter.genExprValue(vv, stmtCtx)));
    return true;
  }
  return false;
}

bool ClauseProcessor::processUntied(aiir::omp::UntiedClauseOps &result) const {
  return markClauseOccurrence<omp::clause::Untied>(result.untied);
}

//===----------------------------------------------------------------------===//
// ClauseProcessor repeatable clauses
//===----------------------------------------------------------------------===//
static llvm::StringMap<bool> getTargetFeatures(aiir::ModuleOp module) {
  llvm::StringMap<bool> featuresMap;
  llvm::SmallVector<llvm::StringRef> targetFeaturesVec;
  if (aiir::LLVM::TargetFeaturesAttr features =
          fir::getTargetFeatures(module)) {
    llvm::ArrayRef<aiir::StringAttr> featureAttrs = features.getFeatures();
    for (auto &featureAttr : featureAttrs) {
      llvm::StringRef featureKeyString = featureAttr.strref();
      featuresMap[featureKeyString.substr(1)] = (featureKeyString[0] == '+');
    }
  }
  return featuresMap;
}

bool ClauseProcessor::processAffinity(
    aiir::omp::AffinityClauseOps &result) const {
  return findRepeatableClause<omp::clause::Affinity>(
      [&](const omp::clause::Affinity &clause, const parser::CharBlock &) {
        const auto &objects = std::get<omp::ObjectList>(clause.t);
        lower::StatementContext stmtCtx;
        auto &builder = converter.getFirOpBuilder();
        auto &context = converter.getAIIRContext();
        aiir::Location clauseLocation = converter.getCurrentLocation();

        aiir::Type refI8Ty = fir::ReferenceType::get(builder.getIntegerType(8));
        aiir::Type entryTy = aiir::omp::AffinityEntryType::get(
            &context, refI8Ty, builder.getI64Type());
        aiir::Type iterTy =
            aiir::omp::IteratedType::get(&converter.getAIIRContext(), entryTy);

        auto makeAffinityEntry = [&](fir::FirOpBuilder &b, aiir::Location l,
                                     aiir::Type entryTy, aiir::Value addr,
                                     aiir::Value len) -> aiir::Value {
          aiir::Value addrI8 = fir::ConvertOp::create(b, l, refI8Ty, addr);
          return aiir::omp::AffinityEntryOp::create(b, l, entryTy, addrI8, len)
              .getResult();
        };

        llvm::SmallVector<IteratorRange> iteratorRanges;
        llvm::SmallPtrSet<const Fortran::semantics::Symbol *, 4> ivSyms;

        auto &iteratorModifier =
            std::get<std::optional<omp::clause::Iterator>>(clause.t);
        collectIteratorIVs(clause, converter, stmtCtx, iteratorRanges, ivSyms);

        for (const omp::Object &object : objects) {
          llvm::SmallVector<aiir::Value> bounds;
          std::stringstream asFortran;
          if (iteratorModifier.has_value() &&
              hasIteratorIVReference(object, ivSyms)) {
            aiir::Value iterHandle = buildIteratorOp(
                converter, clauseLocation, iterTy, iteratorRanges,
                [&](fir::FirOpBuilder &builder, aiir::Location loc,
                    llvm::ArrayRef<aiir::Value> /*ivs*/) -> aiir::Value {
                  lower::StatementContext iterStmtCtx;

                  if (std::optional<llvm::SmallVector<aiir::Value>>
                          loweredIndices = getIteratorElementIndices(
                              converter, object, iterStmtCtx, loc)) {
                    const Fortran::semantics::Symbol *sym = object.sym();
                    assert(sym && "expected symbol for iterator object");
                    fir::factory::AddrAndBoundsInfo info =
                        Fortran::lower::getDataOperandBaseAddr(
                            converter, builder, *sym, loc,
                            /*unwrapFirBox=*/false);
                    hlfir::Entity entity{info.addr};
                    aiir::Value iteratedAddr = genIteratorCoordinate(
                        converter, entity, *loweredIndices, loc);
                    aiir::Value len = genElementSizeInBytes(
                        builder, loc, builder.getDataLayout(), entity);
                    return makeAffinityEntry(builder, loc, entryTy,
                                             iteratedAddr, len);
                  }

                  TODO(loc, "object type not supported by iterator modifier");
                });
            result.iterated.push_back(iterHandle);
          } else {
            aiir::Value addr =
                genAffinityAddr(converter, object, stmtCtx, clauseLocation);
            // get hlfir.declare for length calculation
            fir::factory::AddrAndBoundsInfo info =
                lower::gatherDataOperandAddrAndBounds<aiir::omp::MapBoundsOp,
                                                      aiir::omp::MapBoundsType>(
                    converter, builder, semaCtx, stmtCtx, *object.sym(),
                    object.ref(), clauseLocation, asFortran, bounds,
                    treatIndexAsSection);
            aiir::Value len =
                genAffinityLen(builder, clauseLocation, builder.getDataLayout(),
                               hlfir::Entity{info.addr}, bounds);
            result.affinityVars.push_back(
                makeAffinityEntry(builder, clauseLocation, entryTy, addr, len));
          }
        }

        return true;
      });
}

static void
addAlignedClause(lower::AbstractConverter &converter,
                 const omp::clause::Aligned &clause,
                 llvm::SmallVectorImpl<aiir::Value> &alignedVars,
                 llvm::SmallVectorImpl<aiir::Attribute> &alignments) {
  using Aligned = omp::clause::Aligned;
  lower::StatementContext stmtCtx;
  aiir::IntegerAttr alignmentValueAttr;
  int64_t alignment = 0;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  if (auto &alignmentValueParserExpr =
          std::get<std::optional<Aligned::Alignment>>(clause.t)) {
    aiir::Value operand = fir::getBase(
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
  // 0 or not a power of two
  if (alignment > 0 && ((alignment & (alignment - 1)) == 0)) {
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
    aiir::omp::AlignedClauseOps &result) const {
  return findRepeatableClause<omp::clause::Aligned>(
      [&](const omp::clause::Aligned &clause, const parser::CharBlock &) {
        addAlignedClause(converter, clause, result.alignedVars,
                         result.alignments);
      });
}

bool ClauseProcessor::processAllocate(
    aiir::omp::AllocateClauseOps &result) const {
  return findRepeatableClause<omp::clause::Allocate>(
      [&](const omp::clause::Allocate &clause, const parser::CharBlock &) {
        genAllocateClause(converter, clause, result.allocatorVars,
                          result.allocateVars);
      });
}

bool ClauseProcessor::processCopyin() const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  aiir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());
  auto checkAndCopyHostAssociateVar =
      [&](semantics::Symbol *sym,
          aiir::OpBuilder::InsertPoint *copyAssignIP = nullptr) {
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
    aiir::omp::BarrierOp::create(firOpBuilder, converter.getCurrentLocation());
  return hasCopyin;
}

/// Class that extracts information from the specified type.
class TypeInfo {
public:
  TypeInfo(aiir::Type ty) { typeScan(ty); }

  // Returns the length of character types.
  std::optional<fir::CharacterType::LenType> getCharLength() const {
    return charLen;
  }

  // Returns the shape of array types.
  llvm::ArrayRef<int64_t> getShape() const { return shape; }

  // Is the type inside a box?
  bool isBox() const { return inBox; }

  bool isBoxChar() const { return inBoxChar; }

private:
  void typeScan(aiir::Type type);

  std::optional<fir::CharacterType::LenType> charLen;
  llvm::SmallVector<int64_t> shape;
  bool inBox = false;
  bool inBoxChar = false;
};

void TypeInfo::typeScan(aiir::Type ty) {
  if (auto sty = aiir::dyn_cast<fir::SequenceType>(ty)) {
    assert(shape.empty() && !sty.getShape().empty());
    shape = llvm::SmallVector<int64_t>(sty.getShape());
    typeScan(sty.getEleTy());
  } else if (auto bty = aiir::dyn_cast<fir::BoxType>(ty)) {
    inBox = true;
    typeScan(bty.getEleTy());
  } else if (auto cty = aiir::dyn_cast<fir::ClassType>(ty)) {
    inBox = true;
    typeScan(cty.getEleTy());
  } else if (auto cty = aiir::dyn_cast<fir::CharacterType>(ty)) {
    charLen = cty.getLen();
  } else if (auto cty = aiir::dyn_cast<fir::BoxCharType>(ty)) {
    inBoxChar = true;
    typeScan(cty.getEleTy());
  } else if (auto hty = aiir::dyn_cast<fir::HeapType>(ty)) {
    typeScan(hty.getEleTy());
  } else if (auto pty = aiir::dyn_cast<fir::PointerType>(ty)) {
    typeScan(pty.getEleTy());
  } else {
    // The scan ends when reaching any built-in, record or boxproc type.
    assert(ty.isIntOrIndexOrFloat() || aiir::isa<aiir::ComplexType>(ty) ||
           aiir::isa<fir::LogicalType>(ty) || aiir::isa<fir::RecordType>(ty) ||
           aiir::isa<fir::BoxProcType>(ty));
  }
}

// Create a function that performs a copy between two variables, compatible
// with their types and attributes.
static aiir::func::FuncOp
createCopyFunc(aiir::Location loc, lower::AbstractConverter &converter,
               aiir::Type varType, fir::FortranVariableFlagsEnum varAttrs) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  aiir::ModuleOp module = builder.getModule();
  aiir::Type eleTy = fir::unwrapRefType(varType);
  TypeInfo typeInfo(eleTy);
  std::string copyFuncName =
      fir::getTypeAsString(varType, builder.getKindMap(), "_copy");

  if (auto decl = module.lookupSymbol<aiir::func::FuncOp>(copyFuncName))
    return decl;

  // create function
  aiir::OpBuilder::InsertionGuard guard(builder);
  aiir::OpBuilder modBuilder(module.getBodyRegion());
  llvm::SmallVector<aiir::Type> argsTy = {varType, varType};
  auto funcType = aiir::FunctionType::get(builder.getContext(), argsTy, {});
  aiir::func::FuncOp funcOp =
      aiir::func::FuncOp::create(modBuilder, loc, copyFuncName, funcType);
  funcOp.setVisibility(aiir::SymbolTable::Visibility::Private);
  fir::factory::setInternalLinkage(funcOp);
  builder.createBlock(&funcOp.getRegion(), funcOp.getRegion().end(), argsTy,
                      {loc, loc});
  builder.setInsertionPointToStart(&funcOp.getRegion().back());
  // generate body
  fir::FortranVariableFlagsAttr attrs;
  if (varAttrs != fir::FortranVariableFlagsEnum::None)
    attrs = fir::FortranVariableFlagsAttr::get(builder.getContext(), varAttrs);
  aiir::Value shape;
  if (!typeInfo.isBox() && !typeInfo.getShape().empty()) {
    llvm::SmallVector<aiir::Value> extents;
    for (auto extent : typeInfo.getShape())
      extents.push_back(
          builder.createIntegerConstant(loc, builder.getIndexType(), extent));
    shape = fir::ShapeOp::create(builder, loc, extents);
  }
  aiir::Value dst = funcOp.getArgument(0);
  aiir::Value src = funcOp.getArgument(1);
  llvm::SmallVector<aiir::Value> typeparams;
  if (typeInfo.isBoxChar()) {
    // fir.boxchar will be passed here as fir.ref<fir.boxchar>
    auto loadDst = fir::LoadOp::create(builder, loc, dst);
    auto loadSrc = fir::LoadOp::create(builder, loc, src);
    // get the actual fir.ref<fir.char> type
    aiir::Type refType =
        fir::ReferenceType::get(aiir::cast<fir::BoxCharType>(eleTy).getEleTy());
    auto unboxedDst = fir::UnboxCharOp::create(builder, loc, refType,
                                               builder.getIndexType(), loadDst);
    auto unboxedSrc = fir::UnboxCharOp::create(builder, loc, refType,
                                               builder.getIndexType(), loadSrc);
    // Add length to type parameters
    typeparams.push_back(unboxedDst.getResult(1));
    dst = unboxedDst.getResult(0);
    src = unboxedSrc.getResult(0);
  } else if (typeInfo.getCharLength().has_value()) {
    aiir::Value charLen = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), *typeInfo.getCharLength());
    typeparams.push_back(charLen);
  }
  auto declDst = hlfir::DeclareOp::create(
      builder, loc, dst, copyFuncName + "_dst", shape, typeparams,
      /*dummy_scope=*/nullptr, /*storage=*/nullptr,
      /*storage_offset=*/0, attrs);
  auto declSrc = hlfir::DeclareOp::create(
      builder, loc, src, copyFuncName + "_src", shape, typeparams,
      /*dummy_scope=*/nullptr, /*storage=*/nullptr,
      /*storage_offset=*/0, attrs);
  converter.copyVar(loc, declDst.getBase(), declSrc.getBase(), varAttrs);
  aiir::func::ReturnOp::create(builder, loc);
  return funcOp;
}

bool ClauseProcessor::processCopyprivate(
    aiir::Location currentLocation,
    aiir::omp::CopyprivateClauseOps &result) const {
  auto addCopyPrivateVar = [&](semantics::Symbol *sym) {
    aiir::Value symVal = converter.getSymbolAddress(*sym);
    auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>();
    if (!declOp)
      fir::emitFatalError(currentLocation,
                          "COPYPRIVATE is supported only in HLFIR mode");
    symVal = declOp.getBase();
    aiir::Type symType = symVal.getType();
    fir::FortranVariableFlagsEnum attrs =
        declOp.getFortranAttrs().has_value()
            ? *declOp.getFortranAttrs()
            : fir::FortranVariableFlagsEnum::None;
    aiir::Value cpVar = symVal;

    // CopyPrivate variables must be passed by reference. However, in the case
    // of assumed shapes/vla the type is not a !fir.ref, but a !fir.box.
    // In the case of character types, the passed in type can also be
    // !fir.boxchar. In these cases to retrieve the appropriate
    // !fir.ref<!fir.box<...>> or !fir.ref<!fir.boxchar<..>> to access the data
    // we need we must perform an alloca and then store to it and retrieve the
    // data from the new alloca.
    if (aiir::isa<fir::BaseBoxType>(symType) ||
        aiir::isa<fir::BoxCharType>(symType)) {
      fir::FirOpBuilder &builder = converter.getFirOpBuilder();
      auto alloca = fir::AllocaOp::create(builder, currentLocation, symType);
      fir::StoreOp::create(builder, currentLocation, symVal, alloca);
      cpVar = alloca;
    }

    result.copyprivateVars.push_back(cpVar);
    aiir::func::FuncOp funcOp =
        createCopyFunc(currentLocation, converter, cpVar.getType(), attrs);
    result.copyprivateSyms.push_back(aiir::SymbolRefAttr::get(funcOp));
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
                                    aiir::omp::DependClauseOps &result) const {
  auto process = [&](const omp::clause::Depend &clause,
                     const parser::CharBlock &) {
    auto depType = std::get<clause::DependenceType>(clause.t);
    auto &objects = std::get<omp::ObjectList>(clause.t);
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();

    if (std::get<std::optional<omp::clause::Iterator>>(clause.t)) {
      TODO(converter.getCurrentLocation(),
           "Support for iterator modifiers is not implemented yet");
    }
    aiir::omp::ClauseTaskDependAttr dependTypeOperand =
        genDependKindAttr(converter, depType);
    result.dependKinds.append(objects.size(), dependTypeOperand);

    for (const omp::Object &object : objects) {
      assert(object.ref() && "Expecting designator");
      aiir::Value dependVar;
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
      if (auto ref = aiir::dyn_cast<fir::ReferenceType>(dependVar.getType()))
        if (fir::isa_box_type(ref.getElementType()))
          dependVar = fir::LoadOp::create(
              builder, converter.getCurrentLocation(), dependVar);

      // The openmp dialect doesn't know what to do with boxes (and it would
      // break layering to teach it about them). The dependency variable can be
      // a box because it was an array section or because the original symbol
      // was mapped to a box.
      // Getting the address of the box data is okay because all the runtime
      // ultimately cares about is the base address of the array.
      if (fir::isa_box_type(dependVar.getType()))
        dependVar = fir::BoxAddrOp::create(
            builder, converter.getCurrentLocation(), dependVar);

      result.dependVars.push_back(dependVar);
    }
  };

  return findRepeatableClause<omp::clause::Depend>(process);
}

bool ClauseProcessor::processGrainsize(
    lower::StatementContext &stmtCtx,
    aiir::omp::GrainsizeClauseOps &result) const {
  using Grainsize = omp::clause::Grainsize;
  if (auto *clause = findUniqueClause<Grainsize>()) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    aiir::AIIRContext *context = firOpBuilder.getContext();
    const auto &modifier =
        std::get<std::optional<Grainsize::Prescriptiveness>>(clause->t);
    if (modifier && *modifier == Grainsize::Prescriptiveness::Strict) {
      result.grainsizeMod = aiir::omp::ClauseGrainsizeTypeAttr::get(
          context, aiir::omp::ClauseGrainsizeType::Strict);
    }
    const auto &grainsizeExpr = std::get<omp::SomeExpr>(clause->t);
    result.grainsize =
        fir::getBase(converter.genExprValue(grainsizeExpr, stmtCtx));
    return true;
  }
  return false;
}

bool ClauseProcessor::processHasDeviceAddr(
    lower::StatementContext &stmtCtx, aiir::omp::HasDeviceAddrClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &hasDeviceSyms) const {
  // For HAS_DEVICE_ADDR objects, implicitly map the top-level entities.
  // Their address (or the whole descriptor, if the entity had one) will be
  // passed to the target region.
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;
  bool clauseFound = findRepeatableClause<omp::clause::HasDeviceAddr>(
      [&](const omp::clause::HasDeviceAddr &clause,
          const parser::CharBlock &source) {
        aiir::Location location = converter.genLocation(source);
        aiir::omp::ClauseMapFlags mapTypeBits =
            aiir::omp::ClauseMapFlags::to | aiir::omp::ClauseMapFlags::implicit;
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
    aiir::omp::IfClauseOps &result) const {
  bool found = false;
  findRepeatableClause<omp::clause::If>([&](const omp::clause::If &clause,
                                            const parser::CharBlock &source) {
    aiir::Location clauseLocation = converter.genLocation(source);
    aiir::Value operand =
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

template <typename T>
void collectReductionSyms(
    const T &reduction,
    llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) {
  const auto &objectList{std::get<omp::ObjectList>(reduction.t)};
  for (const Object &object : objectList) {
    const semantics::Symbol *symbol = object.sym();
    reductionSyms.push_back(symbol);
  }
}

bool ClauseProcessor::processInReduction(
    aiir::Location currentLocation, aiir::omp::InReductionClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &outReductionSyms) const {
  return findRepeatableClause<omp::clause::InReduction>(
      [&](const omp::clause::InReduction &clause, const parser::CharBlock &) {
        llvm::SmallVector<aiir::Value> inReductionVars;
        llvm::SmallVector<bool> inReduceVarByRef;
        llvm::SmallVector<aiir::Attribute> inReductionDeclSymbols;
        llvm::SmallVector<const semantics::Symbol *> inReductionSyms;
        collectReductionSyms(clause, inReductionSyms);

        ReductionProcessor rp;
        if (!rp.processReductionArguments<aiir::omp::DeclareReductionOp>(
                currentLocation, converter,
                std::get<typename omp::clause::ReductionOperatorList>(clause.t),
                inReductionVars, inReduceVarByRef, inReductionDeclSymbols,
                inReductionSyms))
          TODO(currentLocation, "Lowering unrecognised reduction type");

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
    lower::StatementContext &stmtCtx, aiir::omp::IsDevicePtrClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &isDeviceSyms) const {
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;
  bool clauseFound = findRepeatableClause<omp::clause::IsDevicePtr>(
      [&](const omp::clause::IsDevicePtr &clause,
          const parser::CharBlock &source) {
        aiir::Location location = converter.genLocation(source);
        // Force a map so the descriptor is materialized on the device with the
        // device address inside.
        aiir::omp::ClauseMapFlags mapTypeBits =
            aiir::omp::ClauseMapFlags::is_device_ptr |
            aiir::omp::ClauseMapFlags::to;
        processMapObjects(stmtCtx, location, clause.v, mapTypeBits,
                          parentMemberIndices, result.isDevicePtrVars,
                          isDeviceSyms);
      });

  insertChildMapInfoIntoParent(converter, semaCtx, stmtCtx, parentMemberIndices,
                               result.isDevicePtrVars, isDeviceSyms);
  return clauseFound;
}

bool ClauseProcessor::processLinear(aiir::omp::LinearClauseOps &result,
                                    bool isDeclareSimd) const {
  lower::StatementContext stmtCtx;
  std::vector<aiir::Attribute> typeAttrs;
  std::vector<aiir::Attribute> linearModAttrs;
  return findRepeatableClause<
      omp::clause::Linear>([&](const omp::clause::Linear &clause,
                               const parser::CharBlock &) {
    auto &objects = std::get<omp::ObjectList>(clause.t);

    std::optional<aiir::omp::LinearModifier> explicitLinearMod;
    if (auto &linearModifier =
            std::get<std::optional<omp::clause::Linear::LinearModifier>>(
                clause.t)) {
      switch (*linearModifier) {
      case omp::clause::Linear::LinearModifier::Val:
        explicitLinearMod = aiir::omp::LinearModifier::val;
        break;
      case omp::clause::Linear::LinearModifier::Ref:
        explicitLinearMod = aiir::omp::LinearModifier::ref;
        break;
      case omp::clause::Linear::LinearModifier::Uval:
        explicitLinearMod = aiir::omp::LinearModifier::uval;
        break;
      }
    }

    for (const omp::Object &object : objects) {
      semantics::Symbol *sym = object.sym();
      const aiir::Value variable = converter.getSymbolAddress(*sym);
      result.linearVars.push_back(variable);
      aiir::Type ty = converter.genType(*sym);
      typeAttrs.push_back(aiir::TypeAttr::get(ty));

      if (auto &mod =
              std::get<std::optional<omp::clause::Linear::StepComplexModifier>>(
                  clause.t)) {
        aiir::Value operand =
            fir::getBase(converter.genExprValue(toEvExpr(*mod), stmtCtx));
        result.linearStepVars.append(objects.size(), operand);
      } else {
        // If nothing is present, add the default step of 1.
        fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
        aiir::Location currentLocation = converter.getCurrentLocation();
        aiir::Type integerTy = ty.isInteger() ? ty : firOpBuilder.getI32Type();
        aiir::Value operand =
            firOpBuilder.createIntegerConstant(currentLocation, integerTy, 1);
        result.linearStepVars.append(objects.size(), operand);
      }

      // Determine the linear modifier:
      // 1. Use explicit modifier if provided.
      // 2. For OpenMP >= 5.2 (Section 5.4.6: "the default linear-modifier
      //    is val"):
      //    - declare simd: "ref" for POINTER or non-VALUE dummy args,
      //      "val" otherwise.
      //    - do/simd: always "val".
      // 3. Otherwise, leave unset (UnitAttr placeholder).
      auto getDeclareSimdDefaultMod = [](const semantics::Symbol &sym) {
        const auto &ultimate = sym.GetUltimate();
        if (semantics::IsPointer(ultimate))
          return aiir::omp::LinearModifier::ref;
        if (const auto *obj =
                ultimate.detailsIf<semantics::ObjectEntityDetails>())
          if (obj->isDummy() && !semantics::IsValue(ultimate))
            return aiir::omp::LinearModifier::ref;
        return aiir::omp::LinearModifier::val;
      };

      std::optional<aiir::omp::LinearModifier> linearMod;
      if (explicitLinearMod)
        linearMod = *explicitLinearMod;
      else if (semaCtx.langOptions().OpenMPVersion >= 52)
        linearMod = isDeclareSimd ? getDeclareSimdDefaultMod(*sym)
                                  : aiir::omp::LinearModifier::val;

      if (linearMod)
        linearModAttrs.push_back(aiir::omp::LinearModifierAttr::get(
            &converter.getAIIRContext(), *linearMod));
      else
        linearModAttrs.push_back(
            aiir::UnitAttr::get(&converter.getAIIRContext()));
    }
    result.linearVarTypes =
        aiir::ArrayAttr::get(&converter.getAIIRContext(), typeAttrs);
    result.linearModifiers =
        aiir::ArrayAttr::get(&converter.getAIIRContext(), linearModAttrs);
  });
}

bool ClauseProcessor::processLink(
    llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &result) const {
  return findRepeatableClause<omp::clause::Link>(
      [&](const omp::clause::Link &clause, const parser::CharBlock &) {
        // Case: declare target link(var1, var2)...
        gatherFuncAndVarSyms(
            clause.v, aiir::omp::DeclareTargetCaptureClause::link, result,
            /*automap=*/false);
      });
}

void ClauseProcessor::processMapObjects(
    lower::StatementContext &stmtCtx, aiir::Location clauseLocation,
    const omp::ObjectList &objects, aiir::omp::ClauseMapFlags mapTypeBits,
    std::map<Object, OmpMapParentAndMemberData> &parentMemberIndices,
    llvm::SmallVectorImpl<aiir::Value> &mapVars,
    llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms,
    llvm::StringRef mapperIdNameRef, bool isMotionModifier) const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  auto getSymbolDerivedType = [](const semantics::Symbol &symbol)
      -> const semantics::DerivedTypeSpec * {
    const semantics::Symbol &ultimate = symbol.GetUltimate();
    if (const semantics::DeclTypeSpec *declType = ultimate.GetType())
      if (const auto *derived = declType->AsDerived())
        return derived;
    return nullptr;
  };

  auto addImplicitMapper = [&](const omp::Object &object,
                               std::string &mapperIdName,
                               bool allowGenerate) -> aiir::FlatSymbolRefAttr {
    if (mapperIdName.empty())
      return aiir::FlatSymbolRefAttr();

    if (converter.getModuleOp().lookupSymbol(mapperIdName))
      return aiir::FlatSymbolRefAttr::get(&converter.getAIIRContext(),
                                          mapperIdName);

    if (!allowGenerate)
      return aiir::FlatSymbolRefAttr();

    const semantics::DerivedTypeSpec *typeSpec =
        getSymbolDerivedType(*object.sym());
    if (!typeSpec && object.sym()->owner().IsDerivedType())
      typeSpec = object.sym()->owner().derivedTypeSpec();

    if (!typeSpec)
      return aiir::FlatSymbolRefAttr();

    aiir::Type type = converter.genType(*typeSpec);
    auto recordType = aiir::dyn_cast<fir::RecordType>(type);
    if (!recordType)
      return aiir::FlatSymbolRefAttr();

    return utils::openmp::getOrGenImplicitDefaultDeclareMapper(
        converter.getFirOpBuilder(), clauseLocation, recordType, mapperIdName,
        [&](std::string &mapperIdName, llvm::StringRef memberName) {
          defaultMangler(converter, mapperIdName, memberName);
        });
  };

  auto getDefaultMapperID =
      [&](const semantics::DerivedTypeSpec *typeSpec) -> std::string {
    if (aiir::isa<aiir::omp::DeclareMapperOp>(
            firOpBuilder.getRegion().getParentOp()) ||
        !typeSpec)
      return {};

    std::string mapperIdName =
        typeSpec->name().ToString() + llvm::omp::OmpDefaultMapperName;
    if (auto *sym = converter.getCurrentScope().FindSymbol(mapperIdName)) {
      mapperIdName =
          converter.mangleName(mapperIdName, sym->GetUltimate().owner());
    } else {
      mapperIdName = converter.mangleName(mapperIdName, *typeSpec->GetScope());
    }

    // Make sure we don't return a mapper to self.
    if (auto declMapOp = aiir::dyn_cast<aiir::omp::DeclareMapperOp>(
            firOpBuilder.getRegion().getParentOp()))
      if (mapperIdName == declMapOp.getSymName())
        return {};
    return mapperIdName;
  };

  // Create the mapper symbol from its name, if specified.
  aiir::FlatSymbolRefAttr mapperId;
  if (!mapperIdNameRef.empty() && !objects.empty() &&
      mapperIdNameRef != "__implicit_mapper") {
    std::string mapperIdName = mapperIdNameRef.str();
    const omp::Object &object = objects.front();
    if (mapperIdNameRef == "default") {
      const semantics::DerivedTypeSpec *typeSpec =
          getSymbolDerivedType(*object.sym());
      if (!typeSpec && object.sym()->owner().IsDerivedType())
        typeSpec = object.sym()->owner().derivedTypeSpec();
      mapperIdName = getDefaultMapperID(typeSpec);
    }
    assert(converter.getModuleOp().lookupSymbol(mapperIdName) &&
           "mapper not found");
    mapperId =
        aiir::FlatSymbolRefAttr::get(&converter.getAIIRContext(), mapperIdName);
  }

  for (const omp::Object &object : objects) {
    llvm::SmallVector<aiir::Value> bounds;
    std::stringstream asFortran;
    std::optional<omp::Object> parentObj;

    fir::factory::AddrAndBoundsInfo info =
        lower::gatherDataOperandAddrAndBounds<aiir::omp::MapBoundsOp,
                                              aiir::omp::MapBoundsType>(
            converter, firOpBuilder, semaCtx, stmtCtx, *object.sym(),
            object.ref(), clauseLocation, asFortran, bounds,
            treatIndexAsSection);

    aiir::Value baseOp = info.rawInput;
    if (object.sym()->owner().IsDerivedType() && !isMotionModifier) {
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

    const semantics::DerivedTypeSpec *objectTypeSpec =
        getSymbolDerivedType(*object.sym());

    if (mapperIdNameRef == "__implicit_mapper") {
      if (parentObj.has_value()) {
        mapperId = aiir::FlatSymbolRefAttr();
      } else if (objectTypeSpec) {
        std::string mapperIdName = getDefaultMapperID(objectTypeSpec);
        bool isAllocOrPointer =
            semantics::IsAllocatableOrObjectPointer(object.sym());
        bool isPointer = semantics::IsPointer(*object.sym());
        bool isImplicitMap =
            (mapTypeBits & aiir::omp::ClauseMapFlags::implicit) ==
            aiir::omp::ClauseMapFlags::implicit;
        bool needsDefaultMapper =
            isAllocOrPointer ||
            requiresImplicitDefaultDeclareMapper(*objectTypeSpec);
        // For implicit captures, avoid synthesizing default mappers for pointer
        // entities (which can over-map pointer payloads) and for plain
        // non-allocatable/non-pointer entities. Keep implicit mapper support
        // for allocatables.
        if (isImplicitMap && (isPointer || !isAllocOrPointer))
          needsDefaultMapper = false;
        if (!mapperIdName.empty())
          mapperId = addImplicitMapper(object, mapperIdName,
                                       /*allowGenerate=*/needsDefaultMapper);
        else
          mapperId = aiir::FlatSymbolRefAttr();
      } else {
        mapperId = aiir::FlatSymbolRefAttr();
      }
    }

    // Explicit map captures are captured ByRef by default,
    // optimisation passes may alter this to ByCopy or other capture
    // types to optimise
    auto location = aiir::NameLoc::get(
        aiir::StringAttr::get(firOpBuilder.getContext(), asFortran.str()),
        baseOp.getLoc());
    aiir::omp::MapInfoOp mapOp = utils::openmp::createMapInfoOp(
        firOpBuilder, location, baseOp,
        /*varPtrPtr=*/aiir::Value{}, asFortran.str(), bounds,
        /*members=*/{}, /*membersIndex=*/aiir::ArrayAttr{}, mapTypeBits,
        aiir::omp::VariableCaptureKind::ByRef, baseOp.getType(),
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

/// Extract and mangle the mapper identifier name from a mapper clause.
/// Returns "__implicit_mapper" if no mapper is specified, or "default" if
/// the default mapper is specified, otherwise returns the mangled mapper name.
/// This handles both the Map clause (which uses a vector of mappers) and
/// To/From clauses (which use a DefinedOperator).
template <typename MapperType>
static std::string
getMapperIdentifier(lower::AbstractConverter &converter,
                    const std::optional<MapperType> &mapper) {
  if (!mapper)
    return "__implicit_mapper";

  // Handle mapper types (both have the same structure)
  assert(mapper->size() == 1 && "more than one mapper");
  const semantics::Symbol *mapperSym = mapper->front().v.id().symbol;

  std::string mapperIdName = mapperSym->name().ToString();
  if (mapperIdName != "default") {
    // Mangle with the ultimate owner so that use-associated mapper
    // identifiers resolve to the same symbol as their defining scope.
    const semantics::Symbol &ultimate = mapperSym->GetUltimate();
    mapperIdName = converter.mangleName(mapperIdName, ultimate.owner());
  }
  return mapperIdName;
}

bool ClauseProcessor::processMap(
    aiir::Location currentLocation, lower::StatementContext &stmtCtx,
    aiir::omp::MapClauseOps &result, llvm::omp::Directive directive,
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
    aiir::Location clauseLocation = converter.genLocation(source);
    const auto &[mapType, typeMods, attachMod, refMod, mappers, iterator,
                 objects] = clause.t;
    if (attachMod)
      TODO(currentLocation, "ATTACH modifier is not implemented yet");
    aiir::omp::ClauseMapFlags mapTypeBits = aiir::omp::ClauseMapFlags::none;
    // For data-motion directives we avoid auto-attaching implicit default
    // mappers. Deep recursive mapping there can conflict with explicit
    // component enter/exit maps users commonly spell out.
    std::string mapperIdName = getMapperIdentifier(converter, mappers);
    if ((directive == llvm::omp::Directive::OMPD_target_enter_data ||
         directive == llvm::omp::Directive::OMPD_target_exit_data ||
         directive == llvm::omp::Directive::OMPD_target_update) &&
        mapperIdName == "__implicit_mapper")
      mapperIdName.clear();
    // If the map type is specified, then process it else set the appropriate
    // default value
    Map::MapType type;
    if (directive == llvm::omp::Directive::OMPD_target_enter_data &&
        semaCtx.langOptions().OpenMPVersion >= 52)
      type = mapType.value_or(Map::MapType::To);
    else if (directive == llvm::omp::Directive::OMPD_target_exit_data &&
             semaCtx.langOptions().OpenMPVersion >= 52)
      type = mapType.value_or(Map::MapType::From);
    else
      type = mapType.value_or(Map::MapType::Tofrom);

    switch (type) {
    case Map::MapType::To:
      mapTypeBits |= aiir::omp::ClauseMapFlags::to;
      break;
    case Map::MapType::From:
      mapTypeBits |= aiir::omp::ClauseMapFlags::from;
      break;
    case Map::MapType::Tofrom:
      mapTypeBits |=
          aiir::omp::ClauseMapFlags::to | aiir::omp::ClauseMapFlags::from;
      break;
    case Map::MapType::Storage:
      mapTypeBits |= aiir::omp::ClauseMapFlags::storage;
      break;
    }

    if (typeMods) {
      // TODO: Still requires "self" modifier, an OpenMP 6.0+ feature
      if (llvm::is_contained(*typeMods, Map::MapTypeModifier::Always))
        mapTypeBits |= aiir::omp::ClauseMapFlags::always;
      if (llvm::is_contained(*typeMods, Map::MapTypeModifier::Present))
        mapTypeBits |= aiir::omp::ClauseMapFlags::present;
      if (llvm::is_contained(*typeMods, Map::MapTypeModifier::Close))
        mapTypeBits |= aiir::omp::ClauseMapFlags::close;
      if (llvm::is_contained(*typeMods, Map::MapTypeModifier::Delete))
        mapTypeBits |= aiir::omp::ClauseMapFlags::del;
      if (llvm::is_contained(*typeMods, Map::MapTypeModifier::OmpxHold))
        mapTypeBits |= aiir::omp::ClauseMapFlags::ompx_hold;
    }

    if (iterator) {
      TODO(currentLocation,
           "Support for iterator modifiers is not implemented yet");
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
                                           aiir::omp::MapClauseOps &result) {
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;
  llvm::SmallVector<const semantics::Symbol *> mapSymbols;

  auto callbackFn = [&](const auto &clause, const parser::CharBlock &source) {
    aiir::Location clauseLocation = converter.genLocation(source);
    const auto &[expectation, mapper, iterator, objects] = clause.t;

    aiir::omp::ClauseMapFlags mapTypeBits =
        std::is_same_v<llvm::remove_cvref_t<decltype(clause)>, omp::clause::To>
            ? aiir::omp::ClauseMapFlags::to
            : aiir::omp::ClauseMapFlags::from;
    if (expectation && *expectation == omp::clause::To::Expectation::Present)
      mapTypeBits |= aiir::omp::ClauseMapFlags::present;

    // Support motion modifiers: mapper, iterator.
    std::string mapperIdName = getMapperIdentifier(converter, mapper);
    if (mapperIdName == "__implicit_mapper")
      mapperIdName.clear();
    if (iterator) {
      TODO(clauseLocation, "Iterator modifier is not supported yet");
    }

    processMapObjects(stmtCtx, clauseLocation, objects, mapTypeBits,
                      parentMemberIndices, result.mapVars, mapSymbols,
                      mapperIdName, /*isMotionModifier=*/true);
  };

  bool clauseFound = findRepeatableClause<omp::clause::To>(callbackFn);
  clauseFound =
      findRepeatableClause<omp::clause::From>(callbackFn) || clauseFound;

  insertChildMapInfoIntoParent(converter, semaCtx, stmtCtx, parentMemberIndices,
                               result.mapVars, mapSymbols);

  return clauseFound;
}

bool ClauseProcessor::processNontemporal(
    aiir::omp::NontemporalClauseOps &result) const {
  return findRepeatableClause<omp::clause::Nontemporal>(
      [&](const omp::clause::Nontemporal &clause, const parser::CharBlock &) {
        for (const Object &object : clause.v) {
          semantics::Symbol *sym = object.sym();
          aiir::Value symVal = converter.getSymbolAddress(*sym);
          result.nontemporalVars.push_back(symVal);
        }
      });
}

bool ClauseProcessor::processReduction(
    aiir::Location currentLocation, aiir::omp::ReductionClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &outReductionSyms) const {
  return findRepeatableClause<omp::clause::Reduction>(
      [&](const omp::clause::Reduction &clause, const parser::CharBlock &) {
        llvm::SmallVector<aiir::Value> reductionVars;
        llvm::SmallVector<bool> reduceVarByRef;
        llvm::SmallVector<aiir::Attribute> reductionDeclSymbols;
        llvm::SmallVector<const semantics::Symbol *> reductionSyms;
        collectReductionSyms(clause, reductionSyms);

        auto mod = std::get<std::optional<ReductionModifier>>(clause.t);
        if (mod.has_value()) {
          if (mod.value() == ReductionModifier::Task)
            TODO(currentLocation, "Reduction modifier `task` is not supported");
          else
            result.reductionMod = aiir::omp::ReductionModifierAttr::get(
                converter.getFirOpBuilder().getContext(),
                translateReductionModifier(mod.value()));
        }

        ReductionProcessor rp;
        if (!rp.processReductionArguments<aiir::omp::DeclareReductionOp>(
                currentLocation, converter,
                std::get<typename omp::clause::ReductionOperatorList>(clause.t),
                reductionVars, reduceVarByRef, reductionDeclSymbols,
                reductionSyms))
          TODO(currentLocation, "Lowering unrecognised reduction type");
        // Copy local lists into the output.
        llvm::copy(reductionVars, std::back_inserter(result.reductionVars));
        llvm::copy(reduceVarByRef, std::back_inserter(result.reductionByref));
        llvm::copy(reductionDeclSymbols,
                   std::back_inserter(result.reductionSyms));
        llvm::copy(reductionSyms, std::back_inserter(outReductionSyms));
      });
}

bool ClauseProcessor::processTaskReduction(
    aiir::Location currentLocation, aiir::omp::TaskReductionClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &outReductionSyms) const {
  return findRepeatableClause<omp::clause::TaskReduction>(
      [&](const omp::clause::TaskReduction &clause, const parser::CharBlock &) {
        llvm::SmallVector<aiir::Value> taskReductionVars;
        llvm::SmallVector<bool> taskReduceVarByRef;
        llvm::SmallVector<aiir::Attribute> taskReductionDeclSymbols;
        llvm::SmallVector<const semantics::Symbol *> taskReductionSyms;
        collectReductionSyms(clause, taskReductionSyms);

        ReductionProcessor rp;
        if (!rp.processReductionArguments<aiir::omp::DeclareReductionOp>(
                currentLocation, converter,
                std::get<typename omp::clause::ReductionOperatorList>(clause.t),
                taskReductionVars, taskReduceVarByRef, taskReductionDeclSymbols,
                taskReductionSyms))
          TODO(currentLocation, "Lowering unrecognised reduction type");
        // Copy local lists into the output.
        llvm::copy(taskReductionVars,
                   std::back_inserter(result.taskReductionVars));
        llvm::copy(taskReduceVarByRef,
                   std::back_inserter(result.taskReductionByref));
        llvm::copy(taskReductionDeclSymbols,
                   std::back_inserter(result.taskReductionSyms));
        llvm::copy(taskReductionSyms, std::back_inserter(outReductionSyms));
      });
}

bool ClauseProcessor::processTo(
    llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &result) const {
  return findRepeatableClause<omp::clause::To>(
      [&](const omp::clause::To &clause, const parser::CharBlock &) {
        // Case: declare target to(func, var1, var2)...
        gatherFuncAndVarSyms(std::get<ObjectList>(clause.t),
                             aiir::omp::DeclareTargetCaptureClause::to, result,
                             /*automap=*/false);
      });
}

bool ClauseProcessor::processEnter(
    llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &result) const {
  return findRepeatableClause<omp::clause::Enter>(
      [&](const omp::clause::Enter &clause, const parser::CharBlock &source) {
        bool automap =
            std::get<std::optional<omp::clause::Enter::Modifier>>(clause.t)
                .has_value();
        // Case: declare target enter(func, var1, var2)...
        gatherFuncAndVarSyms(std::get<ObjectList>(clause.t),
                             aiir::omp::DeclareTargetCaptureClause::enter,
                             result, automap);
      });
}

bool ClauseProcessor::processUseDeviceAddr(
    lower::StatementContext &stmtCtx, aiir::omp::UseDeviceAddrClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const {
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;
  bool clauseFound = findRepeatableClause<omp::clause::UseDeviceAddr>(
      [&](const omp::clause::UseDeviceAddr &clause,
          const parser::CharBlock &source) {
        aiir::Location location = converter.genLocation(source);
        aiir::omp::ClauseMapFlags mapTypeBits =
            aiir::omp::ClauseMapFlags::return_param;
        processMapObjects(stmtCtx, location, clause.v, mapTypeBits,
                          parentMemberIndices, result.useDeviceAddrVars,
                          useDeviceSyms);
      });

  insertChildMapInfoIntoParent(converter, semaCtx, stmtCtx, parentMemberIndices,
                               result.useDeviceAddrVars, useDeviceSyms);
  return clauseFound;
}

bool ClauseProcessor::processUseDevicePtr(
    lower::StatementContext &stmtCtx, aiir::omp::UseDevicePtrClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) const {
  std::map<Object, OmpMapParentAndMemberData> parentMemberIndices;

  bool clauseFound = findRepeatableClause<omp::clause::UseDevicePtr>(
      [&](const omp::clause::UseDevicePtr &clause,
          const parser::CharBlock &source) {
        aiir::Location location = converter.genLocation(source);
        aiir::omp::ClauseMapFlags mapTypeBits =
            aiir::omp::ClauseMapFlags::return_param;
        processMapObjects(stmtCtx, location, clause.v, mapTypeBits,
                          parentMemberIndices, result.useDevicePtrVars,
                          useDeviceSyms);
      });

  insertChildMapInfoIntoParent(converter, semaCtx, stmtCtx, parentMemberIndices,
                               result.useDevicePtrVars, useDeviceSyms);
  return clauseFound;
}

bool ClauseProcessor::processUniform(
    aiir::omp::UniformClauseOps &result) const {
  return findRepeatableClause<omp::clause::Uniform>(
      [&](const omp::clause::Uniform &clause, const parser::CharBlock &) {
        const auto &objects = clause.v;
        if (!objects.empty())
          genObjectList(objects, converter, result.uniformVars);
      });
}

bool ClauseProcessor::processInbranch(
    aiir::omp::InbranchClauseOps &result) const {
  if (findUniqueClause<omp::clause::Inbranch>()) {
    result.inbranch = converter.getFirOpBuilder().getUnitAttr();
    return true;
  }
  return false;
}

bool ClauseProcessor::processNotinbranch(
    aiir::omp::NotinbranchClauseOps &result) const {
  if (findUniqueClause<omp::clause::Notinbranch>()) {
    result.notinbranch = converter.getFirOpBuilder().getUnitAttr();
    return true;
  }
  return false;
}

} // namespace omp
} // namespace lower
} // namespace Fortran
