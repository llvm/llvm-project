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
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace mlir;

static const Fortran::parser::Name *
getDesignatorNameIfDataRef(const Fortran::parser::Designator &designator) {
  const auto *dataRef = std::get_if<Fortran::parser::DataRef>(&designator.u);
  return dataRef ? std::get_if<Fortran::parser::Name>(&dataRef->u) : nullptr;
}

template <typename T>
static void createPrivateVarSyms(Fortran::lower::AbstractConverter &converter,
                                 const T *clause) {
  Fortran::semantics::Symbol *sym = nullptr;
  const Fortran::parser::OmpObjectList &ompObjectList = clause->v;
  for (const Fortran::parser::OmpObject &ompObject : ompObjectList.v) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const Fortran::parser::Name *name =
                      getDesignatorNameIfDataRef(designator)) {
                sym = name->symbol;
              }
            },
            [&](const Fortran::parser::Name &name) { sym = name.symbol; }},
        ompObject.u);

    // Privatization for symbols which are pre-determined (like loop index
    // variables) happen separately, for everything else privatize here
    if constexpr (std::is_same_v<T, Fortran::parser::OmpClause::Firstprivate>) {
      converter.copyHostAssociateVar(*sym);
    } else {
      bool success = converter.createHostAssociateVarClone(*sym);
      (void)success;
      assert(success && "Privatization failed due to existing binding");
    }
  }
}

static void privatizeVars(Fortran::lower::AbstractConverter &converter,
                          const Fortran::parser::OmpClauseList &opClauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (const auto &privateClause =
            std::get_if<Fortran::parser::OmpClause::Private>(&clause.u)) {
      createPrivateVarSyms(converter, privateClause);
    } else if (const auto &firstPrivateClause =
                   std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                       &clause.u)) {
      createPrivateVarSyms(converter, firstPrivateClause);
    }
  }
  firOpBuilder.restoreInsertionPoint(insPt);
}

static void genObjectList(const Fortran::parser::OmpObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          llvm::SmallVectorImpl<Value> &operands) {
  auto addOperands = [&](Fortran::lower::SymbolRef sym) {
    const mlir::Value variable = converter.getSymbolAddress(sym);
    if (variable) {
      operands.push_back(variable);
    } else {
      if (const auto *details =
              sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
        operands.push_back(converter.getSymbolAddress(details->symbol()));
        converter.copySymbolBinding(details->symbol(), sym);
      }
    }
  };
  for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::parser::Designator &designator) {
                     if (const Fortran::parser::Name *name =
                             getDesignatorNameIfDataRef(designator)) {
                       addOperands(*name->symbol);
                     }
                   },
                   [&](const Fortran::parser::Name &name) {
                     addOperands(*name.symbol);
                   }},
               ompObject.u);
  }
}

template <typename Op>
static void
createBodyOfOp(Op &op, Fortran::lower::AbstractConverter &converter,
               mlir::Location &loc,
               const Fortran::parser::OmpClauseList *clauses = nullptr,
               const Fortran::semantics::Symbol *arg = nullptr,
               bool outerCombined = false) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  // If an argument for the region is provided then create the block with that
  // argument. Also update the symbol's address with the mlir argument value.
  // e.g. For loops the argument is the induction variable. And all further
  // uses of the induction variable should use this mlir value.
  if (arg) {
    firOpBuilder.createBlock(&op.getRegion(), {}, {converter.genType(*arg)},
                             {loc});
    converter.bindSymbol(*arg, op.getRegion().front().getArgument(0));
  } else {
    firOpBuilder.createBlock(&op.getRegion());
  }
  auto &block = op.getRegion().back();
  firOpBuilder.setInsertionPointToStart(&block);

  // Insert the terminator.
  if constexpr (std::is_same_v<Op, omp::WsLoopOp>) {
    mlir::ValueRange results;
    firOpBuilder.create<mlir::omp::YieldOp>(loc, results);
  } else {
    firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
  }

  // Reset the insertion point to the start of the first block.
  firOpBuilder.setInsertionPointToStart(&block);
  // Handle privatization. Do not privatize if this is the outer operation.
  if (clauses && !outerCombined)
    privatizeVars(converter, *clauses);
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPSimpleStandaloneConstruct
                       &simpleStandaloneConstruct) {
  const auto &directive =
      std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
          simpleStandaloneConstruct.t);
  switch (directive.v) {
  default:
    break;
  case llvm::omp::Directive::OMPD_barrier:
    converter.getFirOpBuilder().create<mlir::omp::BarrierOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    converter.getFirOpBuilder().create<mlir::omp::TaskwaitOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    converter.getFirOpBuilder().create<mlir::omp::TaskyieldOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    TODO(converter.getCurrentLocation(), "OMPD_target_enter_data");
  case llvm::omp::Directive::OMPD_target_exit_data:
    TODO(converter.getCurrentLocation(), "OMPD_target_exit_data");
  case llvm::omp::Directive::OMPD_target_update:
    TODO(converter.getCurrentLocation(), "OMPD_target_update");
  case llvm::omp::Directive::OMPD_ordered:
    TODO(converter.getCurrentLocation(), "OMPD_ordered");
  }
}

static void
genAllocateClause(Fortran::lower::AbstractConverter &converter,
                  const Fortran::parser::OmpAllocateClause &ompAllocateClause,
                  SmallVector<Value> &allocatorOperands,
                  SmallVector<Value> &allocateOperands) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  Fortran::lower::StatementContext stmtCtx;

  mlir::Value allocatorOperand;
  const Fortran::parser::OmpObjectList &ompObjectList =
      std::get<Fortran::parser::OmpObjectList>(ompAllocateClause.t);
  const auto &allocatorValue =
      std::get<std::optional<Fortran::parser::OmpAllocateClause::Allocator>>(
          ompAllocateClause.t);
  // Check if allocate clause has allocator specified. If so, add it
  // to list of allocators, otherwise, add default allocator to
  // list of allocators.
  if (allocatorValue) {
    allocatorOperand = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(allocatorValue->v), stmtCtx));
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

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPSimpleStandaloneConstruct
                  &simpleStandaloneConstruct) {
            genOMP(converter, eval, simpleStandaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
            SmallVector<Value, 4> operandRange;
            if (const auto &ompObjectList =
                    std::get<std::optional<Fortran::parser::OmpObjectList>>(
                        flushConstruct.t))
              genObjectList(*ompObjectList, converter, operandRange);
            const auto &memOrderClause = std::get<std::optional<
                std::list<Fortran::parser::OmpMemoryOrderClause>>>(
                flushConstruct.t);
            if (memOrderClause.has_value() && memOrderClause->size() > 0)
              TODO(converter.getCurrentLocation(),
                   "Handle OmpMemoryOrderClause");
            converter.getFirOpBuilder().create<mlir::omp::FlushOp>(
                converter.getCurrentLocation(), operandRange);
          },
          [&](const Fortran::parser::OpenMPCancelConstruct &cancelConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
          [&](const Fortran::parser::OpenMPCancellationPointConstruct
                  &cancellationPointConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
      },
      standaloneConstruct.u);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &blockDirective =
      std::get<Fortran::parser::OmpBlockDirective>(beginBlockDirective.t);
  const auto &endBlockDirective =
      std::get<Fortran::parser::OmpEndBlockDirective>(blockConstruct.t);
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  Fortran::lower::StatementContext stmtCtx;
  llvm::ArrayRef<mlir::Type> argTy;
  mlir::Value ifClauseOperand, numThreadsClauseOperand, finalClauseOperand,
      priorityClauseOperand;
  mlir::omp::ClauseProcBindKindAttr procBindKindAttr;
  SmallVector<Value> allocateOperands, allocatorOperands;
  mlir::UnitAttr nowaitAttr, untiedAttr, mergeableAttr;

  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginBlockDirective.t);
  for (const auto &clause : opClauseList.v) {
    if (const auto &ifClause =
            std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->v.t);
      mlir::Value ifVal = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
      ifClauseOperand = firOpBuilder.createConvert(
          currentLocation, firOpBuilder.getI1Type(), ifVal);
    } else if (const auto &numThreadsClause =
                   std::get_if<Fortran::parser::OmpClause::NumThreads>(
                       &clause.u)) {
      // OMPIRBuilder expects `NUM_THREAD` clause as a `Value`.
      numThreadsClauseOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numThreadsClause->v), stmtCtx));
    } else if (const auto &procBindClause =
                   std::get_if<Fortran::parser::OmpClause::ProcBind>(
                       &clause.u)) {
      omp::ClauseProcBindKind pbKind;
      switch (procBindClause->v.v) {
      case Fortran::parser::OmpProcBindClause::Type::Master:
        pbKind = omp::ClauseProcBindKind::Master;
        break;
      case Fortran::parser::OmpProcBindClause::Type::Close:
        pbKind = omp::ClauseProcBindKind::Close;
        break;
      case Fortran::parser::OmpProcBindClause::Type::Spread:
        pbKind = omp::ClauseProcBindKind::Spread;
        break;
      case Fortran::parser::OmpProcBindClause::Type::Primary:
        pbKind = omp::ClauseProcBindKind::Primary;
        break;
      }
      procBindKindAttr =
          omp::ClauseProcBindKindAttr::get(firOpBuilder.getContext(), pbKind);
    } else if (const auto &allocateClause =
                   std::get_if<Fortran::parser::OmpClause::Allocate>(
                       &clause.u)) {
      genAllocateClause(converter, allocateClause->v, allocatorOperands,
                        allocateOperands);
    } else if (std::get_if<Fortran::parser::OmpClause::Private>(&clause.u) ||
               std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                   &clause.u)) {
      // Privatisation clauses are handled elsewhere.
      continue;
    } else if (std::get_if<Fortran::parser::OmpClause::Threads>(&clause.u)) {
      // Nothing needs to be done for threads clause.
      continue;
    } else if (const auto &finalClause =
                   std::get_if<Fortran::parser::OmpClause::Final>(&clause.u)) {
      mlir::Value finalVal = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(finalClause->v), stmtCtx));
      finalClauseOperand = firOpBuilder.createConvert(
          currentLocation, firOpBuilder.getI1Type(), finalVal);
    } else if (std::get_if<Fortran::parser::OmpClause::Untied>(&clause.u)) {
      untiedAttr = firOpBuilder.getUnitAttr();
    } else if (std::get_if<Fortran::parser::OmpClause::Mergeable>(&clause.u)) {
      mergeableAttr = firOpBuilder.getUnitAttr();
    } else if (const auto &priorityClause =
                   std::get_if<Fortran::parser::OmpClause::Priority>(
                       &clause.u)) {
      priorityClauseOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(priorityClause->v), stmtCtx));
    } else {
      TODO(currentLocation, "OpenMP Block construct clauses");
    }
  }

  for (const auto &clause :
       std::get<Fortran::parser::OmpClauseList>(endBlockDirective.t).v) {
    if (std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u))
      nowaitAttr = firOpBuilder.getUnitAttr();
  }

  if (blockDirective.v == llvm::omp::OMPD_parallel) {
    // Create and insert the operation.
    auto parallelOp = firOpBuilder.create<mlir::omp::ParallelOp>(
        currentLocation, argTy, ifClauseOperand, numThreadsClauseOperand,
        allocateOperands, allocatorOperands, /*reduction_vars=*/ValueRange(),
        /*reductions=*/nullptr, procBindKindAttr);
    createBodyOfOp<omp::ParallelOp>(parallelOp, converter, currentLocation,
                                    &opClauseList);
  } else if (blockDirective.v == llvm::omp::OMPD_master) {
    auto masterOp =
        firOpBuilder.create<mlir::omp::MasterOp>(currentLocation, argTy);
    createBodyOfOp<omp::MasterOp>(masterOp, converter, currentLocation);
  } else if (blockDirective.v == llvm::omp::OMPD_single) {
    auto singleOp = firOpBuilder.create<mlir::omp::SingleOp>(
        currentLocation, allocateOperands, allocatorOperands, nowaitAttr);
    createBodyOfOp<omp::SingleOp>(singleOp, converter, currentLocation);
  } else if (blockDirective.v == llvm::omp::OMPD_ordered) {
    auto orderedOp = firOpBuilder.create<mlir::omp::OrderedRegionOp>(
        currentLocation, /*simd=*/nullptr);
    createBodyOfOp<omp::OrderedRegionOp>(orderedOp, converter, currentLocation);
  } else if (blockDirective.v == llvm::omp::OMPD_task) {
    auto taskOp = firOpBuilder.create<mlir::omp::TaskOp>(
        currentLocation, ifClauseOperand, finalClauseOperand, untiedAttr,
        mergeableAttr, /*in_reduction_vars=*/ValueRange(),
        /*in_reductions=*/nullptr, priorityClauseOperand, allocateOperands,
        allocatorOperands);
    createBodyOfOp(taskOp, converter, currentLocation, &opClauseList);
  } else {
    TODO(converter.getCurrentLocation(), "Unhandled block directive");
  }
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  llvm::SmallVector<mlir::Value> lowerBound, upperBound, step, linearVars,
      linearStepVars, reductionVars;
  mlir::Value scheduleChunkClauseOperand;
  mlir::Attribute scheduleClauseOperand, collapseClauseOperand,
      noWaitClauseOperand, orderedClauseOperand, orderClauseOperand;
  const auto &wsLoopOpClauseList = std::get<Fortran::parser::OmpClauseList>(
      std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t).t);
  if (llvm::omp::OMPD_do !=
      std::get<Fortran::parser::OmpLoopDirective>(
          std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t).t)
          .v) {
    TODO(converter.getCurrentLocation(), "Combined worksharing loop construct");
  }

  Fortran::lower::pft::Evaluation *doConstructEval =
      &eval.getFirstNestedEvaluation();

  Fortran::lower::pft::Evaluation *doLoop =
      &doConstructEval->getFirstNestedEvaluation();
  auto *doStmt = doLoop->getIf<Fortran::parser::NonLabelDoStmt>();
  assert(doStmt && "Expected do loop to be in the nested evaluation");
  const auto &loopControl =
      std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
  const Fortran::parser::LoopControl::Bounds *bounds =
      std::get_if<Fortran::parser::LoopControl::Bounds>(&loopControl->u);
  assert(bounds && "Expected bounds for worksharing do loop");
  Fortran::semantics::Symbol *iv = nullptr;
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
  iv = bounds->name.thing.symbol;

  // FIXME: Add support for following clauses:
  // 1. linear
  // 2. order
  // 3. collapse
  // 4. schedule (with chunk)
  auto wsLoopOp = firOpBuilder.create<mlir::omp::WsLoopOp>(
      currentLocation, lowerBound, upperBound, step, linearVars, linearStepVars,
      reductionVars, /*reductions=*/nullptr,
      scheduleClauseOperand.dyn_cast_or_null<omp::ClauseScheduleKindAttr>(),
      scheduleChunkClauseOperand, /*schedule_modifiers=*/nullptr,
      /*simd_modifier=*/nullptr,
      collapseClauseOperand.dyn_cast_or_null<IntegerAttr>(),
      noWaitClauseOperand.dyn_cast_or_null<UnitAttr>(),
      orderedClauseOperand.dyn_cast_or_null<IntegerAttr>(),
      orderClauseOperand.dyn_cast_or_null<omp::ClauseOrderKindAttr>(),
      /*inclusive=*/firOpBuilder.getUnitAttr());

  // Handle attribute based clauses.
  for (const Fortran::parser::OmpClause &clause : wsLoopOpClauseList.v) {
    if (const auto &orderedClause =
            std::get_if<Fortran::parser::OmpClause::Ordered>(&clause.u)) {
      if (orderedClause->v.has_value()) {
        const auto *expr = Fortran::semantics::GetExpr(orderedClause->v);
        const std::optional<std::int64_t> orderedClauseValue =
            Fortran::evaluate::ToInt64(*expr);
        wsLoopOp.ordered_valAttr(
            firOpBuilder.getI64IntegerAttr(*orderedClauseValue));
      } else {
        wsLoopOp.ordered_valAttr(firOpBuilder.getI64IntegerAttr(0));
      }
    } else if (const auto &scheduleClause =
                   std::get_if<Fortran::parser::OmpClause::Schedule>(
                       &clause.u)) {
      mlir::MLIRContext *context = firOpBuilder.getContext();
      const auto &scheduleType = scheduleClause->v;
      const auto &scheduleKind =
          std::get<Fortran::parser::OmpScheduleClause::ScheduleType>(
              scheduleType.t);
      switch (scheduleKind) {
      case Fortran::parser::OmpScheduleClause::ScheduleType::Static:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Static));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Dynamic:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Dynamic));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Guided:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Guided));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Auto:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Auto));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Runtime:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Runtime));
        break;
      }
    }
  }
  // In FORTRAN `nowait` clause occur at the end of `omp do` directive.
  // i.e
  // !$omp do
  // <...>
  // !$omp end do nowait
  if (const auto &endClauseList =
          std::get<std::optional<Fortran::parser::OmpEndLoopDirective>>(
              loopConstruct.t)) {
    const auto &clauseList =
        std::get<Fortran::parser::OmpClauseList>((*endClauseList).t);
    for (const Fortran::parser::OmpClause &clause : clauseList.v)
      if (std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u))
        wsLoopOp.nowaitAttr(firOpBuilder.getUnitAttr());
  }

  createBodyOfOp<omp::WsLoopOp>(wsLoopOp, converter, currentLocation,
                                &wsLoopOpClauseList, iv);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPCriticalConstruct &criticalConstruct) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  std::string name;
  const Fortran::parser::OmpCriticalDirective &cd =
      std::get<Fortran::parser::OmpCriticalDirective>(criticalConstruct.t);
  if (std::get<std::optional<Fortran::parser::Name>>(cd.t).has_value()) {
    name =
        std::get<std::optional<Fortran::parser::Name>>(cd.t).value().ToString();
  }

  uint64_t hint = 0;
  const auto &clauseList = std::get<Fortran::parser::OmpClauseList>(cd.t);
  for (const Fortran::parser::OmpClause &clause : clauseList.v)
    if (auto hintClause =
            std::get_if<Fortran::parser::OmpClause::Hint>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(hintClause->v);
      hint = *Fortran::evaluate::ToInt64(*expr);
      break;
    }

  mlir::omp::CriticalOp criticalOp = [&]() {
    if (name.empty()) {
      return firOpBuilder.create<mlir::omp::CriticalOp>(currentLocation,
                                                        FlatSymbolRefAttr());
    } else {
      mlir::ModuleOp module = firOpBuilder.getModule();
      mlir::OpBuilder modBuilder(module.getBodyRegion());
      auto global = module.lookupSymbol<mlir::omp::CriticalDeclareOp>(name);
      if (!global)
        global = modBuilder.create<mlir::omp::CriticalDeclareOp>(
            currentLocation, name, hint);
      return firOpBuilder.create<mlir::omp::CriticalOp>(
          currentLocation, mlir::FlatSymbolRefAttr::get(
                               firOpBuilder.getContext(), global.sym_name()));
    }
  }();
  createBodyOfOp<omp::CriticalOp>(criticalOp, converter, currentLocation);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPSectionConstruct &sectionConstruct) {

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  mlir::omp::SectionOp sectionOp =
      firOpBuilder.create<mlir::omp::SectionOp>(currentLocation);
  createBodyOfOp<omp::SectionOp>(sectionOp, converter, currentLocation);
}

// TODO: Add support for reduction
static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPSectionsConstruct &sectionsConstruct) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  SmallVector<Value> reductionVars, allocateOperands, allocatorOperands;
  mlir::UnitAttr noWaitClauseOperand;
  const auto &sectionsClauseList = std::get<Fortran::parser::OmpClauseList>(
      std::get<Fortran::parser::OmpBeginSectionsDirective>(sectionsConstruct.t)
          .t);
  for (const Fortran::parser::OmpClause &clause : sectionsClauseList.v) {

    // Reduction Clause
    if (std::get_if<Fortran::parser::OmpClause::Reduction>(&clause.u)) {
      TODO(currentLocation, "OMPC_Reduction");

      // Allocate clause
    } else if (const auto &allocateClause =
                   std::get_if<Fortran::parser::OmpClause::Allocate>(
                       &clause.u)) {
      genAllocateClause(converter, allocateClause->v, allocatorOperands,
                        allocateOperands);
    }
  }
  const auto &endSectionsClauseList =
      std::get<Fortran::parser::OmpEndSectionsDirective>(sectionsConstruct.t);
  const auto &clauseList =
      std::get<Fortran::parser::OmpClauseList>(endSectionsClauseList.t);
  for (const auto &clause : clauseList.v) {
    // Nowait clause
    if (std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u)) {
      noWaitClauseOperand = firOpBuilder.getUnitAttr();
    }
  }

  llvm::omp::Directive dir =
      std::get<Fortran::parser::OmpSectionsDirective>(
          std::get<Fortran::parser::OmpBeginSectionsDirective>(
              sectionsConstruct.t)
              .t)
          .v;

  // Parallel Sections Construct
  if (dir == llvm::omp::Directive::OMPD_parallel_sections) {
    auto parallelOp = firOpBuilder.create<mlir::omp::ParallelOp>(
        currentLocation, /*if_expr_var*/ nullptr, /*num_threads_var*/ nullptr,
        allocateOperands, allocatorOperands, /*reduction_vars=*/ValueRange(),
        /*reductions=*/nullptr, /*proc_bind_val*/ nullptr);
    createBodyOfOp(parallelOp, converter, currentLocation);
    auto sectionsOp = firOpBuilder.create<mlir::omp::SectionsOp>(
        currentLocation, /*reduction_vars*/ ValueRange(),
        /*reductions=*/nullptr, /*allocate_vars*/ ValueRange(),
        /*allocators_vars*/ ValueRange(), /*nowait=*/nullptr);
    createBodyOfOp(sectionsOp, converter, currentLocation);

    // Sections Construct
  } else if (dir == llvm::omp::Directive::OMPD_sections) {
    auto sectionsOp = firOpBuilder.create<mlir::omp::SectionsOp>(
        currentLocation, reductionVars, /*reductions = */ nullptr,
        allocateOperands, allocatorOperands, noWaitClauseOperand);
    createBodyOfOp<omp::SectionsOp>(sectionsOp, converter, currentLocation);
  }
}

static void genOmpAtomicHintAndMemoryOrderClauses(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::OmpAtomicClauseList &clauseList,
    mlir::IntegerAttr &hint,
    mlir::omp::ClauseMemoryOrderKindAttr &memory_order) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  for (const auto &clause : clauseList.v) {
    if (auto ompClause = std::get_if<Fortran::parser::OmpClause>(&clause.u)) {
      if (auto hintClause =
              std::get_if<Fortran::parser::OmpClause::Hint>(&ompClause->u)) {
        const auto *expr = Fortran::semantics::GetExpr(hintClause->v);
        uint64_t hintExprValue = *Fortran::evaluate::ToInt64(*expr);
        hint = firOpBuilder.getI64IntegerAttr(hintExprValue);
      }
    } else if (auto ompMemoryOrderClause =
                   std::get_if<Fortran::parser::OmpMemoryOrderClause>(
                       &clause.u)) {
      if (std::get_if<Fortran::parser::OmpClause::Acquire>(
              &ompMemoryOrderClause->v.u)) {
        memory_order = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Acquire);
      } else if (std::get_if<Fortran::parser::OmpClause::Relaxed>(
                     &ompMemoryOrderClause->v.u)) {
        memory_order = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Relaxed);
      } else if (std::get_if<Fortran::parser::OmpClause::SeqCst>(
                     &ompMemoryOrderClause->v.u)) {
        memory_order = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Seq_cst);
      } else if (std::get_if<Fortran::parser::OmpClause::Release>(
                     &ompMemoryOrderClause->v.u)) {
        memory_order = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Release);
      }
    }
  }
}

static void
genOmpAtomicWrite(Fortran::lower::AbstractConverter &converter,
                  Fortran::lower::pft::Evaluation &eval,
                  const Fortran::parser::OmpAtomicWrite &atomicWrite) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  mlir::Value address;
  // If no hint clause is specified, the effect is as if
  // hint(omp_sync_hint_none) had been specified.
  mlir::IntegerAttr hint = nullptr;
  mlir::omp::ClauseMemoryOrderKindAttr memory_order = nullptr;
  const Fortran::parser::OmpAtomicClauseList &rightHandClauseList =
      std::get<2>(atomicWrite.t);
  const Fortran::parser::OmpAtomicClauseList &leftHandClauseList =
      std::get<0>(atomicWrite.t);
  const auto &assignmentStmtExpr =
      std::get<Fortran::parser::Expr>(std::get<3>(atomicWrite.t).statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<3>(atomicWrite.t).statement.t);
  Fortran::lower::StatementContext stmtCtx;
  auto value = fir::getBase(converter.genExprValue(
      *Fortran::semantics::GetExpr(assignmentStmtExpr), stmtCtx));
  if (auto varDesignator = std::get_if<
          Fortran::common::Indirection<Fortran::parser::Designator>>(
          &assignmentStmtVariable.u)) {
    if (const auto *name = getDesignatorNameIfDataRef(varDesignator->value())) {
      address = converter.getSymbolAddress(*name->symbol);
    }
  }

  genOmpAtomicHintAndMemoryOrderClauses(converter, leftHandClauseList, hint,
                                        memory_order);
  genOmpAtomicHintAndMemoryOrderClauses(converter, rightHandClauseList, hint,
                                        memory_order);
  firOpBuilder.create<mlir::omp::AtomicWriteOp>(currentLocation, address, value,
                                                hint, memory_order);
}

static void genOmpAtomicRead(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::pft::Evaluation &eval,
                             const Fortran::parser::OmpAtomicRead &atomicRead) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  mlir::Value to_address;
  mlir::Value from_address;
  // If no hint clause is specified, the effect is as if
  // hint(omp_sync_hint_none) had been specified.
  mlir::IntegerAttr hint = nullptr;
  mlir::omp::ClauseMemoryOrderKindAttr memory_order = nullptr;
  const Fortran::parser::OmpAtomicClauseList &rightHandClauseList =
      std::get<2>(atomicRead.t);
  const Fortran::parser::OmpAtomicClauseList &leftHandClauseList =
      std::get<0>(atomicRead.t);
  const auto &assignmentStmtExpr =
      std::get<Fortran::parser::Expr>(std::get<3>(atomicRead.t).statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<3>(atomicRead.t).statement.t);
  if (auto exprDesignator = std::get_if<
          Fortran::common::Indirection<Fortran::parser::Designator>>(
          &assignmentStmtExpr.u)) {
    if (const auto *name =
            getDesignatorNameIfDataRef(exprDesignator->value())) {
      from_address = converter.getSymbolAddress(*name->symbol);
    }
  }

  if (auto varDesignator = std::get_if<
          Fortran::common::Indirection<Fortran::parser::Designator>>(
          &assignmentStmtVariable.u)) {
    if (const auto *name = getDesignatorNameIfDataRef(varDesignator->value())) {
      to_address = converter.getSymbolAddress(*name->symbol);
    }
  }

  genOmpAtomicHintAndMemoryOrderClauses(converter, leftHandClauseList, hint,
                                        memory_order);
  genOmpAtomicHintAndMemoryOrderClauses(converter, rightHandClauseList, hint,
                                        memory_order);
  firOpBuilder.create<mlir::omp::AtomicReadOp>(currentLocation, from_address,
                                               to_address, hint, memory_order);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
  std::visit(Fortran::common::visitors{
                 [&](const Fortran::parser::OmpAtomicRead &atomicRead) {
                   genOmpAtomicRead(converter, eval, atomicRead);
                 },
                 [&](const Fortran::parser::OmpAtomicWrite &atomicWrite) {
                   genOmpAtomicWrite(converter, eval, atomicWrite);
                 },
                 [&](const auto &) {
                   TODO(converter.getCurrentLocation(),
                        "Atomic update & capture");
                 },
             },
             atomicConstruct.u);
}

void Fortran::lower::genOpenMPConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPConstruct &ompConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenMPStandaloneConstruct
                  &standaloneConstruct) {
            genOMP(converter, eval, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionsConstruct
                  &sectionsConstruct) {
            genOMP(converter, eval, sectionsConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionConstruct &sectionConstruct) {
            genOMP(converter, eval, sectionConstruct);
          },
          [&](const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {
            genOMP(converter, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPExecutableAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPExecutableAllocate");
          },
          [&](const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
            genOMP(converter, eval, blockConstruct);
          },
          [&](const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
            genOMP(converter, eval, atomicConstruct);
          },
          [&](const Fortran::parser::OpenMPCriticalConstruct
                  &criticalConstruct) {
            genOMP(converter, eval, criticalConstruct);
          },
      },
      ompConstruct.u);
}

void Fortran::lower::genOpenMPDeclarativeConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &ompDeclConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &declarativeAllocate) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPDeclareReductionConstruct
                  &declareReductionConstruct) {
            TODO(converter.getCurrentLocation(),
                 "OpenMPDeclareReductionConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareSimdConstruct
                  &declareSimdConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclareSimdConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct
                  &declareTargetConstruct) {
            TODO(converter.getCurrentLocation(),
                 "OpenMPDeclareTargetConstruct");
          },
          [&](const Fortran::parser::OpenMPThreadprivate &threadprivate) {
            TODO(converter.getCurrentLocation(), "OpenMPThreadprivate");
          },
      },
      ompDeclConstruct.u);
}
