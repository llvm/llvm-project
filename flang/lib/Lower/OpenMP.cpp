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
#include "StatementContext.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

static const Fortran::parser::Name *
getDesignatorNameIfDataRef(const Fortran::parser::Designator &designator) {
  const auto *dataRef = std::get_if<Fortran::parser::DataRef>(&designator.u);
  return dataRef ? std::get_if<Fortran::parser::Name>(&dataRef->u) : nullptr;
}

static void genObjectList(const Fortran::parser::OmpObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          SmallVectorImpl<Value> &operands) {
  for (const auto &ompObject : objectList.v) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const auto *name = getDesignatorNameIfDataRef(designator)) {
                const auto variable = converter.getSymbolAddress(*name->symbol);
                operands.push_back(variable);
              }
            },
            [&](const Fortran::parser::Name &name) {
              const auto variable = converter.getSymbolAddress(*name.symbol);
              operands.push_back(variable);
            }},
        ompObject.u);
  }
}

template <typename Op>
static void createBodyOfOp(Op &op, Fortran::lower::AbstractConverter &converter,
                           mlir::Location &loc,
                           const Fortran::semantics::Symbol *arg = nullptr) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  // If an argument for the region is provided then create the block with that
  // argument. Also update the symbol's address with the mlir argument value.
  // e.g. For loops the argument is the induction variable. And all further
  // uses of the induction variable should use this mlir value.
  if (arg) {
    firOpBuilder.createBlock(&op.getRegion(), {}, {converter.genType(*arg)});
    converter.bindSymbol(*arg, op.getRegion().front().getArgument(0));
  } else {
    firOpBuilder.createBlock(&op.getRegion());
  }
  auto &block = op.getRegion().back();
  firOpBuilder.setInsertionPointToStart(&block);
  // Ensure the block is well-formed by inserting terminators.
  if constexpr (std::is_same_v<Op, omp::WsLoopOp>) {
    mlir::ValueRange results;
    firOpBuilder.create<mlir::omp::YieldOp>(loc, results);
  } else {
    firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
  }
  // Reset the insertion point to the start of the first block.
  firOpBuilder.setInsertionPointToStart(&block);
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
            // FIXME: Add support for handling memory order clause. Adding
            // a TODO will invoke a crash, so commented it for now.
            // if (std::get<std::optional<
            //        std::list<Fortran::parser::OmpMemoryOrderClause>>>(
            //        flushConstruct.t))
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

template <typename Directive>
static void createParallelOp(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::pft::Evaluation &eval,
                             const Directive &directive) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  Fortran::lower::StatementContext stmtCtx;
  llvm::ArrayRef<mlir::Type> argTy;
  mlir::Value ifClauseOperand, numThreadsClauseOperand;
  SmallVector<Value, 4> privateClauseOperands, firstprivateClauseOperands,
      sharedClauseOperands, copyinClauseOperands, allocatorOperands,
      allocateOperands;
  Attribute defaultClauseOperand, procBindClauseOperand;
  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(directive.t);
  for (const auto &clause : opClauseList.v) {
    if (const auto &ifClause =
            std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->v.t);
      ifClauseOperand = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
    } else if (const auto &numThreadsClause =
                   std::get_if<Fortran::parser::OmpClause::NumThreads>(
                       &clause.u)) {
      numThreadsClauseOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numThreadsClause->v), stmtCtx));
    } else if (const auto &privateClause =
                   std::get_if<Fortran::parser::OmpClause::Private>(
                       &clause.u)) {
      const Fortran::parser::OmpObjectList &ompObjectList = privateClause->v;
      genObjectList(ompObjectList, converter, privateClauseOperands);
    } else if (const auto &firstprivateClause =
                   std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                       &clause.u)) {
      const Fortran::parser::OmpObjectList &ompObjectList =
          firstprivateClause->v;
      genObjectList(ompObjectList, converter, firstprivateClauseOperands);
    } else if (const auto &sharedClause =
                   std::get_if<Fortran::parser::OmpClause::Shared>(&clause.u)) {
      const Fortran::parser::OmpObjectList &ompObjectList = sharedClause->v;
      genObjectList(ompObjectList, converter, sharedClauseOperands);
    } else if (const auto &copyinClause =
                   std::get_if<Fortran::parser::OmpClause::Copyin>(&clause.u)) {
      const Fortran::parser::OmpObjectList &ompObjectList = copyinClause->v;
      genObjectList(ompObjectList, converter, copyinClauseOperands);
    } else if (const auto &allocateClause =
                   std::get_if<Fortran::parser::OmpClause::Allocate>(
                       &clause.u)) {
      mlir::Value allocatorOperand;
      const Fortran::parser::OmpAllocateClause &ompAllocateClause =
          allocateClause->v;
      const Fortran::parser::OmpObjectList &ompObjectList =
          std::get<Fortran::parser::OmpObjectList>(ompAllocateClause.t);
      // Check if allocate clause has allocator specified. If so, add it
      // to list of allocators, otherwise, add default allocator to
      // list of allocators.
      const auto &allocatorValue = std::get<
          std::optional<Fortran::parser::OmpAllocateClause::Allocator>>(
          ompAllocateClause.t);
      if (allocatorValue) {
        allocatorOperand = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(allocatorValue->v), stmtCtx));
        allocatorOperands.insert(allocatorOperands.end(),
                                 ompObjectList.v.size(), allocatorOperand);
      } else {
        allocatorOperand = firOpBuilder.createIntegerConstant(
            currentLocation, firOpBuilder.getI32Type(), 1);
        allocatorOperands.insert(allocatorOperands.end(),
                                 ompObjectList.v.size(), allocatorOperand);
      }
      genObjectList(ompObjectList, converter, allocateOperands);
    }
  }
  // Create and insert the operation.
  auto parallelOp = firOpBuilder.create<mlir::omp::ParallelOp>(
      currentLocation, argTy, ifClauseOperand, numThreadsClauseOperand,
      defaultClauseOperand.dyn_cast_or_null<StringAttr>(),
      privateClauseOperands, firstprivateClauseOperands, sharedClauseOperands,
      copyinClauseOperands, allocateOperands, allocatorOperands,
      procBindClauseOperand.dyn_cast_or_null<StringAttr>());
  for (const auto &clause : opClauseList.v) {
    if (const auto &defaultClause =
            std::get_if<Fortran::parser::OmpClause::Default>(&clause.u)) {
      const auto &ompDefaultClause{defaultClause->v};
      switch (ompDefaultClause.v) {
      case Fortran::parser::OmpDefaultClause::Type::Private:
        parallelOp.default_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseDefault(omp::ClauseDefault::defprivate)));
        break;
      case Fortran::parser::OmpDefaultClause::Type::Firstprivate:
        parallelOp.default_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseDefault(omp::ClauseDefault::deffirstprivate)));
        break;
      case Fortran::parser::OmpDefaultClause::Type::Shared:
        parallelOp.default_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseDefault(omp::ClauseDefault::defshared)));
        break;
      case Fortran::parser::OmpDefaultClause::Type::None:
        parallelOp.default_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseDefault(omp::ClauseDefault::defnone)));
        break;
      }
    }
    if (const auto &procBindClause =
            std::get_if<Fortran::parser::OmpClause::ProcBind>(&clause.u)) {
      const auto &ompProcBindClause{procBindClause->v};
      switch (ompProcBindClause.v) {
      case Fortran::parser::OmpProcBindClause::Type::Master:
        parallelOp.proc_bind_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseProcBindKind(omp::ClauseProcBindKind::master)));
        break;
      case Fortran::parser::OmpProcBindClause::Type::Close:
        parallelOp.proc_bind_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseProcBindKind(omp::ClauseProcBindKind::close)));
        break;
      case Fortran::parser::OmpProcBindClause::Type::Spread:
        parallelOp.proc_bind_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseProcBindKind(omp::ClauseProcBindKind::spread)));
        break;
      }
    }
  }
  createBodyOfOp<omp::ParallelOp>(parallelOp, converter, currentLocation);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &blockDirective =
      std::get<Fortran::parser::OmpBlockDirective>(beginBlockDirective.t);

  if (blockDirective.v == llvm::omp::OMPD_parallel) {
    createParallelOp<Fortran::parser::OmpBeginBlockDirective>(
        converter, eval,
        std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t));
  } else if (blockDirective.v == llvm::omp::OMPD_master) {
    auto &firOpBuilder = converter.getFirOpBuilder();
    auto currentLocation = converter.getCurrentLocation();
    auto masterOp = firOpBuilder.create<mlir::omp::MasterOp>(currentLocation);
    createBodyOfOp<omp::MasterOp>(masterOp, converter, currentLocation);
  }
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  SmallVector<Value, 4> lowerBound, upperBound, step, privateClauseOperands,
      firstPrivateClauseOperands, lastPrivateClauseOperands, linearVars,
      linearStepVars;
  mlir::Value scheduleChunkClauseOperand;
  mlir::Attribute scheduleClauseOperand, collapseClauseOperand,
      noWaitClauseOperand, orderedClauseOperand, orderClauseOperand;
  const auto &wsLoopOpClauseList = std::get<Fortran::parser::OmpClauseList>(
      std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t).t);
  if (llvm::omp::OMPD_parallel_do ==
      std::get<Fortran::parser::OmpLoopDirective>(
          std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t).t)
          .v) {
    createParallelOp<Fortran::parser::OmpBeginLoopDirective>(
        converter, eval,
        std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t));
  } else {
    for (const auto &clause : wsLoopOpClauseList.v) {
      if (const auto &privateClause =
              std::get_if<Fortran::parser::OmpClause::Private>(&clause.u)) {
        const Fortran::parser::OmpObjectList &ompObjectList = privateClause->v;
        genObjectList(ompObjectList, converter, privateClauseOperands);
      } else if (const auto &firstPrivateClause =
                     std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                         &clause.u)) {
        const Fortran::parser::OmpObjectList &ompObjectList =
            firstPrivateClause->v;
        genObjectList(ompObjectList, converter, firstPrivateClauseOperands);
      }
    }
  }
  for (const auto &clause : wsLoopOpClauseList.v) {
    if (const auto &lastPrivateClause =
            std::get_if<Fortran::parser::OmpClause::Lastprivate>(&clause.u)) {
      const Fortran::parser::OmpObjectList &ompObjectList =
          lastPrivateClause->v;
      genObjectList(ompObjectList, converter, lastPrivateClauseOperands);
    }
  }
  // FIXME: Can be done in a better way ?
  auto &doConstructEval =
      eval.getFirstNestedEvaluation().getFirstNestedEvaluation();
  auto *doStmt = doConstructEval.getIf<Fortran::parser::NonLabelDoStmt>();

  const auto &loopControl =
      std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
  const Fortran::parser::LoopControl::Bounds *bounds =
      std::get_if<Fortran::parser::LoopControl::Bounds>(&loopControl->u);
  Fortran::semantics::Symbol *iv = nullptr;
  if (bounds) {
    Fortran::lower::StatementContext stmtCtx;
    lowerBound.push_back(fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(bounds->lower), stmtCtx)));
    upperBound.push_back(fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(bounds->upper), stmtCtx)));
    if (bounds->step) {
      step.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(bounds->step), stmtCtx)));
    }
    // If `step` is not present, assume it as `1`.
    else {
      step.push_back(firOpBuilder.createIntegerConstant(
          currentLocation, firOpBuilder.getIntegerType(32), 1));
    }
    iv = bounds->name.thing.symbol;
  }
  // FIXME: Add support for following clauses:
  // 1. linear
  // 2. order
  auto wsLoopOp = firOpBuilder.create<mlir::omp::WsLoopOp>(
      currentLocation, lowerBound, upperBound, step, privateClauseOperands,
      firstPrivateClauseOperands, lastPrivateClauseOperands, linearVars,
      linearStepVars, scheduleClauseOperand.dyn_cast_or_null<StringAttr>(),
      scheduleChunkClauseOperand,
      collapseClauseOperand.dyn_cast_or_null<IntegerAttr>(),
      noWaitClauseOperand.dyn_cast_or_null<UnitAttr>(),
      orderedClauseOperand.dyn_cast_or_null<IntegerAttr>(),
      orderClauseOperand.dyn_cast_or_null<StringAttr>(),
      firOpBuilder.getUnitAttr() /* Inclusive stop */);

  // Handle attribute based clauses.
  for (const auto &clause : wsLoopOpClauseList.v) {
    if (const auto &collapseClause =
            std::get_if<Fortran::parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      const auto collapseValue = Fortran::evaluate::ToInt64(*expr);
      wsLoopOp.collapse_valAttr(firOpBuilder.getI64IntegerAttr(*collapseValue));
    } else if (const auto &orderedClause =
                   std::get_if<Fortran::parser::OmpClause::Ordered>(
                       &clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(orderedClause->v);
      const auto orderedValue = Fortran::evaluate::ToInt64(*expr);
      wsLoopOp.ordered_valAttr(firOpBuilder.getI64IntegerAttr(*orderedValue));
    } else if (const auto &scheduleClause =
                   std::get_if<Fortran::parser::OmpClause::Schedule>(
                       &clause.u)) {
      const auto &scheduleType = scheduleClause->v;
      const auto &scheduleKind =
          std::get<Fortran::parser::OmpScheduleClause::ScheduleType>(
              scheduleType.t);
      switch (scheduleKind) {
      case Fortran::parser::OmpScheduleClause::ScheduleType::Static:
        wsLoopOp.schedule_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseScheduleKind(omp::ClauseScheduleKind::Static)));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Dynamic:
        wsLoopOp.schedule_valAttr(
            firOpBuilder.getStringAttr(omp::stringifyClauseScheduleKind(
                omp::ClauseScheduleKind::Dynamic)));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Guided:
        wsLoopOp.schedule_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseScheduleKind(omp::ClauseScheduleKind::Guided)));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Auto:
        wsLoopOp.schedule_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseScheduleKind(omp::ClauseScheduleKind::Auto)));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Runtime:
        wsLoopOp.schedule_valAttr(
            firOpBuilder.getStringAttr(omp::stringifyClauseScheduleKind(
                omp::ClauseScheduleKind::Runtime)));
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
    for (const auto &clause : clauseList.v)
      if (std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u))
        wsLoopOp.nowaitAttr(firOpBuilder.getUnitAttr());
  }

  createBodyOfOp<omp::WsLoopOp>(wsLoopOp, converter, currentLocation, iv);
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
            TODO(converter.getCurrentLocation(), "OpenMPSectionsConstruct");
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
            TODO(converter.getCurrentLocation(), "OpenMPAtomicConstruct");
          },
          [&](const Fortran::parser::OpenMPCriticalConstruct
                  &criticalConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCriticalConstruct");
          },
      },
      ompConstruct.u);
}
