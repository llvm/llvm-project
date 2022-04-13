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
               bool outerCombined = false) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  firOpBuilder.createBlock(&op.getRegion());
  auto &block = op.getRegion().back();
  firOpBuilder.setInsertionPointToStart(&block);
  // Ensure the block is well-formed.
  firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
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
  mlir::Value ifClauseOperand, numThreadsClauseOperand;
  mlir::omp::ClauseProcBindKindAttr procBindKindAttr;
  SmallVector<Value> allocateOperands, allocatorOperands;
  mlir::UnitAttr nowaitAttr;

  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginBlockDirective.t);
  for (const auto &clause : opClauseList.v) {
    if (const auto &ifClause =
            std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->v.t);
      ifClauseOperand = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
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
                                    &opClauseList, /*isCombined=*/false);
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
  } else {
    TODO(converter.getCurrentLocation(), "Unhandled block directive");
  }
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
            TODO(converter.getCurrentLocation(), "OpenMPLoopConstruct");
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
            genOMP(converter, eval, criticalConstruct);
          },
      },
      ompConstruct.u);
}
