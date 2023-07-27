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
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace mlir;

void Fortran::lower::genOpenMPTerminator(fir::FirOpBuilder &builder,
                                         Operation *op, mlir::Location loc) {
  if (mlir::isa<omp::WsLoopOp, omp::ReductionDeclareOp, omp::AtomicUpdateOp,
                omp::SimdLoopOp>(op))
    builder.create<omp::YieldOp>(loc);
  else
    builder.create<omp::TerminatorOp>(loc);
}

int64_t Fortran::lower::getCollapseValue(
    const Fortran::parser::OmpClauseList &clauseList) {
  for (const auto &clause : clauseList.v) {
    if (const auto &collapseClause =
            std::get_if<Fortran::parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      return Fortran::evaluate::ToInt64(*expr).value();
    }
  }
  return 1;
}

static Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject) {
  Fortran::semantics::Symbol *sym = nullptr;
  std::visit(Fortran::common::visitors{
                 [&](const Fortran::parser::Designator &designator) {
                   if (const Fortran::parser::Name *name =
                           Fortran::semantics::getDesignatorNameIfDataRef(
                               designator)) {
                     sym = name->symbol;
                   }
                 },
                 [&](const Fortran::parser::Name &name) { sym = name.symbol; }},
             ompObject.u);
  return sym;
}

class DataSharingProcessor {
  bool hasLastPrivateOp;
  mlir::OpBuilder::InsertPoint lastPrivIP;
  mlir::OpBuilder::InsertPoint insPt;
  // Symbols in private, firstprivate, and/or lastprivate clauses.
  llvm::SetVector<const Fortran::semantics::Symbol *> privatizedSymbols;
  llvm::SetVector<const Fortran::semantics::Symbol *> defaultSymbols;
  llvm::SetVector<const Fortran::semantics::Symbol *> symbolsInNestedRegions;
  llvm::SetVector<const Fortran::semantics::Symbol *> symbolsInParentRegions;
  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &firOpBuilder;
  const Fortran::parser::OmpClauseList &opClauseList;
  Fortran::lower::pft::Evaluation &eval;

  bool needBarrier();
  void collectSymbols(Fortran::semantics::Symbol::Flag flag);
  void collectOmpObjectListSymbol(
      const Fortran::parser::OmpObjectList &ompObjectList,
      llvm::SetVector<const Fortran::semantics::Symbol *> &symbolSet);
  void collectSymbolsForPrivatization();
  void insertBarrier();
  void collectDefaultSymbols();
  void privatize();
  void defaultPrivatize();
  void copyLastPrivatize(mlir::Operation *op);
  void insertLastPrivateCompare(mlir::Operation *op);
  void cloneSymbol(const Fortran::semantics::Symbol *sym);
  void copyFirstPrivateSymbol(const Fortran::semantics::Symbol *sym);
  void copyLastPrivateSymbol(const Fortran::semantics::Symbol *sym,
                             mlir::OpBuilder::InsertPoint *lastPrivIP);
  void insertDeallocs();

public:
  DataSharingProcessor(Fortran::lower::AbstractConverter &converter,
                       const Fortran::parser::OmpClauseList &opClauseList,
                       Fortran::lower::pft::Evaluation &eval)
      : hasLastPrivateOp(false), converter(converter),
        firOpBuilder(converter.getFirOpBuilder()), opClauseList(opClauseList),
        eval(eval) {}
  // Privatisation is split into two steps.
  // Step1 performs cloning of all privatisation clauses and copying for
  // firstprivates. Step1 is performed at the place where process/processStep1
  // is called. This is usually inside the Operation corresponding to the OpenMP
  // construct, for looping constructs this is just before the Operation. The
  // split into two steps was performed basically to be able to call
  // privatisation for looping constructs before the operation is created since
  // the bounds of the MLIR OpenMP operation can be privatised.
  // Step2 performs the copying for lastprivates and requires knowledge of the
  // MLIR operation to insert the last private update. Step2 adds
  // dealocation code as well.
  void processStep1();
  void processStep2(mlir::Operation *op, bool is_loop);
};

void DataSharingProcessor::processStep1() {
  collectSymbolsForPrivatization();
  collectDefaultSymbols();
  privatize();
  defaultPrivatize();
  insertBarrier();
}

void DataSharingProcessor::processStep2(mlir::Operation *op, bool is_loop) {
  insPt = firOpBuilder.saveInsertionPoint();
  copyLastPrivatize(op);
  firOpBuilder.restoreInsertionPoint(insPt);

  if (is_loop) {
    // push deallocs out of the loop
    firOpBuilder.setInsertionPointAfter(op);
    insertDeallocs();
  } else {
    // insert dummy instruction to mark the insertion position
    mlir::Value undefMarker = firOpBuilder.create<fir::UndefOp>(
        op->getLoc(), firOpBuilder.getIndexType());
    insertDeallocs();
    firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());
  }
}

void DataSharingProcessor::insertDeallocs() {
  for (auto sym : privatizedSymbols)
    if (Fortran::semantics::IsAllocatable(sym->GetUltimate())) {
      converter.createHostAssociateVarCloneDealloc(*sym);
    }
}

void DataSharingProcessor::cloneSymbol(const Fortran::semantics::Symbol *sym) {
  // Privatization for symbols which are pre-determined (like loop index
  // variables) happen separately, for everything else privatize here.
  if (sym->test(Fortran::semantics::Symbol::Flag::OmpPreDetermined))
    return;
  bool success = converter.createHostAssociateVarClone(*sym);
  (void)success;
  assert(success && "Privatization failed due to existing binding");
}

void DataSharingProcessor::copyFirstPrivateSymbol(
    const Fortran::semantics::Symbol *sym) {
  if (sym->test(Fortran::semantics::Symbol::Flag::OmpFirstPrivate))
    converter.copyHostAssociateVar(*sym);
}

void DataSharingProcessor::copyLastPrivateSymbol(
    const Fortran::semantics::Symbol *sym,
    [[maybe_unused]] mlir::OpBuilder::InsertPoint *lastPrivIP) {
  if (sym->test(Fortran::semantics::Symbol::Flag::OmpLastPrivate))
    converter.copyHostAssociateVar(*sym, lastPrivIP);
}

void DataSharingProcessor::collectOmpObjectListSymbol(
    const Fortran::parser::OmpObjectList &ompObjectList,
    llvm::SetVector<const Fortran::semantics::Symbol *> &symbolSet) {
  for (const Fortran::parser::OmpObject &ompObject : ompObjectList.v) {
    Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
    symbolSet.insert(sym);
  }
}

void DataSharingProcessor::collectSymbolsForPrivatization() {
  bool hasCollapse = false;
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (const auto &privateClause =
            std::get_if<Fortran::parser::OmpClause::Private>(&clause.u)) {
      collectOmpObjectListSymbol(privateClause->v, privatizedSymbols);
    } else if (const auto &firstPrivateClause =
                   std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                       &clause.u)) {
      collectOmpObjectListSymbol(firstPrivateClause->v, privatizedSymbols);
    } else if (const auto &lastPrivateClause =
                   std::get_if<Fortran::parser::OmpClause::Lastprivate>(
                       &clause.u)) {
      collectOmpObjectListSymbol(lastPrivateClause->v, privatizedSymbols);
      hasLastPrivateOp = true;
    } else if (std::get_if<Fortran::parser::OmpClause::Collapse>(&clause.u)) {
      hasCollapse = true;
    }
  }

  for (auto *ps : privatizedSymbols) {
    if (ps->has<Fortran::semantics::CommonBlockDetails>())
      TODO(converter.getCurrentLocation(),
           "Common Block in privatization clause");
  }

  if (hasCollapse && hasLastPrivateOp)
    TODO(converter.getCurrentLocation(), "Collapse clause with lastprivate");
}

bool DataSharingProcessor ::needBarrier() {
  for (auto sym : privatizedSymbols) {
    if (sym->test(Fortran::semantics::Symbol::Flag::OmpFirstPrivate) &&
        sym->test(Fortran::semantics::Symbol::Flag::OmpLastPrivate))
      return true;
  }
  return false;
}

void DataSharingProcessor ::insertBarrier() {
  // Emit implicit barrier to synchronize threads and avoid data races on
  // initialization of firstprivate variables and post-update of lastprivate
  // variables.
  // FIXME: Emit barrier for lastprivate clause when 'sections' directive has
  // 'nowait' clause. Otherwise, emit barrier when 'sections' directive has
  // both firstprivate and lastprivate clause.
  // Emit implicit barrier for linear clause. Maybe on somewhere else.
  if (needBarrier())
    firOpBuilder.create<mlir::omp::BarrierOp>(converter.getCurrentLocation());
}

void DataSharingProcessor::insertLastPrivateCompare(mlir::Operation *op) {
  mlir::arith::CmpIOp cmpOp;
  bool cmpCreated = false;
  mlir::OpBuilder::InsertPoint localInsPt = firOpBuilder.saveInsertionPoint();
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (std::get_if<Fortran::parser::OmpClause::Lastprivate>(&clause.u)) {
      // TODO: Add lastprivate support for simd construct
      if (mlir::isa<omp::SectionOp>(op)) {
        if (&eval == &eval.parentConstruct->getLastNestedEvaluation()) {
          // For `omp.sections`, lastprivatized variables occur in
          // lexically final `omp.section` operation. The following FIR
          // shall be generated for the same:
          //
          // omp.sections lastprivate(...) {
          //  omp.section {...}
          //  omp.section {...}
          //  omp.section {
          //      fir.allocate for `private`/`firstprivate`
          //      <More operations here>
          //      fir.if %true {
          //          ^%lpv_update_blk
          //      }
          //  }
          // }
          //
          // To keep code consistency while handling privatization
          // through this control flow, add a `fir.if` operation
          // that always evaluates to true, in order to create
          // a dedicated sub-region in `omp.section` where
          // lastprivate FIR can reside. Later canonicalizations
          // will optimize away this operation.
          if (!eval.lowerAsUnstructured()) {
            auto ifOp = firOpBuilder.create<fir::IfOp>(
                op->getLoc(),
                firOpBuilder.createIntegerConstant(
                    op->getLoc(), firOpBuilder.getIntegerType(1), 0x1),
                /*else*/ false);
            firOpBuilder.setInsertionPointToStart(
                &ifOp.getThenRegion().front());

            const Fortran::parser::OpenMPConstruct *parentOmpConstruct =
                eval.parentConstruct->getIf<Fortran::parser::OpenMPConstruct>();
            assert(parentOmpConstruct &&
                   "Expected a valid enclosing OpenMP construct");
            const Fortran::parser::OpenMPSectionsConstruct *sectionsConstruct =
                std::get_if<Fortran::parser::OpenMPSectionsConstruct>(
                    &parentOmpConstruct->u);
            assert(sectionsConstruct &&
                   "Expected an enclosing omp.sections construct");
            const Fortran::parser::OmpClauseList &sectionsEndClauseList =
                std::get<Fortran::parser::OmpClauseList>(
                    std::get<Fortran::parser::OmpEndSectionsDirective>(
                        sectionsConstruct->t)
                        .t);
            for (const Fortran::parser::OmpClause &otherClause :
                 sectionsEndClauseList.v)
              if (std::get_if<Fortran::parser::OmpClause::Nowait>(
                      &otherClause.u))
                // Emit implicit barrier to synchronize threads and avoid data
                // races on post-update of lastprivate variables when `nowait`
                // clause is present.
                firOpBuilder.create<mlir::omp::BarrierOp>(
                    converter.getCurrentLocation());
            firOpBuilder.setInsertionPointToStart(
                &ifOp.getThenRegion().front());
            lastPrivIP = firOpBuilder.saveInsertionPoint();
            firOpBuilder.setInsertionPoint(ifOp);
            insPt = firOpBuilder.saveInsertionPoint();
          } else {
            // Lastprivate operation is inserted at the end
            // of the lexically last section in the sections
            // construct
            mlir::OpBuilder::InsertPoint unstructuredSectionsIP =
                firOpBuilder.saveInsertionPoint();
            firOpBuilder.setInsertionPointToStart(&op->getRegion(0).back());
            lastPrivIP = firOpBuilder.saveInsertionPoint();
            firOpBuilder.restoreInsertionPoint(unstructuredSectionsIP);
          }
        }
      } else if (mlir::isa<omp::WsLoopOp>(op)) {
        mlir::Operation *lastOper = op->getRegion(0).back().getTerminator();
        firOpBuilder.setInsertionPoint(lastOper);

        // Update the original variable just before exiting the worksharing
        // loop. Conversion as follows:
        //
        //                       omp.wsloop {
        // omp.wsloop {            ...
        //    ...                  store
        //    store       ===>     %cmp = llvm.icmp "eq" %iv %ub
        //    omp.yield            fir.if %cmp {
        // }                         ^%lpv_update_blk:
        //                         }
        //                         omp.yield
        //                       }
        //

        // Only generate the compare once in presence of multiple LastPrivate
        // clauses.
        if (!cmpCreated) {
          cmpOp = firOpBuilder.create<mlir::arith::CmpIOp>(
              op->getLoc(), mlir::arith::CmpIPredicate::eq,
              op->getRegion(0).front().getArguments()[0],
              mlir::dyn_cast<mlir::omp::WsLoopOp>(op).getUpperBound()[0]);
        }
        auto ifOp =
            firOpBuilder.create<fir::IfOp>(op->getLoc(), cmpOp, /*else*/ false);
        firOpBuilder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        lastPrivIP = firOpBuilder.saveInsertionPoint();
      } else {
        TODO(converter.getCurrentLocation(),
             "lastprivate clause in constructs other than "
             "simd/worksharing-loop");
      }
    }
  }
  firOpBuilder.restoreInsertionPoint(localInsPt);
}

void DataSharingProcessor::collectSymbols(
    Fortran::semantics::Symbol::Flag flag) {
  converter.collectSymbolSet(eval, defaultSymbols, flag,
                             /*collectSymbols=*/true,
                             /*collectHostAssociatedSymbols=*/true);
  for (auto &e : eval.getNestedEvaluations()) {
    if (e.hasNestedEvaluations())
      converter.collectSymbolSet(e, symbolsInNestedRegions, flag,
                                 /*collectSymbols=*/true,
                                 /*collectHostAssociatedSymbols=*/false);
    else
      converter.collectSymbolSet(e, symbolsInParentRegions, flag,
                                 /*collectSymbols=*/false,
                                 /*collectHostAssociatedSymbols=*/true);
  }
}

void DataSharingProcessor::collectDefaultSymbols() {
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (const auto &defaultClause =
            std::get_if<Fortran::parser::OmpClause::Default>(&clause.u)) {
      if (defaultClause->v.v ==
          Fortran::parser::OmpDefaultClause::Type::Private)
        collectSymbols(Fortran::semantics::Symbol::Flag::OmpPrivate);
      else if (defaultClause->v.v ==
               Fortran::parser::OmpDefaultClause::Type::Firstprivate)
        collectSymbols(Fortran::semantics::Symbol::Flag::OmpFirstPrivate);
    }
  }
}

void DataSharingProcessor::privatize() {
  for (auto sym : privatizedSymbols) {
    cloneSymbol(sym);
    copyFirstPrivateSymbol(sym);
  }
}

void DataSharingProcessor::copyLastPrivatize(mlir::Operation *op) {
  insertLastPrivateCompare(op);
  for (auto sym : privatizedSymbols)
    copyLastPrivateSymbol(sym, &lastPrivIP);
}

void DataSharingProcessor::defaultPrivatize() {
  for (auto sym : defaultSymbols) {
    if (!symbolsInNestedRegions.contains(sym) &&
        !symbolsInParentRegions.contains(sym) &&
        !privatizedSymbols.contains(sym)) {
      cloneSymbol(sym);
      copyFirstPrivateSymbol(sym);
    }
  }
}

/// The COMMON block is a global structure. \p commonValue is the base address
/// of the the COMMON block. As the offset from the symbol \p sym, generate the
/// COMMON block member value (commonValue + offset) for the symbol.
/// FIXME: Share the code with `instantiateCommon` in ConvertVariable.cpp.
static mlir::Value
genCommonBlockMember(Fortran::lower::AbstractConverter &converter,
                     const Fortran::semantics::Symbol &sym,
                     mlir::Value commonValue) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::IntegerType i8Ty = firOpBuilder.getIntegerType(8);
  mlir::Type i8Ptr = firOpBuilder.getRefType(i8Ty);
  mlir::Type seqTy = firOpBuilder.getRefType(firOpBuilder.getVarLenSeqTy(i8Ty));
  mlir::Value base =
      firOpBuilder.createConvert(currentLocation, seqTy, commonValue);
  std::size_t byteOffset = sym.GetUltimate().offset();
  mlir::Value offs = firOpBuilder.createIntegerConstant(
      currentLocation, firOpBuilder.getIndexType(), byteOffset);
  mlir::Value varAddr = firOpBuilder.create<fir::CoordinateOp>(
      currentLocation, i8Ptr, base, mlir::ValueRange{offs});
  mlir::Type symType = converter.genType(sym);
  return firOpBuilder.createConvert(currentLocation,
                                    firOpBuilder.getRefType(symType), varAddr);
}

// Get the extended value for \p val by extracting additional variable
// information from \p base.
static fir::ExtendedValue getExtendedValue(fir::ExtendedValue base,
                                           mlir::Value val) {
  return base.match(
      [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
        return fir::MutableBoxValue(val, box.nonDeferredLenParams(), {});
      },
      [&](const auto &) -> fir::ExtendedValue {
        return fir::substBase(base, val);
      });
}

static void threadPrivatizeVars(Fortran::lower::AbstractConverter &converter,
                                Fortran::lower::pft::Evaluation &eval) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  auto insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  // Get the original ThreadprivateOp corresponding to the symbol and use the
  // symbol value from that opeartion to create one ThreadprivateOp copy
  // operation inside the parallel region.
  auto genThreadprivateOp = [&](Fortran::lower::SymbolRef sym) -> mlir::Value {
    mlir::Value symOriThreadprivateValue = converter.getSymbolAddress(sym);
    mlir::Operation *op = symOriThreadprivateValue.getDefiningOp();
    assert(mlir::isa<mlir::omp::ThreadprivateOp>(op) &&
           "The threadprivate operation not created");
    mlir::Value symValue =
        mlir::dyn_cast<mlir::omp::ThreadprivateOp>(op).getSymAddr();
    return firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  };

  llvm::SetVector<const Fortran::semantics::Symbol *> threadprivateSyms;
  converter.collectSymbolSet(
      eval, threadprivateSyms,
      Fortran::semantics::Symbol::Flag::OmpThreadprivate);
  std::set<Fortran::semantics::SourceName> threadprivateSymNames;

  // For a COMMON block, the ThreadprivateOp is generated for itself instead of
  // its members, so only bind the value of the new copied ThreadprivateOp
  // inside the parallel region to the common block symbol only once for
  // multiple members in one COMMON block.
  llvm::SetVector<const Fortran::semantics::Symbol *> commonSyms;
  for (std::size_t i = 0; i < threadprivateSyms.size(); i++) {
    auto sym = threadprivateSyms[i];
    mlir::Value symThreadprivateValue;
    // The variable may be used more than once, and each reference has one
    // symbol with the same name. Only do once for references of one variable.
    if (threadprivateSymNames.find(sym->name()) != threadprivateSymNames.end())
      continue;
    threadprivateSymNames.insert(sym->name());
    if (const Fortran::semantics::Symbol *common =
            Fortran::semantics::FindCommonBlockContaining(sym->GetUltimate())) {
      mlir::Value commonThreadprivateValue;
      if (commonSyms.contains(common)) {
        commonThreadprivateValue = converter.getSymbolAddress(*common);
      } else {
        commonThreadprivateValue = genThreadprivateOp(*common);
        converter.bindSymbol(*common, commonThreadprivateValue);
        commonSyms.insert(common);
      }
      symThreadprivateValue =
          genCommonBlockMember(converter, *sym, commonThreadprivateValue);
    } else {
      symThreadprivateValue = genThreadprivateOp(*sym);
    }

    fir::ExtendedValue sexv = converter.getSymbolExtendedValue(*sym);
    fir::ExtendedValue symThreadprivateExv =
        getExtendedValue(sexv, symThreadprivateValue);
    converter.bindSymbol(*sym, symThreadprivateExv);
  }

  firOpBuilder.restoreInsertionPoint(insPt);
}

static void
genCopyinClause(Fortran::lower::AbstractConverter &converter,
                const Fortran::parser::OmpClauseList &opClauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());
  bool hasCopyin = false;
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (const auto &copyinClause =
            std::get_if<Fortran::parser::OmpClause::Copyin>(&clause.u)) {
      hasCopyin = true;
      const Fortran::parser::OmpObjectList &ompObjectList = copyinClause->v;
      for (const Fortran::parser::OmpObject &ompObject : ompObjectList.v) {
        Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
        if (sym->has<Fortran::semantics::CommonBlockDetails>())
          TODO(converter.getCurrentLocation(), "common block in Copyin clause");
        if (Fortran::semantics::IsAllocatableOrPointer(sym->GetUltimate()))
          TODO(converter.getCurrentLocation(),
               "pointer or allocatable variables in Copyin clause");
        assert(sym->has<Fortran::semantics::HostAssocDetails>() &&
               "No host-association found");
        converter.copyHostAssociateVar(*sym);
      }
    }
  }
  // [OMP 5.0, 2.19.6.1] The copy is done after the team is formed and prior to
  // the execution of the associated structured block. Emit implicit barrier to
  // synchronize threads and avoid data races on propagation master's thread
  // values of threadprivate variables to local instances of that variables of
  // all other implicit threads.
  if (hasCopyin)
    firOpBuilder.create<mlir::omp::BarrierOp>(converter.getCurrentLocation());
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
    Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
    addOperands(*sym);
  }
}

static mlir::Value
getIfClauseOperand(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::StatementContext &stmtCtx,
                   const Fortran::parser::OmpClause::If *ifClause,
                   mlir::Location clauseLocation) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->v.t);
  mlir::Value ifVal = fir::getBase(
      converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
  return firOpBuilder.createConvert(clauseLocation, firOpBuilder.getI1Type(),
                                    ifVal);
}

static mlir::Type getLoopVarType(Fortran::lower::AbstractConverter &converter,
                                 std::size_t loopVarTypeSize) {
  // OpenMP runtime requires 32-bit or 64-bit loop variables.
  loopVarTypeSize = loopVarTypeSize * 8;
  if (loopVarTypeSize < 32) {
    loopVarTypeSize = 32;
  } else if (loopVarTypeSize > 64) {
    loopVarTypeSize = 64;
    mlir::emitWarning(converter.getCurrentLocation(),
                      "OpenMP loop iteration variable cannot have more than 64 "
                      "bits size and will be narrowed into 64 bits.");
  }
  assert((loopVarTypeSize == 32 || loopVarTypeSize == 64) &&
         "OpenMP loop iteration variable size must be transformed into 32-bit "
         "or 64-bit");
  return converter.getFirOpBuilder().getIntegerType(loopVarTypeSize);
}

/// Create empty blocks for the current region.
/// These blocks replace blocks parented to an enclosing region.
void createEmptyRegionBlocks(
    fir::FirOpBuilder &firOpBuilder,
    std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
  auto *region = &firOpBuilder.getRegion();
  for (auto &eval : evaluationList) {
    if (eval.block) {
      if (eval.block->empty()) {
        eval.block->erase();
        eval.block = firOpBuilder.createBlock(region);
      } else {
        [[maybe_unused]] auto &terminatorOp = eval.block->back();
        assert((mlir::isa<mlir::omp::TerminatorOp>(terminatorOp) ||
                mlir::isa<mlir::omp::YieldOp>(terminatorOp)) &&
               "expected terminator op");
      }
    }
    if (!eval.isDirective() && eval.hasNestedEvaluations())
      createEmptyRegionBlocks(firOpBuilder, eval.getNestedEvaluations());
  }
}

void resetBeforeTerminator(fir::FirOpBuilder &firOpBuilder,
                           mlir::Operation *storeOp, mlir::Block &block) {
  if (storeOp)
    firOpBuilder.setInsertionPointAfter(storeOp);
  else
    firOpBuilder.setInsertionPointToStart(&block);
}

/// Create the body (block) for an OpenMP Operation.
///
/// \param [in]    op - the operation the body belongs to.
/// \param [inout] converter - converter to use for the clauses.
/// \param [in]    loc - location in source code.
/// \param [in]    eval - current PFT node/evaluation.
/// \oaran [in]    clauses - list of clauses to process.
/// \param [in]    args - block arguments (induction variable[s]) for the
////                      region.
/// \param [in]    outerCombined - is this an outer operation - prevents
///                                privatization.
template <typename Op>
static void
createBodyOfOp(Op &op, Fortran::lower::AbstractConverter &converter,
               mlir::Location &loc, Fortran::lower::pft::Evaluation &eval,
               const Fortran::parser::OmpClauseList *clauses = nullptr,
               const SmallVector<const Fortran::semantics::Symbol *> &args = {},
               bool outerCombined = false,
               DataSharingProcessor *dsp = nullptr) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  // If an argument for the region is provided then create the block with that
  // argument. Also update the symbol's address with the mlir argument value.
  // e.g. For loops the argument is the induction variable. And all further
  // uses of the induction variable should use this mlir value.
  mlir::Operation *storeOp = nullptr;
  if (args.size()) {
    std::size_t loopVarTypeSize = 0;
    for (const Fortran::semantics::Symbol *arg : args)
      loopVarTypeSize = std::max(loopVarTypeSize, arg->GetUltimate().size());
    mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
    SmallVector<Type> tiv;
    SmallVector<Location> locs;
    for (int i = 0; i < (int)args.size(); i++) {
      tiv.push_back(loopVarType);
      locs.push_back(loc);
    }
    firOpBuilder.createBlock(&op.getRegion(), {}, tiv, locs);
    int argIndex = 0;
    // The argument is not currently in memory, so make a temporary for the
    // argument, and store it there, then bind that location to the argument.
    for (const Fortran::semantics::Symbol *arg : args) {
      mlir::Value val =
          fir::getBase(op.getRegion().front().getArgument(argIndex));
      mlir::Value temp = firOpBuilder.createTemporary(
          loc, loopVarType,
          llvm::ArrayRef<mlir::NamedAttribute>{
              Fortran::lower::getAdaptToByRefAttr(firOpBuilder)});
      storeOp = firOpBuilder.create<fir::StoreOp>(loc, val, temp);
      converter.bindSymbol(*arg, temp);
      argIndex++;
    }
  } else {
    firOpBuilder.createBlock(&op.getRegion());
  }
  // Set the insert for the terminator operation to go at the end of the
  // block - this is either empty or the block with the stores above,
  // the end of the block works for both.
  mlir::Block &block = op.getRegion().back();
  firOpBuilder.setInsertionPointToEnd(&block);

  // If it is an unstructured region and is not the outer region of a combined
  // construct, create empty blocks for all evaluations.
  if (eval.lowerAsUnstructured() && !outerCombined)
    createEmptyRegionBlocks(firOpBuilder, eval.getNestedEvaluations());

  // Insert the terminator.
  if constexpr (std::is_same_v<Op, omp::WsLoopOp> ||
                std::is_same_v<Op, omp::SimdLoopOp>) {
    mlir::ValueRange results;
    firOpBuilder.create<mlir::omp::YieldOp>(loc, results);
  } else {
    firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
  }
  // Reset the insert point to before the terminator.
  resetBeforeTerminator(firOpBuilder, storeOp, block);

  // Handle privatization. Do not privatize if this is the outer operation.
  if (clauses && !outerCombined) {
    constexpr bool is_loop = std::is_same_v<Op, omp::WsLoopOp> ||
                             std::is_same_v<Op, omp::SimdLoopOp>;
    if (!dsp) {
      DataSharingProcessor proc(converter, *clauses, eval);
      proc.processStep1();
      proc.processStep2(op, is_loop);
    } else {
      dsp->processStep2(op, is_loop);
    }

    if (storeOp)
      firOpBuilder.setInsertionPointAfter(storeOp);
  }

  if constexpr (std::is_same_v<Op, omp::ParallelOp>) {
    threadPrivatizeVars(converter, eval);
    if (clauses)
      genCopyinClause(converter, *clauses);
  }
}

static void createBodyOfTargetOp(
    Fortran::lower::AbstractConverter &converter, mlir::omp::DataOp &dataOp,
    const llvm::SmallVector<mlir::Type> &useDeviceTypes,
    const llvm::SmallVector<mlir::Location> &useDeviceLocs,
    const SmallVector<const Fortran::semantics::Symbol *> &useDeviceSymbols,
    const mlir::Location &currentLocation) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Region &region = dataOp.getRegion();

  firOpBuilder.createBlock(&region, {}, useDeviceTypes, useDeviceLocs);
  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);
  firOpBuilder.setInsertionPointToStart(&region.front());

  unsigned argIndex = 0;
  for (auto *sym : useDeviceSymbols) {
    const mlir::BlockArgument &arg = region.front().getArgument(argIndex);
    mlir::Value val = fir::getBase(arg);
    fir::ExtendedValue extVal = converter.getSymbolExtendedValue(*sym);
    if (auto refType = val.getType().dyn_cast<fir::ReferenceType>()) {
      if (fir::isa_builtin_cptr_type(refType.getElementType())) {
        converter.bindSymbol(*sym, val);
      } else {
        extVal.match(
            [&](const fir::MutableBoxValue &mbv) {
              converter.bindSymbol(
                  *sym,
                  fir::MutableBoxValue(
                      val, fir::factory::getNonDeferredLenParams(extVal), {}));
            },
            [&](const auto &) {
              TODO(converter.getCurrentLocation(),
                   "use_device clause operand unsupported type");
            });
      }
    } else {
      TODO(converter.getCurrentLocation(),
           "use_device clause operand unsupported type");
    }
    argIndex++;
  }
}

static void createTargetOp(Fortran::lower::AbstractConverter &converter,
                           const Fortran::parser::OmpClauseList &opClauseList,
                           const llvm::omp::Directive &directive,
                           mlir::Location currentLocation,
                           Fortran::lower::pft::Evaluation *eval = nullptr) {
  Fortran::lower::StatementContext stmtCtx;
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  mlir::Value ifClauseOperand, deviceOperand, threadLmtOperand;
  mlir::UnitAttr nowaitAttr;
  llvm::SmallVector<mlir::Value> mapOperands, devicePtrOperands,
      deviceAddrOperands;
  llvm::SmallVector<mlir::IntegerAttr> mapTypes;
  llvm::SmallVector<mlir::Type> useDeviceTypes;
  llvm::SmallVector<mlir::Location> useDeviceLocs;
  SmallVector<const Fortran::semantics::Symbol *> useDeviceSymbols;

  /// Check for unsupported map operand types.
  auto checkType = [](mlir::Location location, mlir::Type type) {
    if (auto refType = type.dyn_cast<fir::ReferenceType>())
      type = refType.getElementType();
    if (auto boxType = type.dyn_cast_or_null<fir::BoxType>())
      if (!boxType.getElementType().isa<fir::PointerType>())
        TODO(location, "OMPD_target_data MapOperand BoxType");
  };

  auto addMapClause = [&](const auto &mapClause, mlir::Location &location) {
    const auto &oMapType =
        std::get<std::optional<Fortran::parser::OmpMapType>>(mapClause->v.t);
    llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE;
    // If the map type is specified, then process it else Tofrom is the default.
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
        // alloc and release is the default map_type for the Target Data Ops,
        // i.e. if no bits for map_type is supplied then alloc/release is
        // implicitly assumed based on the target directive. Default value for
        // Target Data and Enter Data is alloc and for Exit Data it is release.
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

    // TODO: Add support MapTypeModifiers close, mapper, present, iterator

    mlir::IntegerAttr mapTypeAttr = firOpBuilder.getIntegerAttr(
        firOpBuilder.getI64Type(),
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            mapTypeBits));

    llvm::SmallVector<mlir::Value> mapOperand;
    /// Check for unsupported map operand types.
    for (const Fortran::parser::OmpObject &ompObject :
         std::get<Fortran::parser::OmpObjectList>(mapClause->v.t).v) {
      if (Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(ompObject) ||
          Fortran::parser::Unwrap<Fortran::parser::StructureComponent>(
              ompObject))
        TODO(location,
             "OMPD_target_data for Array Expressions or Structure Components");
    }
    genObjectList(std::get<Fortran::parser::OmpObjectList>(mapClause->v.t),
                  converter, mapOperand);

    for (mlir::Value mapOp : mapOperand) {
      checkType(mapOp.getLoc(), mapOp.getType());
      mapOperands.push_back(mapOp);
      mapTypes.push_back(mapTypeAttr);
    }
  };

  auto addUseDeviceClause = [&](const auto &useDeviceClause, auto &operands) {
    genObjectList(useDeviceClause, converter, operands);
    for (auto &operand : operands) {
      checkType(operand.getLoc(), operand.getType());
      useDeviceTypes.push_back(operand.getType());
      useDeviceLocs.push_back(operand.getLoc());
    }
    for (const Fortran::parser::OmpObject &ompObject : useDeviceClause.v) {
      Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
      useDeviceSymbols.push_back(sym);
    }
  };

  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto &ifClause =
            std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      ifClauseOperand =
          getIfClauseOperand(converter, stmtCtx, ifClause, clauseLocation);
    } else if (const auto &deviceClause =
                   std::get_if<Fortran::parser::OmpClause::Device>(&clause.u)) {
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
        deviceOperand =
            fir::getBase(converter.genExprValue(*deviceExpr, stmtCtx));
      }
    } else if (const auto &threadLmtClause =
                   std::get_if<Fortran::parser::OmpClause::ThreadLimit>(
                       &clause.u)) {
      threadLmtOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(threadLmtClause->v), stmtCtx));
    } else if (std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u)) {
      nowaitAttr = firOpBuilder.getUnitAttr();
    } else if (const auto &devPtrClause =
                   std::get_if<Fortran::parser::OmpClause::UseDevicePtr>(
                       &clause.u)) {
      addUseDeviceClause(devPtrClause->v, devicePtrOperands);
    } else if (const auto &devAddrClause =
                   std::get_if<Fortran::parser::OmpClause::UseDeviceAddr>(
                       &clause.u)) {
      addUseDeviceClause(devAddrClause->v, deviceAddrOperands);
    } else if (const auto &mapClause =
                   std::get_if<Fortran::parser::OmpClause::Map>(&clause.u)) {
      addMapClause(mapClause, clauseLocation);
    } else {
      TODO(clauseLocation, "OMPD_target unhandled clause");
    }
  }

  llvm::SmallVector<mlir::Attribute> mapTypesAttr(mapTypes.begin(),
                                                  mapTypes.end());
  mlir::ArrayAttr mapTypesArrayAttr =
      ArrayAttr::get(firOpBuilder.getContext(), mapTypesAttr);

  if (directive == llvm::omp::Directive::OMPD_target) {
    auto targetOp = firOpBuilder.create<omp::TargetOp>(
        currentLocation, ifClauseOperand, deviceOperand, threadLmtOperand,
        nowaitAttr, mapOperands, mapTypesArrayAttr);
    createBodyOfOp(targetOp, converter, currentLocation, *eval, &opClauseList);
  } else if (directive == llvm::omp::Directive::OMPD_target_data) {
    auto dataOp = firOpBuilder.create<omp::DataOp>(
        currentLocation, ifClauseOperand, deviceOperand, devicePtrOperands,
        deviceAddrOperands, mapOperands, mapTypesArrayAttr);
    createBodyOfTargetOp(converter, dataOp, useDeviceTypes, useDeviceLocs,
                         useDeviceSymbols, currentLocation);
  } else if (directive == llvm::omp::Directive::OMPD_target_enter_data) {
    firOpBuilder.create<omp::EnterDataOp>(currentLocation, ifClauseOperand,
                                          deviceOperand, nowaitAttr,
                                          mapOperands, mapTypesArrayAttr);
  } else if (directive == llvm::omp::Directive::OMPD_target_exit_data) {
    firOpBuilder.create<omp::ExitDataOp>(currentLocation, ifClauseOperand,
                                         deviceOperand, nowaitAttr, mapOperands,
                                         mapTypesArrayAttr);
  } else {
    TODO(currentLocation, "OMPD_target directive unknown");
  }
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPSimpleStandaloneConstruct
                       &simpleStandaloneConstruct) {
  const auto &directive =
      std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
          simpleStandaloneConstruct.t);
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  const Fortran::parser::OmpClauseList &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(simpleStandaloneConstruct.t);
  mlir::Location currentLocation = converter.genLocation(directive.source);

  switch (directive.v) {
  default:
    break;
  case llvm::omp::Directive::OMPD_barrier:
    firOpBuilder.create<omp::BarrierOp>(currentLocation);
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    firOpBuilder.create<omp::TaskwaitOp>(currentLocation);
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    firOpBuilder.create<omp::TaskyieldOp>(currentLocation);
    break;
  case llvm::omp::Directive::OMPD_target_data:
  case llvm::omp::Directive::OMPD_target_enter_data:
  case llvm::omp::Directive::OMPD_target_exit_data:
    createTargetOp(converter, opClauseList, directive.v, currentLocation);
    break;
  case llvm::omp::Directive::OMPD_target_update:
    TODO(currentLocation, "OMPD_target_update");
  case llvm::omp::Directive::OMPD_ordered:
    TODO(currentLocation, "OMPD_ordered");
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
    TODO(converter.getCurrentLocation(), "OmpAllocateClause ALIGN modifier");
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

static omp::ClauseProcBindKindAttr genProcBindKindAttr(
    fir::FirOpBuilder &firOpBuilder,
    const Fortran::parser::OmpClause::ProcBind *procBindClause) {
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
  return omp::ClauseProcBindKindAttr::get(firOpBuilder.getContext(), pbKind);
}

static omp::ClauseTaskDependAttr
genDependKindAttr(fir::FirOpBuilder &firOpBuilder,
                  const Fortran::parser::OmpClause::Depend *dependClause) {
  omp::ClauseTaskDepend pbKind;
  switch (
      std::get<Fortran::parser::OmpDependenceType>(
          std::get<Fortran::parser::OmpDependClause::InOut>(dependClause->v.u)
              .t)
          .v) {
  case Fortran::parser::OmpDependenceType::Type::In:
    pbKind = omp::ClauseTaskDepend::taskdependin;
    break;
  case Fortran::parser::OmpDependenceType::Type::Out:
    pbKind = omp::ClauseTaskDepend::taskdependout;
    break;
  case Fortran::parser::OmpDependenceType::Type::Inout:
    pbKind = omp::ClauseTaskDepend::taskdependinout;
    break;
  default:
    llvm_unreachable("unknown parser task dependence type");
    break;
  }
  return omp::ClauseTaskDependAttr::get(firOpBuilder.getContext(), pbKind);
}

/* When parallel is used in a combined construct, then use this function to
 * create the parallel operation. It handles the parallel specific clauses
 * and leaves the rest for handling at the inner operations.
 * TODO: Refactor clause handling
 */
template <typename Directive>
static void
createCombinedParallelOp(Fortran::lower::AbstractConverter &converter,
                         Fortran::lower::pft::Evaluation &eval,
                         const Directive &directive) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  Fortran::lower::StatementContext stmtCtx;
  llvm::ArrayRef<mlir::Type> argTy;
  mlir::Value ifClauseOperand, numThreadsClauseOperand;
  SmallVector<Value> allocatorOperands, allocateOperands;
  mlir::omp::ClauseProcBindKindAttr procBindKindAttr;
  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(directive.t);
  // TODO: Handle the following clauses
  // 1. default
  // Note: rest of the clauses are handled when the inner operation is created
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto &ifClause =
            std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      ifClauseOperand =
          getIfClauseOperand(converter, stmtCtx, ifClause, clauseLocation);
    } else if (const auto &numThreadsClause =
                   std::get_if<Fortran::parser::OmpClause::NumThreads>(
                       &clause.u)) {
      numThreadsClauseOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numThreadsClause->v), stmtCtx));
    } else if (const auto &procBindClause =
                   std::get_if<Fortran::parser::OmpClause::ProcBind>(
                       &clause.u)) {
      procBindKindAttr = genProcBindKindAttr(firOpBuilder, procBindClause);
    }
  }
  // Create and insert the operation.
  auto parallelOp = firOpBuilder.create<mlir::omp::ParallelOp>(
      currentLocation, argTy, ifClauseOperand, numThreadsClauseOperand,
      allocateOperands, allocatorOperands, /*reduction_vars=*/ValueRange(),
      /*reductions=*/nullptr, procBindKindAttr);

  createBodyOfOp<omp::ParallelOp>(parallelOp, converter, currentLocation, eval,
                                  &opClauseList, /*iv=*/{},
                                  /*isCombined=*/true);
}

/// This function returns the identity value of the operator \p reductionOpName.
/// For example:
///    0 + x = x,
///    1 * x = x
static int getOperationIdentity(llvm::StringRef reductionOpName,
                                mlir::Location loc) {
  if (reductionOpName.contains("add") || reductionOpName.contains("or") ||
      reductionOpName.contains("neqv"))
    return 0;
  if (reductionOpName.contains("multiply") || reductionOpName.contains("and") ||
      reductionOpName.contains("eqv"))
    return 1;
  TODO(loc, "Reduction of some intrinsic operators is not supported");
}

static Value getReductionInitValue(mlir::Location loc, mlir::Type type,
                                   llvm::StringRef reductionOpName,
                                   fir::FirOpBuilder &builder) {
  assert((fir::isa_integer(type) || fir::isa_real(type) ||
          type.isa<fir::LogicalType>()) &&
         "only integer, logical and real types are currently supported");
  if (reductionOpName.contains("max")) {
    if (auto ty = type.dyn_cast<mlir::FloatType>()) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, type, llvm::APFloat::getLargest(sem, /*Negative=*/true));
    }
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t minInt = llvm::APInt::getSignedMinValue(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, minInt);
  } else if (reductionOpName.contains("min")) {
    if (auto ty = type.dyn_cast<mlir::FloatType>()) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, type, llvm::APFloat::getSmallest(sem, /*Negative=*/true));
    }
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t maxInt = llvm::APInt::getSignedMaxValue(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, maxInt);
  } else if (reductionOpName.contains("ior")) {
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t zeroInt = llvm::APInt::getZero(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, zeroInt);
  } else if (reductionOpName.contains("ieor")) {
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t zeroInt = llvm::APInt::getZero(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, zeroInt);
  } else if (reductionOpName.contains("iand")) {
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t allOnInt = llvm::APInt::getAllOnes(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, allOnInt);
  } else {
    if (type.isa<FloatType>())
      return builder.create<mlir::arith::ConstantOp>(
          loc, type,
          builder.getFloatAttr(
              type, (double)getOperationIdentity(reductionOpName, loc)));

    if (type.isa<fir::LogicalType>()) {
      Value intConst = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI1Type(),
          builder.getIntegerAttr(builder.getI1Type(),
                                 getOperationIdentity(reductionOpName, loc)));
      return builder.createConvert(loc, type, intConst);
    }

    return builder.create<mlir::arith::ConstantOp>(
        loc, type,
        builder.getIntegerAttr(type,
                               getOperationIdentity(reductionOpName, loc)));
  }
}

template <typename FloatOp, typename IntegerOp>
static Value getReductionOperation(fir::FirOpBuilder &builder, mlir::Type type,
                                   mlir::Location loc, mlir::Value op1,
                                   mlir::Value op2) {
  assert(type.isIntOrIndexOrFloat() &&
         "only integer and float types are currently supported");
  if (type.isIntOrIndex())
    return builder.create<IntegerOp>(loc, op1, op2);
  return builder.create<FloatOp>(loc, op1, op2);
}

static omp::ReductionDeclareOp
createMinimalReductionDecl(fir::FirOpBuilder &builder,
                           llvm::StringRef reductionOpName, mlir::Type type,
                           mlir::Location loc) {
  mlir::ModuleOp module = builder.getModule();
  mlir::OpBuilder modBuilder(module.getBodyRegion());

  mlir::omp::ReductionDeclareOp decl =
      modBuilder.create<omp::ReductionDeclareOp>(loc, reductionOpName, type);
  builder.createBlock(&decl.getInitializerRegion(),
                      decl.getInitializerRegion().end(), {type}, {loc});
  builder.setInsertionPointToEnd(&decl.getInitializerRegion().back());
  Value init = getReductionInitValue(loc, type, reductionOpName, builder);
  builder.create<omp::YieldOp>(loc, init);

  builder.createBlock(&decl.getReductionRegion(),
                      decl.getReductionRegion().end(), {type, type},
                      {loc, loc});

  return decl;
}

/// Creates an OpenMP reduction declaration and inserts it into the provided
/// symbol table. The declaration has a constant initializer with the neutral
/// value `initValue`, and the reduction combiner carried over from `reduce`.
/// TODO: Generalize this for non-integer types, add atomic region.
static omp::ReductionDeclareOp
createReductionDecl(fir::FirOpBuilder &builder, llvm::StringRef reductionOpName,
                    const Fortran::parser::ProcedureDesignator &procDesignator,
                    mlir::Type type, mlir::Location loc) {
  OpBuilder::InsertionGuard guard(builder);
  mlir::ModuleOp module = builder.getModule();

  auto decl =
      module.lookupSymbol<mlir::omp::ReductionDeclareOp>(reductionOpName);
  if (decl)
    return decl;

  decl = createMinimalReductionDecl(builder, reductionOpName, type, loc);
  builder.setInsertionPointToEnd(&decl.getReductionRegion().back());
  mlir::Value op1 = decl.getReductionRegion().front().getArgument(0);
  mlir::Value op2 = decl.getReductionRegion().front().getArgument(1);

  Value reductionOp;
  if (const auto *name{
          Fortran::parser::Unwrap<Fortran::parser::Name>(procDesignator)}) {
    if (name->source == "max") {
      reductionOp =
          getReductionOperation<mlir::arith::MaxFOp, mlir::arith::MaxSIOp>(
              builder, type, loc, op1, op2);
    } else if (name->source == "min") {
      reductionOp =
          getReductionOperation<mlir::arith::MinFOp, mlir::arith::MinSIOp>(
              builder, type, loc, op1, op2);
    } else if (name->source == "ior") {
      assert((type.isIntOrIndex()) && "only integer is expected");
      reductionOp = builder.create<mlir::arith::OrIOp>(loc, op1, op2);
    } else if (name->source == "ieor") {
      assert((type.isIntOrIndex()) && "only integer is expected");
      reductionOp = builder.create<mlir::arith::XOrIOp>(loc, op1, op2);
    } else if (name->source == "iand") {
      assert((type.isIntOrIndex()) && "only integer is expected");
      reductionOp = builder.create<mlir::arith::AndIOp>(loc, op1, op2);
    } else {
      TODO(loc, "Reduction of some intrinsic operators is not supported");
    }
  }

  builder.create<omp::YieldOp>(loc, reductionOp);
  return decl;
}

/// Creates an OpenMP reduction declaration and inserts it into the provided
/// symbol table. The declaration has a constant initializer with the neutral
/// value `initValue`, and the reduction combiner carried over from `reduce`.
/// TODO: Generalize this for non-integer types, add atomic region.
static omp::ReductionDeclareOp createReductionDecl(
    fir::FirOpBuilder &builder, llvm::StringRef reductionOpName,
    Fortran::parser::DefinedOperator::IntrinsicOperator intrinsicOp,
    mlir::Type type, mlir::Location loc) {
  OpBuilder::InsertionGuard guard(builder);
  mlir::ModuleOp module = builder.getModule();

  auto decl =
      module.lookupSymbol<mlir::omp::ReductionDeclareOp>(reductionOpName);
  if (decl)
    return decl;

  decl = createMinimalReductionDecl(builder, reductionOpName, type, loc);
  builder.setInsertionPointToEnd(&decl.getReductionRegion().back());
  mlir::Value op1 = decl.getReductionRegion().front().getArgument(0);
  mlir::Value op2 = decl.getReductionRegion().front().getArgument(1);

  Value reductionOp;
  switch (intrinsicOp) {
  case Fortran::parser::DefinedOperator::IntrinsicOperator::Add:
    reductionOp =
        getReductionOperation<mlir::arith::AddFOp, mlir::arith::AddIOp>(
            builder, type, loc, op1, op2);
    break;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::Multiply:
    reductionOp =
        getReductionOperation<mlir::arith::MulFOp, mlir::arith::MulIOp>(
            builder, type, loc, op1, op2);
    break;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::AND: {
    Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    Value andiOp = builder.create<mlir::arith::AndIOp>(loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, andiOp);
    break;
  }
  case Fortran::parser::DefinedOperator::IntrinsicOperator::OR: {
    Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    Value oriOp = builder.create<mlir::arith::OrIOp>(loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, oriOp);
    break;
  }
  case Fortran::parser::DefinedOperator::IntrinsicOperator::EQV: {
    Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    Value cmpiOp = builder.create<mlir::arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  case Fortran::parser::DefinedOperator::IntrinsicOperator::NEQV: {
    Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    Value cmpiOp = builder.create<mlir::arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  default:
    TODO(loc, "Reduction of some intrinsic operators is not supported");
  }

  builder.create<omp::YieldOp>(loc, reductionOp);
  return decl;
}

static mlir::omp::ScheduleModifier
translateModifier(const Fortran::parser::OmpScheduleModifierType &m) {
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
        return translateModifier(modType2->v);

      return mlir::omp::ScheduleModifier::none;
    }

    return translateModifier(modType1.v);
  }
  return mlir::omp::ScheduleModifier::none;
}

static mlir::omp::ScheduleModifier
getSIMDModifier(const Fortran::parser::OmpScheduleClause &x) {
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

static std::string getReductionName(llvm::StringRef name, mlir::Type ty) {
  return (llvm::Twine(name) +
          (ty.isIntOrIndex() ? llvm::Twine("_i_") : llvm::Twine("_f_")) +
          llvm::Twine(ty.getIntOrFloatBitWidth()))
      .str();
}

static std::string getReductionName(
    Fortran::parser::DefinedOperator::IntrinsicOperator intrinsicOp,
    mlir::Type ty) {
  std::string reductionName;

  switch (intrinsicOp) {
  case Fortran::parser::DefinedOperator::IntrinsicOperator::Add:
    reductionName = "add_reduction";
    break;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::Multiply:
    reductionName = "multiply_reduction";
    break;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::AND:
    return "and_reduction";
  case Fortran::parser::DefinedOperator::IntrinsicOperator::EQV:
    return "eqv_reduction";
  case Fortran::parser::DefinedOperator::IntrinsicOperator::OR:
    return "or_reduction";
  case Fortran::parser::DefinedOperator::IntrinsicOperator::NEQV:
    return "neqv_reduction";
  default:
    reductionName = "other_reduction";
    break;
  }

  return getReductionName(reductionName, ty);
}

/// Creates a reduction declaration and associates it with an
/// OpenMP block directive
static void
addReductionDecl(mlir::Location currentLocation,
                 Fortran::lower::AbstractConverter &converter,
                 const Fortran::parser::OmpReductionClause &reduction,
                 SmallVector<Value> &reductionVars,
                 SmallVector<Attribute> &reductionDeclSymbols) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  omp::ReductionDeclareOp decl;
  const auto &redOperator{
      std::get<Fortran::parser::OmpReductionOperator>(reduction.t)};
  const auto &objectList{std::get<Fortran::parser::OmpObjectList>(reduction.t)};
  if (const auto &redDefinedOp =
          std::get_if<Fortran::parser::DefinedOperator>(&redOperator.u)) {
    const auto &intrinsicOp{
        std::get<Fortran::parser::DefinedOperator::IntrinsicOperator>(
            redDefinedOp->u)};
    switch (intrinsicOp) {
    case Fortran::parser::DefinedOperator::IntrinsicOperator::Add:
    case Fortran::parser::DefinedOperator::IntrinsicOperator::Multiply:
    case Fortran::parser::DefinedOperator::IntrinsicOperator::AND:
    case Fortran::parser::DefinedOperator::IntrinsicOperator::EQV:
    case Fortran::parser::DefinedOperator::IntrinsicOperator::OR:
    case Fortran::parser::DefinedOperator::IntrinsicOperator::NEQV:
      break;

    default:
      TODO(currentLocation,
           "Reduction of some intrinsic operators is not supported");
      break;
    }
    for (const auto &ompObject : objectList.v) {
      if (const auto *name{
              Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
        if (const auto *symbol{name->symbol}) {
          mlir::Value symVal = converter.getSymbolAddress(*symbol);
          mlir::Type redType =
              symVal.getType().cast<fir::ReferenceType>().getEleTy();
          reductionVars.push_back(symVal);
          if (redType.isa<fir::LogicalType>())
            decl = createReductionDecl(
                firOpBuilder,
                getReductionName(intrinsicOp, firOpBuilder.getI1Type()),
                intrinsicOp, redType, currentLocation);
          else if (redType.isIntOrIndexOrFloat()) {
            decl = createReductionDecl(firOpBuilder,
                                       getReductionName(intrinsicOp, redType),
                                       intrinsicOp, redType, currentLocation);
          } else {
            TODO(currentLocation, "Reduction of some types is not supported");
          }
          reductionDeclSymbols.push_back(
              SymbolRefAttr::get(firOpBuilder.getContext(), decl.getSymName()));
        }
      }
    }
  } else if (auto reductionIntrinsic =
                 std::get_if<Fortran::parser::ProcedureDesignator>(
                     &redOperator.u)) {
    if (const auto *name{Fortran::parser::Unwrap<Fortran::parser::Name>(
            reductionIntrinsic)}) {
      if ((name->source != "max") && (name->source != "min") &&
          (name->source != "ior") && (name->source != "ieor") &&
          (name->source != "iand")) {
        TODO(currentLocation,
             "Reduction of intrinsic procedures is not supported");
      }
      std::string intrinsicOp = name->ToString();
      for (const auto &ompObject : objectList.v) {
        if (const auto *name{
                Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
          if (const auto *symbol{name->symbol}) {
            mlir::Value symVal = converter.getSymbolAddress(*symbol);
            mlir::Type redType =
                symVal.getType().cast<fir::ReferenceType>().getEleTy();
            reductionVars.push_back(symVal);
            assert(redType.isIntOrIndexOrFloat() &&
                   "Unsupported reduction type");
            decl = createReductionDecl(
                firOpBuilder, getReductionName(intrinsicOp, redType),
                *reductionIntrinsic, redType, currentLocation);
            reductionDeclSymbols.push_back(SymbolRefAttr::get(
                firOpBuilder.getContext(), decl.getSymName()));
          }
        }
      }
    }
  }
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Value> lowerBound, upperBound, step, linearVars,
      linearStepVars, reductionVars, alignedVars, nontemporalVars;
  mlir::Value scheduleChunkClauseOperand, ifClauseOperand;
  mlir::Attribute scheduleClauseOperand, noWaitClauseOperand,
      orderedClauseOperand, orderClauseOperand;
  mlir::IntegerAttr simdlenClauseOperand, safelenClauseOperand;
  SmallVector<Attribute> reductionDeclSymbols;
  Fortran::lower::StatementContext stmtCtx;
  const auto &loopOpClauseList = std::get<Fortran::parser::OmpClauseList>(
      std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t).t);

  const auto &beginLoopDirective =
      std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t);
  mlir::Location currentLocation =
      converter.genLocation(beginLoopDirective.source);
  const auto ompDirective =
      std::get<Fortran::parser::OmpLoopDirective>(beginLoopDirective.t).v;

  if (llvm::omp::OMPD_parallel_do == ompDirective) {
    createCombinedParallelOp<Fortran::parser::OmpBeginLoopDirective>(
        converter, eval,
        std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t));
  } else if (llvm::omp::OMPD_do != ompDirective &&
             llvm::omp::OMPD_simd != ompDirective) {
    TODO(currentLocation, "Construct enclosing do loop");
  }

  DataSharingProcessor dsp(converter, loopOpClauseList, eval);
  dsp.processStep1();

  // Collect the loops to collapse.
  auto *doConstructEval = &eval.getFirstNestedEvaluation();
  if (doConstructEval->getIf<Fortran::parser::DoConstruct>()
          ->IsDoConcurrent()) {
    TODO(currentLocation, "Do Concurrent in Worksharing loop construct");
  }

  std::int64_t collapseValue =
      Fortran::lower::getCollapseValue(loopOpClauseList);
  std::size_t loopVarTypeSize = 0;
  SmallVector<const Fortran::semantics::Symbol *> iv;
  do {
    auto *doLoop = &doConstructEval->getFirstNestedEvaluation();
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

  for (const auto &clause : loopOpClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto &scheduleClause =
            std::get_if<Fortran::parser::OmpClause::Schedule>(&clause.u)) {
      if (const auto &chunkExpr =
              std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                  scheduleClause->v.t)) {
        if (const auto *expr = Fortran::semantics::GetExpr(*chunkExpr)) {
          scheduleChunkClauseOperand =
              fir::getBase(converter.genExprValue(*expr, stmtCtx));
        }
      }
    } else if (const auto &ifClause =
                   std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      ifClauseOperand =
          getIfClauseOperand(converter, stmtCtx, ifClause, clauseLocation);
    } else if (const auto &reductionClause =
                   std::get_if<Fortran::parser::OmpClause::Reduction>(
                       &clause.u)) {
      addReductionDecl(currentLocation, converter, reductionClause->v,
                       reductionVars, reductionDeclSymbols);
    } else if (const auto &simdlenClause =
                   std::get_if<Fortran::parser::OmpClause::Simdlen>(
                       &clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(simdlenClause->v);
      const std::optional<std::int64_t> simdlenVal =
          Fortran::evaluate::ToInt64(*expr);
      simdlenClauseOperand = firOpBuilder.getI64IntegerAttr(*simdlenVal);
    } else if (const auto &safelenClause =
                   std::get_if<Fortran::parser::OmpClause::Safelen>(
                       &clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(safelenClause->v);
      const std::optional<std::int64_t> safelenVal =
          Fortran::evaluate::ToInt64(*expr);
      safelenClauseOperand = firOpBuilder.getI64IntegerAttr(*safelenVal);
    }
  }

  // The types of lower bound, upper bound, and step are converted into the
  // type of the loop variable if necessary.
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  for (unsigned it = 0; it < (unsigned)lowerBound.size(); it++) {
    lowerBound[it] = firOpBuilder.createConvert(currentLocation, loopVarType,
                                                lowerBound[it]);
    upperBound[it] = firOpBuilder.createConvert(currentLocation, loopVarType,
                                                upperBound[it]);
    step[it] =
        firOpBuilder.createConvert(currentLocation, loopVarType, step[it]);
  }

  // 2.9.3.1 SIMD construct
  // TODO: Support all the clauses
  if (llvm::omp::OMPD_simd == ompDirective) {
    TypeRange resultType;
    auto simdLoopOp = firOpBuilder.create<mlir::omp::SimdLoopOp>(
        currentLocation, resultType, lowerBound, upperBound, step, alignedVars,
        nullptr, ifClauseOperand, nontemporalVars,
        orderClauseOperand.dyn_cast_or_null<omp::ClauseOrderKindAttr>(),
        simdlenClauseOperand, safelenClauseOperand,
        /*inclusive=*/firOpBuilder.getUnitAttr());
    createBodyOfOp<omp::SimdLoopOp>(simdLoopOp, converter, currentLocation,
                                    eval, &loopOpClauseList, iv,
                                    /*outer=*/false, &dsp);
    return;
  }

  // FIXME: Add support for following clauses:
  // 1. linear
  // 2. order
  auto wsLoopOp = firOpBuilder.create<mlir::omp::WsLoopOp>(
      currentLocation, lowerBound, upperBound, step, linearVars, linearStepVars,
      reductionVars,
      reductionDeclSymbols.empty()
          ? nullptr
          : mlir::ArrayAttr::get(firOpBuilder.getContext(),
                                 reductionDeclSymbols),
      scheduleClauseOperand.dyn_cast_or_null<omp::ClauseScheduleKindAttr>(),
      scheduleChunkClauseOperand, /*schedule_modifiers=*/nullptr,
      /*simd_modifier=*/nullptr,
      noWaitClauseOperand.dyn_cast_or_null<UnitAttr>(),
      orderedClauseOperand.dyn_cast_or_null<IntegerAttr>(),
      orderClauseOperand.dyn_cast_or_null<omp::ClauseOrderKindAttr>(),
      /*inclusive=*/firOpBuilder.getUnitAttr());

  // Handle attribute based clauses.
  for (const Fortran::parser::OmpClause &clause : loopOpClauseList.v) {
    if (const auto &orderedClause =
            std::get_if<Fortran::parser::OmpClause::Ordered>(&clause.u)) {
      if (orderedClause->v.has_value()) {
        const auto *expr = Fortran::semantics::GetExpr(orderedClause->v);
        const std::optional<std::int64_t> orderedClauseValue =
            Fortran::evaluate::ToInt64(*expr);
        wsLoopOp.setOrderedValAttr(
            firOpBuilder.getI64IntegerAttr(*orderedClauseValue));
      } else {
        wsLoopOp.setOrderedValAttr(firOpBuilder.getI64IntegerAttr(0));
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
        wsLoopOp.setScheduleValAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Static));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Dynamic:
        wsLoopOp.setScheduleValAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Dynamic));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Guided:
        wsLoopOp.setScheduleValAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Guided));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Auto:
        wsLoopOp.setScheduleValAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Auto));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Runtime:
        wsLoopOp.setScheduleValAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Runtime));
        break;
      }
      mlir::omp::ScheduleModifier scheduleModifier =
          getScheduleModifier(scheduleClause->v);
      if (scheduleModifier != mlir::omp::ScheduleModifier::none)
        wsLoopOp.setScheduleModifierAttr(
            omp::ScheduleModifierAttr::get(context, scheduleModifier));
      if (getSIMDModifier(scheduleClause->v) !=
          mlir::omp::ScheduleModifier::none)
        wsLoopOp.setSimdModifierAttr(firOpBuilder.getUnitAttr());
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
        wsLoopOp.setNowaitAttr(firOpBuilder.getUnitAttr());
  }

  createBodyOfOp<omp::WsLoopOp>(wsLoopOp, converter, currentLocation, eval,
                                &loopOpClauseList, iv, /*outer=*/false, &dsp);
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
  mlir::Location currentLocation = converter.genLocation(blockDirective.source);

  Fortran::lower::StatementContext stmtCtx;
  llvm::ArrayRef<mlir::Type> argTy;
  mlir::Value ifClauseOperand, numThreadsClauseOperand, finalClauseOperand,
      priorityClauseOperand;
  mlir::omp::ClauseProcBindKindAttr procBindKindAttr;
  SmallVector<Value> allocateOperands, allocatorOperands, dependOperands,
      reductionVars;
  SmallVector<Attribute> dependTypeOperands, reductionDeclSymbols;
  mlir::UnitAttr nowaitAttr, untiedAttr, mergeableAttr;

  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginBlockDirective.t);
  for (const auto &clause : opClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto &ifClause =
            std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      ifClauseOperand =
          getIfClauseOperand(converter, stmtCtx, ifClause, clauseLocation);
    } else if (const auto &numThreadsClause =
                   std::get_if<Fortran::parser::OmpClause::NumThreads>(
                       &clause.u)) {
      // OMPIRBuilder expects `NUM_THREAD` clause as a `Value`.
      numThreadsClauseOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numThreadsClause->v), stmtCtx));
    } else if (const auto &procBindClause =
                   std::get_if<Fortran::parser::OmpClause::ProcBind>(
                       &clause.u)) {
      procBindKindAttr = genProcBindKindAttr(firOpBuilder, procBindClause);
    } else if (const auto &allocateClause =
                   std::get_if<Fortran::parser::OmpClause::Allocate>(
                       &clause.u)) {
      genAllocateClause(converter, allocateClause->v, allocatorOperands,
                        allocateOperands);
    } else if (std::get_if<Fortran::parser::OmpClause::Private>(&clause.u) ||
               std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                   &clause.u) ||
               std::get_if<Fortran::parser::OmpClause::Copyin>(&clause.u)) {
      // Privatisation and copyin clauses are handled elsewhere.
      continue;
    } else if (std::get_if<Fortran::parser::OmpClause::Shared>(&clause.u)) {
      // Shared is the default behavior in the IR, so no handling is required.
      continue;
    } else if (const auto &defaultClause =
                   std::get_if<Fortran::parser::OmpClause::Default>(
                       &clause.u)) {
      if ((defaultClause->v.v ==
           Fortran::parser::OmpDefaultClause::Type::Shared) ||
          (defaultClause->v.v ==
           Fortran::parser::OmpDefaultClause::Type::None)) {
        // Default clause with shared or none do not require any handling since
        // Shared is the default behavior in the IR and None is only required
        // for semantic checks.
        continue;
      }
    } else if (std::get_if<Fortran::parser::OmpClause::Threads>(&clause.u)) {
      // Nothing needs to be done for threads clause.
      continue;
    } else if (std::get_if<Fortran::parser::OmpClause::Map>(&clause.u)) {
      // Map clause is exclusive to Target Data directives. It is handled
      // as part of the TargetOp creation.
      continue;
    } else if (std::get_if<Fortran::parser::OmpClause::UseDevicePtr>(
                   &clause.u)) {
      // UseDevicePtr clause is exclusive to Target Data directives. It is
      // handled as part of the TargetOp creation.
      continue;
    } else if (std::get_if<Fortran::parser::OmpClause::UseDeviceAddr>(
                   &clause.u)) {
      // UseDeviceAddr clause is exclusive to Target Data directives. It is
      // handled as part of the TargetOp creation.
      continue;
    } else if (std::get_if<Fortran::parser::OmpClause::ThreadLimit>(
                   &clause.u)) {
      // Handled as part of TargetOp creation.
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
    } else if (const auto &reductionClause =
                   std::get_if<Fortran::parser::OmpClause::Reduction>(
                       &clause.u)) {
      addReductionDecl(currentLocation, converter, reductionClause->v,
                       reductionVars, reductionDeclSymbols);
    } else if (const auto &dependClause =
                   std::get_if<Fortran::parser::OmpClause::Depend>(&clause.u)) {
      const std::list<Fortran::parser::Designator> &depVal =
          std::get<std::list<Fortran::parser::Designator>>(
              std::get<Fortran::parser::OmpDependClause::InOut>(
                  dependClause->v.u)
                  .t);
      omp::ClauseTaskDependAttr dependTypeOperand =
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
        dependOperands.push_back(((variable)));
      }
    } else {
      TODO(converter.getCurrentLocation(), "OpenMP Block construct clause");
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
        allocateOperands, allocatorOperands, reductionVars,
        reductionDeclSymbols.empty()
            ? nullptr
            : mlir::ArrayAttr::get(firOpBuilder.getContext(),
                                   reductionDeclSymbols),
        procBindKindAttr);
    createBodyOfOp<omp::ParallelOp>(parallelOp, converter, currentLocation,
                                    eval, &opClauseList);
  } else if (blockDirective.v == llvm::omp::OMPD_master) {
    auto masterOp =
        firOpBuilder.create<mlir::omp::MasterOp>(currentLocation, argTy);
    createBodyOfOp<omp::MasterOp>(masterOp, converter, currentLocation, eval);
  } else if (blockDirective.v == llvm::omp::OMPD_single) {
    auto singleOp = firOpBuilder.create<mlir::omp::SingleOp>(
        currentLocation, allocateOperands, allocatorOperands, nowaitAttr);
    createBodyOfOp<omp::SingleOp>(singleOp, converter, currentLocation, eval,
                                  &opClauseList);
  } else if (blockDirective.v == llvm::omp::OMPD_ordered) {
    auto orderedOp = firOpBuilder.create<mlir::omp::OrderedRegionOp>(
        currentLocation, /*simd=*/false);
    createBodyOfOp<omp::OrderedRegionOp>(orderedOp, converter, currentLocation,
                                         eval);
  } else if (blockDirective.v == llvm::omp::OMPD_task) {
    auto taskOp = firOpBuilder.create<mlir::omp::TaskOp>(
        currentLocation, ifClauseOperand, finalClauseOperand, untiedAttr,
        mergeableAttr, /*in_reduction_vars=*/ValueRange(),
        /*in_reductions=*/nullptr, priorityClauseOperand,
        dependTypeOperands.empty()
            ? nullptr
            : mlir::ArrayAttr::get(firOpBuilder.getContext(),
                                   dependTypeOperands),
        dependOperands, allocateOperands, allocatorOperands);
    createBodyOfOp(taskOp, converter, currentLocation, eval, &opClauseList);
  } else if (blockDirective.v == llvm::omp::OMPD_taskgroup) {
    // TODO: Add task_reduction support
    auto taskGroupOp = firOpBuilder.create<mlir::omp::TaskGroupOp>(
        currentLocation, /*task_reduction_vars=*/ValueRange(),
        /*task_reductions=*/nullptr, allocateOperands, allocatorOperands);
    createBodyOfOp(taskGroupOp, converter, currentLocation, eval,
                   &opClauseList);
  } else if (blockDirective.v == llvm::omp::OMPD_target) {
    createTargetOp(converter, opClauseList, blockDirective.v, currentLocation,
                   &eval);
  } else if (blockDirective.v == llvm::omp::OMPD_target_data) {
    createTargetOp(converter, opClauseList, blockDirective.v, currentLocation,
                   &eval);
  } else {
    TODO(currentLocation, "Unhandled block directive");
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
                               firOpBuilder.getContext(), global.getSymName()));
    }
  }();
  createBodyOfOp<omp::CriticalOp>(criticalOp, converter, currentLocation, eval);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPSectionConstruct &sectionConstruct) {

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  const Fortran::parser::OpenMPConstruct *parentOmpConstruct =
      eval.parentConstruct->getIf<Fortran::parser::OpenMPConstruct>();
  assert(parentOmpConstruct &&
         "No enclosing parent OpenMPConstruct on SECTION construct");
  const Fortran::parser::OpenMPSectionsConstruct *sectionsConstruct =
      std::get_if<Fortran::parser::OpenMPSectionsConstruct>(
          &parentOmpConstruct->u);
  assert(sectionsConstruct && "SECTION construct must have parent"
                              "SECTIONS construct");
  const Fortran::parser::OmpClauseList &sectionsClauseList =
      std::get<Fortran::parser::OmpClauseList>(
          std::get<Fortran::parser::OmpBeginSectionsDirective>(
              sectionsConstruct->t)
              .t);
  // Currently only private/firstprivate clause is handled, and
  // all privatization is done within `omp.section` operations.
  mlir::omp::SectionOp sectionOp =
      firOpBuilder.create<mlir::omp::SectionOp>(currentLocation);
  createBodyOfOp<omp::SectionOp>(sectionOp, converter, currentLocation, eval,
                                 &sectionsClauseList);
}

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
    createCombinedParallelOp<Fortran::parser::OmpBeginSectionsDirective>(
        converter, eval,
        std::get<Fortran::parser::OmpBeginSectionsDirective>(
            sectionsConstruct.t));
    auto sectionsOp = firOpBuilder.create<mlir::omp::SectionsOp>(
        currentLocation, /*reduction_vars*/ ValueRange(),
        /*reductions=*/nullptr, allocateOperands, allocatorOperands,
        /*nowait=*/nullptr);
    createBodyOfOp(sectionsOp, converter, currentLocation, eval);

    // Sections Construct
  } else if (dir == llvm::omp::Directive::OMPD_sections) {
    auto sectionsOp = firOpBuilder.create<mlir::omp::SectionsOp>(
        currentLocation, reductionVars, /*reductions = */ nullptr,
        allocateOperands, allocatorOperands, noWaitClauseOperand);
    createBodyOfOp<omp::SectionsOp>(sectionsOp, converter, currentLocation,
                                    eval);
  }
}

static bool checkForSingleVariableOnRHS(
    const Fortran::parser::AssignmentStmt &assignmentStmt) {
  // Check if the assignment statement has a single variable on the RHS
  const Fortran::parser::Expr &expr{
      std::get<Fortran::parser::Expr>(assignmentStmt.t)};
  const Fortran::common::Indirection<Fortran::parser::Designator> *designator =
      std::get_if<Fortran::common::Indirection<Fortran::parser::Designator>>(
          &expr.u);
  const Fortran::parser::Name *name =
      designator
          ? Fortran::semantics::getDesignatorNameIfDataRef(designator->value())
          : nullptr;
  return name != nullptr;
}

static bool
checkForSymbolMatch(const Fortran::parser::AssignmentStmt &assignmentStmt) {
  // Check if the symbol on the LHS of the assignment statement is present in
  // the RHS expression
  const auto &var{std::get<Fortran::parser::Variable>(assignmentStmt.t)};
  const auto &expr{std::get<Fortran::parser::Expr>(assignmentStmt.t)};
  const auto *e{Fortran::semantics::GetExpr(expr)};
  const auto *v{Fortran::semantics::GetExpr(var)};
  const Fortran::semantics::Symbol &varSymbol =
      Fortran::evaluate::GetSymbolVector(*v).front();
  for (const Fortran::semantics::Symbol &symbol :
       Fortran::evaluate::GetSymbolVector(*e))
    if (varSymbol == symbol)
      return true;
  return false;
}

static void genOmpAtomicHintAndMemoryOrderClauses(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::OmpAtomicClauseList &clauseList,
    mlir::IntegerAttr &hint,
    mlir::omp::ClauseMemoryOrderKindAttr &memoryOrder) {
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
        memoryOrder = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Acquire);
      } else if (std::get_if<Fortran::parser::OmpClause::Relaxed>(
                     &ompMemoryOrderClause->v.u)) {
        memoryOrder = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Relaxed);
      } else if (std::get_if<Fortran::parser::OmpClause::SeqCst>(
                     &ompMemoryOrderClause->v.u)) {
        memoryOrder = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Seq_cst);
      } else if (std::get_if<Fortran::parser::OmpClause::Release>(
                     &ompMemoryOrderClause->v.u)) {
        memoryOrder = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Release);
      }
    }
  }
}

static void genOmpAtomicCaptureStatement(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval, mlir::Value from_address,
    mlir::Value to_address,
    const Fortran::parser::OmpAtomicClauseList *leftHandClauseList,
    const Fortran::parser::OmpAtomicClauseList *rightHandClauseList,
    mlir::Type elementType) {
  // Generate `omp.atomic.read` operation for atomic assigment statements
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  // If no hint clause is specified, the effect is as if
  // hint(omp_sync_hint_none) had been specified.
  mlir::IntegerAttr hint = nullptr;

  mlir::omp::ClauseMemoryOrderKindAttr memory_order = nullptr;
  if (leftHandClauseList)
    genOmpAtomicHintAndMemoryOrderClauses(converter, *leftHandClauseList, hint,
                                          memory_order);
  if (rightHandClauseList)
    genOmpAtomicHintAndMemoryOrderClauses(converter, *rightHandClauseList, hint,
                                          memory_order);
  firOpBuilder.create<mlir::omp::AtomicReadOp>(
      currentLocation, from_address, to_address,
      mlir::TypeAttr::get(elementType), hint, memory_order);
}

static void genOmpAtomicWriteStatement(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval, mlir::Value lhs_addr,
    mlir::Value rhs_expr,
    const Fortran::parser::OmpAtomicClauseList *leftHandClauseList,
    const Fortran::parser::OmpAtomicClauseList *rightHandClauseList,
    mlir::Value *evaluatedExprValue = nullptr) {
  // Generate `omp.atomic.write` operation for atomic assignment statements
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  // If no hint clause is specified, the effect is as if
  // hint(omp_sync_hint_none) had been specified.
  mlir::IntegerAttr hint = nullptr;
  mlir::omp::ClauseMemoryOrderKindAttr memory_order = nullptr;
  if (leftHandClauseList)
    genOmpAtomicHintAndMemoryOrderClauses(converter, *leftHandClauseList, hint,
                                          memory_order);
  if (rightHandClauseList)
    genOmpAtomicHintAndMemoryOrderClauses(converter, *rightHandClauseList, hint,
                                          memory_order);
  firOpBuilder.create<mlir::omp::AtomicWriteOp>(currentLocation, lhs_addr,
                                                rhs_expr, hint, memory_order);
}

static void genOmpAtomicUpdateStatement(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval, mlir::Value lhs_addr,
    mlir::Type varType, const Fortran::parser::Variable &assignmentStmtVariable,
    const Fortran::parser::Expr &assignmentStmtExpr,
    const Fortran::parser::OmpAtomicClauseList *leftHandClauseList,
    const Fortran::parser::OmpAtomicClauseList *rightHandClauseList) {
  // Generate `omp.atomic.update` operation for atomic assignment statements
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  // If no hint clause is specified, the effect is as if
  // hint(omp_sync_hint_none) had been specified.
  mlir::IntegerAttr hint = nullptr;
  mlir::omp::ClauseMemoryOrderKindAttr memoryOrder = nullptr;
  if (leftHandClauseList)
    genOmpAtomicHintAndMemoryOrderClauses(converter, *leftHandClauseList, hint,
                                          memoryOrder);
  if (rightHandClauseList)
    genOmpAtomicHintAndMemoryOrderClauses(converter, *rightHandClauseList, hint,
                                          memoryOrder);
  auto atomicUpdateOp = firOpBuilder.create<mlir::omp::AtomicUpdateOp>(
      currentLocation, lhs_addr, hint, memoryOrder);

  //// Generate body of Atomic Update operation
  // If an argument for the region is provided then create the block with that
  // argument. Also update the symbol's address with the argument mlir value.
  SmallVector<Type> varTys = {varType};
  SmallVector<Location> locs = {currentLocation};
  firOpBuilder.createBlock(&atomicUpdateOp.getRegion(), {}, varTys, locs);
  mlir::Value val =
      fir::getBase(atomicUpdateOp.getRegion().front().getArgument(0));
  auto varDesignator =
      std::get_if<Fortran::common::Indirection<Fortran::parser::Designator>>(
          &assignmentStmtVariable.u);
  assert(varDesignator && "Variable designator for atomic update assignment "
                          "statement does not exist");
  const auto *name =
      Fortran::semantics::getDesignatorNameIfDataRef(varDesignator->value());
  if (!name)
    TODO(converter.getCurrentLocation(),
         "Array references as atomic update variable");
  assert(name && name->symbol &&
         "No symbol attached to atomic update variable");
  converter.bindSymbol(*name->symbol, val);
  // Set the insert for the terminator operation to go at the end of the
  // block.
  mlir::Block &block = atomicUpdateOp.getRegion().back();
  firOpBuilder.setInsertionPointToEnd(&block);

  Fortran::lower::StatementContext stmtCtx;
  mlir::Value rhs_expr = fir::getBase(converter.genExprValue(
      *Fortran::semantics::GetExpr(assignmentStmtExpr), stmtCtx));
  mlir::Value convertResult =
      firOpBuilder.createConvert(currentLocation, varType, rhs_expr);
  // Insert the terminator: YieldOp.
  firOpBuilder.create<mlir::omp::YieldOp>(currentLocation, convertResult);
  // Reset the insert point to before the terminator.
  firOpBuilder.setInsertionPointToStart(&block);
}

static void
genOmpAtomicWrite(Fortran::lower::AbstractConverter &converter,
                  Fortran::lower::pft::Evaluation &eval,
                  const Fortran::parser::OmpAtomicWrite &atomicWrite) {
  // Get the value and address of atomic write operands.
  const Fortran::parser::OmpAtomicClauseList &rightHandClauseList =
      std::get<2>(atomicWrite.t);
  const Fortran::parser::OmpAtomicClauseList &leftHandClauseList =
      std::get<0>(atomicWrite.t);
  const Fortran::parser::AssignmentStmt &stmt =
      std::get<3>(atomicWrite.t).statement;
  const Fortran::evaluate::Assignment &assign = *stmt.typedAssignment->v;
  Fortran::lower::StatementContext stmtCtx;
  // Get the value and address of atomic write operands.
  mlir::Value rhs_expr =
      fir::getBase(converter.genExprValue(assign.rhs, stmtCtx));

  mlir::Value lhs_addr =
      fir::getBase(converter.genExprAddr(assign.lhs, stmtCtx));
  genOmpAtomicWriteStatement(converter, eval, lhs_addr, rhs_expr,
                             &leftHandClauseList, &rightHandClauseList);
}

static void genOmpAtomicRead(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::pft::Evaluation &eval,
                             const Fortran::parser::OmpAtomicRead &atomicRead) {
  // Get the address of atomic read operands.
  const Fortran::parser::OmpAtomicClauseList &rightHandClauseList =
      std::get<2>(atomicRead.t);
  const Fortran::parser::OmpAtomicClauseList &leftHandClauseList =
      std::get<0>(atomicRead.t);
  const auto &assignmentStmtExpr =
      std::get<Fortran::parser::Expr>(std::get<3>(atomicRead.t).statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<3>(atomicRead.t).statement.t);

  Fortran::lower::StatementContext stmtCtx;
  const Fortran::semantics::SomeExpr &fromExpr =
      *Fortran::semantics::GetExpr(assignmentStmtExpr);
  mlir::Type elementType = converter.genType(fromExpr);
  mlir::Value fromAddress =
      fir::getBase(converter.genExprAddr(fromExpr, stmtCtx));
  mlir::Value toAddress = fir::getBase(converter.genExprAddr(
      *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx));
  genOmpAtomicCaptureStatement(converter, eval, fromAddress, toAddress,
                               &leftHandClauseList, &rightHandClauseList,
                               elementType);
}

static void
genOmpAtomicUpdate(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OmpAtomicUpdate &atomicUpdate) {
  const Fortran::parser::OmpAtomicClauseList &rightHandClauseList =
      std::get<2>(atomicUpdate.t);
  const Fortran::parser::OmpAtomicClauseList &leftHandClauseList =
      std::get<0>(atomicUpdate.t);
  const auto &assignmentStmtExpr =
      std::get<Fortran::parser::Expr>(std::get<3>(atomicUpdate.t).statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<3>(atomicUpdate.t).statement.t);

  Fortran::lower::StatementContext stmtCtx;
  mlir::Value lhs_addr = fir::getBase(converter.genExprAddr(
      *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx));
  mlir::Type varType =
      fir::getBase(
          converter.genExprValue(
              *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx))
          .getType();
  genOmpAtomicUpdateStatement(converter, eval, lhs_addr, varType,
                              assignmentStmtVariable, assignmentStmtExpr,
                              &leftHandClauseList, &rightHandClauseList);
}

static void genOmpAtomic(Fortran::lower::AbstractConverter &converter,
                         Fortran::lower::pft::Evaluation &eval,
                         const Fortran::parser::OmpAtomic &atomicConstruct) {
  const Fortran::parser::OmpAtomicClauseList &atomicClauseList =
      std::get<Fortran::parser::OmpAtomicClauseList>(atomicConstruct.t);
  const auto &assignmentStmtExpr = std::get<Fortran::parser::Expr>(
      std::get<Fortran::parser::Statement<Fortran::parser::AssignmentStmt>>(
          atomicConstruct.t)
          .statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<Fortran::parser::Statement<Fortran::parser::AssignmentStmt>>(
          atomicConstruct.t)
          .statement.t);
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value lhs_addr = fir::getBase(converter.genExprAddr(
      *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx));
  mlir::Type varType =
      fir::getBase(
          converter.genExprValue(
              *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx))
          .getType();
  // If atomic-clause is not present on the construct, the behaviour is as if
  // the update clause is specified
  genOmpAtomicUpdateStatement(converter, eval, lhs_addr, varType,
                              assignmentStmtVariable, assignmentStmtExpr,
                              &atomicClauseList, nullptr);
}

static void
genOmpAtomicCapture(Fortran::lower::AbstractConverter &converter,
                    Fortran::lower::pft::Evaluation &eval,
                    const Fortran::parser::OmpAtomicCapture &atomicCapture) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  mlir::IntegerAttr hint = nullptr;
  mlir::omp::ClauseMemoryOrderKindAttr memory_order = nullptr;
  const Fortran::parser::OmpAtomicClauseList &rightHandClauseList =
      std::get<2>(atomicCapture.t);
  const Fortran::parser::OmpAtomicClauseList &leftHandClauseList =
      std::get<0>(atomicCapture.t);
  genOmpAtomicHintAndMemoryOrderClauses(converter, leftHandClauseList, hint,
                                        memory_order);
  genOmpAtomicHintAndMemoryOrderClauses(converter, rightHandClauseList, hint,
                                        memory_order);

  const Fortran::parser::AssignmentStmt &stmt1 =
      std::get<3>(atomicCapture.t).v.statement;
  const auto &stmt1Var{std::get<Fortran::parser::Variable>(stmt1.t)};
  const auto &stmt1Expr{std::get<Fortran::parser::Expr>(stmt1.t)};
  const Fortran::parser::AssignmentStmt &stmt2 =
      std::get<4>(atomicCapture.t).v.statement;
  const auto &stmt2Var{std::get<Fortran::parser::Variable>(stmt2.t)};
  const auto &stmt2Expr{std::get<Fortran::parser::Expr>(stmt2.t)};

  // Pre-evaluate expressions to be used in the various operations inside
  // `omp.atomic.capture` since it is not desirable to have anything other than
  // a `omp.atomic.read`, `omp.atomic.write`, or `omp.atomic.update` operation
  // inside `omp.atomic.capture`
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value stmt1LHSArg, stmt1RHSArg, stmt2LHSArg, stmt2RHSArg;
  mlir::Type elementType;
  // LHS evaluations are common to all combinations of `omp.atomic.capture`
  stmt1LHSArg = fir::getBase(
      converter.genExprAddr(*Fortran::semantics::GetExpr(stmt1Var), stmtCtx));
  stmt2LHSArg = fir::getBase(
      converter.genExprAddr(*Fortran::semantics::GetExpr(stmt2Var), stmtCtx));

  // Operation specific RHS evaluations
  if (checkForSingleVariableOnRHS(stmt1)) {
    // Atomic capture construct is of the form [capture-stmt, update-stmt] or
    // of the form [capture-stmt, write-stmt]
    stmt1RHSArg = fir::getBase(converter.genExprAddr(
        *Fortran::semantics::GetExpr(stmt1Expr), stmtCtx));
    stmt2RHSArg = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(stmt2Expr), stmtCtx));

  } else {
    // Atomic capture construct is of the form [update-stmt, capture-stmt]
    stmt1RHSArg = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(stmt1Expr), stmtCtx));
    stmt2RHSArg = fir::getBase(converter.genExprAddr(
        *Fortran::semantics::GetExpr(stmt2Expr), stmtCtx));
  }
  // Type information used in generation of `omp.atomic.update` operation
  mlir::Type stmt1VarType =
      fir::getBase(converter.genExprValue(
                       *Fortran::semantics::GetExpr(stmt1Var), stmtCtx))
          .getType();
  mlir::Type stmt2VarType =
      fir::getBase(converter.genExprValue(
                       *Fortran::semantics::GetExpr(stmt2Var), stmtCtx))
          .getType();

  auto atomicCaptureOp = firOpBuilder.create<mlir::omp::AtomicCaptureOp>(
      currentLocation, hint, memory_order);
  firOpBuilder.createBlock(&atomicCaptureOp.getRegion());
  mlir::Block &block = atomicCaptureOp.getRegion().back();
  firOpBuilder.setInsertionPointToStart(&block);
  if (checkForSingleVariableOnRHS(stmt1)) {
    if (checkForSymbolMatch(stmt2)) {
      // Atomic capture construct is of the form [capture-stmt, update-stmt]
      const Fortran::semantics::SomeExpr &fromExpr =
          *Fortran::semantics::GetExpr(stmt1Expr);
      elementType = converter.genType(fromExpr);
      genOmpAtomicCaptureStatement(converter, eval, stmt1RHSArg, stmt1LHSArg,
                                   /*leftHandClauseList=*/nullptr,
                                   /*rightHandClauseList=*/nullptr,
                                   elementType);
      genOmpAtomicUpdateStatement(converter, eval, stmt1RHSArg, stmt2VarType,
                                  stmt2Var, stmt2Expr,
                                  /*leftHandClauseList=*/nullptr,
                                  /*rightHandClauseList=*/nullptr);
    } else {
      // Atomic capture construct is of the form [capture-stmt, write-stmt]
      const Fortran::semantics::SomeExpr &fromExpr =
          *Fortran::semantics::GetExpr(stmt1Expr);
      elementType = converter.genType(fromExpr);
      genOmpAtomicCaptureStatement(converter, eval, stmt1RHSArg, stmt1LHSArg,
                                   /*leftHandClauseList=*/nullptr,
                                   /*rightHandClauseList=*/nullptr,
                                   elementType);
      genOmpAtomicWriteStatement(converter, eval, stmt1RHSArg, stmt2RHSArg,
                                 /*leftHandClauseList=*/nullptr,
                                 /*rightHandClauseList=*/nullptr);
    }
  } else {
    // Atomic capture construct is of the form [update-stmt, capture-stmt]
    firOpBuilder.setInsertionPointToEnd(&block);
    const Fortran::semantics::SomeExpr &fromExpr =
        *Fortran::semantics::GetExpr(stmt2Expr);
    elementType = converter.genType(fromExpr);
    genOmpAtomicCaptureStatement(converter, eval, stmt1LHSArg, stmt2LHSArg,
                                 /*leftHandClauseList=*/nullptr,
                                 /*rightHandClauseList=*/nullptr, elementType);
    firOpBuilder.setInsertionPointToStart(&block);
    genOmpAtomicUpdateStatement(converter, eval, stmt1LHSArg, stmt1VarType,
                                stmt1Var, stmt1Expr,
                                /*leftHandClauseList=*/nullptr,
                                /*rightHandClauseList=*/nullptr);
  }
  firOpBuilder.setInsertionPointToEnd(&block);
  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);
  firOpBuilder.setInsertionPointToStart(&block);
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
                 [&](const Fortran::parser::OmpAtomic &atomicConstruct) {
                   genOmpAtomic(converter, eval, atomicConstruct);
                 },
                 [&](const Fortran::parser::OmpAtomicUpdate &atomicUpdate) {
                   genOmpAtomicUpdate(converter, eval, atomicUpdate);
                 },
                 [&](const Fortran::parser::OmpAtomicCapture &atomicCapture) {
                   genOmpAtomicCapture(converter, eval, atomicCapture);
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
          [&](const Fortran::parser::OpenMPAllocatorsConstruct
                  &allocsConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPAllocatorsConstruct");
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

fir::GlobalOp globalInitialization(Fortran::lower::AbstractConverter &converter,
                                   fir::FirOpBuilder &firOpBuilder,
                                   const Fortran::semantics::Symbol &sym,
                                   const Fortran::lower::pft::Variable &var,
                                   mlir::Location currentLocation) {
  mlir::Type ty = converter.genType(sym);
  std::string globalName = converter.mangleName(sym);
  mlir::StringAttr linkage = firOpBuilder.createInternalLinkage();
  fir::GlobalOp global =
      firOpBuilder.createGlobal(currentLocation, ty, globalName, linkage);

  // Create default initialization for non-character scalar.
  if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
    mlir::Type baseAddrType = ty.dyn_cast<fir::BoxType>().getEleTy();
    Fortran::lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value nullAddr =
              b.createNullConstant(currentLocation, baseAddrType);
          mlir::Value box =
              b.create<fir::EmboxOp>(currentLocation, ty, nullAddr);
          b.create<fir::HasValueOp>(currentLocation, box);
        });
  } else {
    Fortran::lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value undef = b.create<fir::UndefOp>(currentLocation, ty);
          b.create<fir::HasValueOp>(currentLocation, undef);
        });
  }

  return global;
}

void Fortran::lower::genThreadprivateOp(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  const Fortran::semantics::Symbol &sym = var.getSymbol();
  mlir::Value symThreadprivateValue;
  if (const Fortran::semantics::Symbol *common =
          Fortran::semantics::FindCommonBlockContaining(sym.GetUltimate())) {
    mlir::Value commonValue = converter.getSymbolAddress(*common);
    if (mlir::isa<mlir::omp::ThreadprivateOp>(commonValue.getDefiningOp())) {
      // Generate ThreadprivateOp for a common block instead of its members and
      // only do it once for a common block.
      return;
    }
    // Generate ThreadprivateOp and rebind the common block.
    mlir::Value commonThreadprivateValue =
        firOpBuilder.create<mlir::omp::ThreadprivateOp>(
            currentLocation, commonValue.getType(), commonValue);
    converter.bindSymbol(*common, commonThreadprivateValue);
    // Generate the threadprivate value for the common block member.
    symThreadprivateValue =
        genCommonBlockMember(converter, sym, commonThreadprivateValue);
  } else if (!var.isGlobal()) {
    // Non-global variable which can be in threadprivate directive must be one
    // variable in main program, and it has implicit SAVE attribute. Take it as
    // with SAVE attribute, so to create GlobalOp for it to simplify the
    // translation to LLVM IR.
    fir::GlobalOp global = globalInitialization(converter, firOpBuilder, sym,
                                                var, currentLocation);

    mlir::Value symValue = firOpBuilder.create<fir::AddrOfOp>(
        currentLocation, global.resultType(), global.getSymbol());
    symThreadprivateValue = firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  } else {
    mlir::Value symValue = converter.getSymbolAddress(sym);
    mlir::Operation *op = symValue.getDefiningOp();
    // The symbol may be use-associated multiple times, and nothing needs to be
    // done after the original symbol is mapped to the threadprivatized value
    // for the first time. Use the threadprivatized value directly.
    if (mlir::isa<mlir::omp::ThreadprivateOp>(op))
      return;
    symThreadprivateValue = firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  }

  fir::ExtendedValue sexv = converter.getSymbolExtendedValue(sym);
  fir::ExtendedValue symThreadprivateExv =
      getExtendedValue(sexv, symThreadprivateValue);
  converter.bindSymbol(sym, symThreadprivateExv);
}

// This function replicates threadprivate's behaviour of generating
// an internal fir.GlobalOp for non-global variables in the main program
// that have the implicit SAVE attribute, to simplifiy LLVM-IR and MLIR
// generation.
void Fortran::lower::genDeclareTargetIntGlobal(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  if (!var.isGlobal()) {
    // A non-global variable which can be in a declare target directive must
    // be a variable in the main program, and it has the implicit SAVE
    // attribute. We create a GlobalOp for it to simplify the translation to
    // LLVM IR.
    globalInitialization(converter, converter.getFirOpBuilder(),
                         var.getSymbol(), var, converter.getCurrentLocation());
  }
}

void handleDeclareTarget(Fortran::lower::AbstractConverter &converter,
                         Fortran::lower::pft::Evaluation &eval,
                         const Fortran::parser::OpenMPDeclareTargetConstruct
                             &declareTargetConstruct) {
  llvm::SmallVector<std::pair<mlir::omp::DeclareTargetCaptureClause,
                              Fortran::semantics::Symbol>,
                    0>
      symbolAndClause;
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();

  auto findFuncAndVarSyms = [&](const Fortran::parser::OmpObjectList &objList,
                                mlir::omp::DeclareTargetCaptureClause clause) {
    for (const Fortran::parser::OmpObject &ompObject : objList.v) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Designator &designator) {
                if (const Fortran::parser::Name *name =
                        Fortran::semantics::getDesignatorNameIfDataRef(
                            designator)) {
                  symbolAndClause.push_back(
                      std::make_pair(clause, *name->symbol));
                }
              },
              [&](const Fortran::parser::Name &name) {
                symbolAndClause.push_back(std::make_pair(clause, *name.symbol));
              }},
          ompObject.u);
    }
  };

  // The default capture type
  Fortran::parser::OmpDeviceTypeClause::Type deviceType =
      Fortran::parser::OmpDeviceTypeClause::Type::Any;
  const auto &spec = std::get<Fortran::parser::OmpDeclareTargetSpecifier>(
      declareTargetConstruct.t);
  if (const auto *objectList{
          Fortran::parser::Unwrap<Fortran::parser::OmpObjectList>(spec.u)}) {
    // Case: declare target(func, var1, var2)
    findFuncAndVarSyms(*objectList, mlir::omp::DeclareTargetCaptureClause::to);
  } else if (const auto *clauseList{
                 Fortran::parser::Unwrap<Fortran::parser::OmpClauseList>(
                     spec.u)}) {
    if (clauseList->v.empty()) {
      // Case: declare target, implicit capture of function
      symbolAndClause.push_back(
          std::make_pair(mlir::omp::DeclareTargetCaptureClause::to,
                         eval.getOwningProcedure()->getSubprogramSymbol()));
    }

    for (const Fortran::parser::OmpClause &clause : clauseList->v) {
      if (const auto *toClause =
              std::get_if<Fortran::parser::OmpClause::To>(&clause.u)) {
        // Case: declare target to(func, var1, var2)...
        findFuncAndVarSyms(toClause->v,
                           mlir::omp::DeclareTargetCaptureClause::to);
      } else if (const auto *linkClause =
                     std::get_if<Fortran::parser::OmpClause::Link>(&clause.u)) {
        // Case: declare target link(var1, var2)...
        findFuncAndVarSyms(linkClause->v,
                           mlir::omp::DeclareTargetCaptureClause::link);
      } else if (const auto *deviceClause =
                     std::get_if<Fortran::parser::OmpClause::DeviceType>(
                         &clause.u)) {
        // Case: declare target ... device_type(any | host | nohost)
        deviceType = deviceClause->v.v;
      }
    }
  }

  for (std::pair<mlir::omp::DeclareTargetCaptureClause,
                 Fortran::semantics::Symbol>
           symClause : symbolAndClause) {
    mlir::Operation *op =
        mod.lookupSymbol(converter.mangleName(std::get<1>(symClause)));
    // There's several cases this can currently be triggered and it could be
    // one of the following:
    // 1) Invalid argument passed to a declare target that currently isn't
    // captured by a frontend semantic check
    // 2) The symbol of a valid argument is not correctly updated by one of
    // the prior passes, resulting in missing symbol information
    // 3) It's a variable internal to a module or program, that is legal by
    // Fortran OpenMP standards, but is currently unhandled as they do not
    // appear in the symbol table as they are represented as allocas
    if (!op)
      TODO(converter.getCurrentLocation(),
           "Missing symbol, possible case of currently unsupported use of "
           "a program local variable in declare target or erroneous symbol "
           "information ");

    auto declareTargetOp = dyn_cast<mlir::omp::DeclareTargetInterface>(op);
    if (!declareTargetOp)
      fir::emitFatalError(
          converter.getCurrentLocation(),
          "Attempt to apply declare target on unsupported operation");

    mlir::omp::DeclareTargetDeviceType newDeviceType;
    switch (deviceType) {
    case Fortran::parser::OmpDeviceTypeClause::Type::Nohost:
      newDeviceType = mlir::omp::DeclareTargetDeviceType::nohost;
      break;
    case Fortran::parser::OmpDeviceTypeClause::Type::Host:
      newDeviceType = mlir::omp::DeclareTargetDeviceType::host;
      break;
    case Fortran::parser::OmpDeviceTypeClause::Type::Any:
      newDeviceType = mlir::omp::DeclareTargetDeviceType::any;
      break;
    }

    // The function or global already has a declare target applied to it,
    // very likely through implicit capture (usage in another declare
    // target function/subroutine). It should be marked as any if it has
    // been assigned both host and nohost, else we skip, as there is no
    // change
    if (declareTargetOp.isDeclareTarget()) {
      if (declareTargetOp.getDeclareTargetDeviceType() != newDeviceType)
        declareTargetOp.setDeclareTarget(
            mlir::omp::DeclareTargetDeviceType::any, std::get<0>(symClause));
      continue;
    }

    declareTargetOp.setDeclareTarget(newDeviceType, std::get<0>(symClause));
  }
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
            handleDeclareTarget(converter, eval, declareTargetConstruct);
          },
          [&](const Fortran::parser::OpenMPRequiresConstruct
                  &requiresConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPRequiresConstruct");
          },
          [&](const Fortran::parser::OpenMPThreadprivate &threadprivate) {
            // The directive is lowered when instantiating the variable to
            // support the case of threadprivate variable declared in module.
          },
      },
      ompDeclConstruct.u);
}

static mlir::Operation *getCompareFromReductionOp(mlir::Operation *reductionOp,
                                                  mlir::Value loadVal) {
  for (auto reductionOperand : reductionOp->getOperands()) {
    if (auto compareOp = reductionOperand.getDefiningOp()) {
      if (compareOp->getOperand(0) == loadVal ||
          compareOp->getOperand(1) == loadVal)
        assert((mlir::isa<mlir::arith::CmpIOp>(compareOp) ||
                mlir::isa<mlir::arith::CmpFOp>(compareOp)) &&
               "Expected comparison not found in reduction intrinsic");
      return compareOp;
    }
  }
  return nullptr;
}

// Generate an OpenMP reduction operation.
// TODO: Currently assumes it is either an integer addition/multiplication
// reduction, or a logical and reduction. Generalize this for various reduction
// operation types.
// TODO: Generate the reduction operation during lowering instead of creating
// and removing operations since this is not a robust approach. Also, removing
// ops in the builder (instead of a rewriter) is probably not the best approach.
void Fortran::lower::genOpenMPReduction(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  for (const auto &clause : clauseList.v) {
    if (const auto &reductionClause =
            std::get_if<Fortran::parser::OmpClause::Reduction>(&clause.u)) {
      const auto &redOperator{std::get<Fortran::parser::OmpReductionOperator>(
          reductionClause->v.t)};
      const auto &objectList{
          std::get<Fortran::parser::OmpObjectList>(reductionClause->v.t)};
      if (auto reductionOp =
              std::get_if<Fortran::parser::DefinedOperator>(&redOperator.u)) {
        const auto &intrinsicOp{
            std::get<Fortran::parser::DefinedOperator::IntrinsicOperator>(
                reductionOp->u)};

        switch (intrinsicOp) {
        case Fortran::parser::DefinedOperator::IntrinsicOperator::Add:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::Multiply:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::AND:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::EQV:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::OR:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::NEQV:
          break;
        default:
          continue;
        }
        for (const auto &ompObject : objectList.v) {
          if (const auto *name{
                  Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
            if (const auto *symbol{name->symbol}) {
              mlir::Value reductionVal = converter.getSymbolAddress(*symbol);
              mlir::Type reductionType =
                  reductionVal.getType().cast<fir::ReferenceType>().getEleTy();
              if (!reductionType.isa<fir::LogicalType>()) {
                if (!reductionType.isIntOrIndexOrFloat())
                  continue;
              }
              for (mlir::OpOperand &reductionValUse : reductionVal.getUses()) {
                if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(
                        reductionValUse.getOwner())) {
                  mlir::Value loadVal = loadOp.getRes();
                  if (reductionType.isa<fir::LogicalType>()) {
                    mlir::Operation *reductionOp = findReductionChain(loadVal);
                    fir::ConvertOp convertOp =
                        getConvertFromReductionOp(reductionOp, loadVal);
                    updateReduction(reductionOp, firOpBuilder, loadVal,
                                    reductionVal, &convertOp);
                    removeStoreOp(reductionOp, reductionVal);
                  } else if (auto reductionOp =
                                 findReductionChain(loadVal, &reductionVal)) {
                    updateReduction(reductionOp, firOpBuilder, loadVal,
                                    reductionVal);
                  }
                }
              }
            }
          }
        }
      } else if (auto reductionIntrinsic =
                     std::get_if<Fortran::parser::ProcedureDesignator>(
                         &redOperator.u)) {
        if (const auto *name{Fortran::parser::Unwrap<Fortran::parser::Name>(
                reductionIntrinsic)}) {
          std::string redName = name->ToString();
          if ((name->source != "max") && (name->source != "min") &&
              (name->source != "ior") && (name->source != "ieor") &&
              (name->source != "iand")) {
            continue;
          }
          for (const auto &ompObject : objectList.v) {
            if (const auto *name{Fortran::parser::Unwrap<Fortran::parser::Name>(
                    ompObject)}) {
              if (const auto *symbol{name->symbol}) {
                mlir::Value reductionVal = converter.getSymbolAddress(*symbol);
                for (mlir::OpOperand &reductionValUse :
                     reductionVal.getUses()) {
                  if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(
                          reductionValUse.getOwner())) {
                    mlir::Value loadVal = loadOp.getRes();
                    // Max is lowered as a compare -> select.
                    // Match the pattern here.
                    mlir::Operation *reductionOp =
                        findReductionChain(loadVal, &reductionVal);
                    if (reductionOp == nullptr)
                      continue;

                    if (redName == "max" || redName == "min") {
                      assert(mlir::isa<mlir::arith::SelectOp>(reductionOp) &&
                             "Selection Op not found in reduction intrinsic");
                      mlir::Operation *compareOp =
                          getCompareFromReductionOp(reductionOp, loadVal);
                      updateReduction(compareOp, firOpBuilder, loadVal,
                                      reductionVal);
                    }
                    if (redName == "ior" || redName == "ieor" ||
                        redName == "iand") {

                      updateReduction(reductionOp, firOpBuilder, loadVal,
                                      reductionVal);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

mlir::Operation *Fortran::lower::findReductionChain(mlir::Value loadVal,
                                                    mlir::Value *reductionVal) {
  for (mlir::OpOperand &loadOperand : loadVal.getUses()) {
    if (auto reductionOp = loadOperand.getOwner()) {
      if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(reductionOp)) {
        for (mlir::OpOperand &convertOperand : convertOp.getRes().getUses()) {
          if (auto reductionOp = convertOperand.getOwner())
            return reductionOp;
        }
      }
      for (mlir::OpOperand &reductionOperand : reductionOp->getUses()) {
        if (auto store =
                mlir::dyn_cast<fir::StoreOp>(reductionOperand.getOwner())) {
          if (store.getMemref() == *reductionVal) {
            store.erase();
            return reductionOp;
          }
        }
      }
    }
  }
  return nullptr;
}

void Fortran::lower::updateReduction(mlir::Operation *op,
                                     fir::FirOpBuilder &firOpBuilder,
                                     mlir::Value loadVal,
                                     mlir::Value reductionVal,
                                     fir::ConvertOp *convertOp) {
  mlir::OpBuilder::InsertPoint insertPtDel = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPoint(op);

  mlir::Value reductionOp;
  if (convertOp)
    reductionOp = convertOp->getOperand();
  else if (op->getOperand(0) == loadVal)
    reductionOp = op->getOperand(1);
  else
    reductionOp = op->getOperand(0);

  firOpBuilder.create<mlir::omp::ReductionOp>(op->getLoc(), reductionOp,
                                              reductionVal);
  firOpBuilder.restoreInsertionPoint(insertPtDel);
}

// for a logical operator 'op' reduction X = X op Y
// This function returns the operation responsible for converting Y from
// fir.logical<4> to i1
fir::ConvertOp
Fortran::lower::getConvertFromReductionOp(mlir::Operation *reductionOp,
                                          mlir::Value loadVal) {
  for (auto reductionOperand : reductionOp->getOperands()) {
    if (auto convertOp =
            mlir::dyn_cast<fir::ConvertOp>(reductionOperand.getDefiningOp())) {
      if (convertOp.getOperand() == loadVal)
        continue;
      return convertOp;
    }
  }
  return nullptr;
}

void Fortran::lower::removeStoreOp(mlir::Operation *reductionOp,
                                   mlir::Value symVal) {
  for (auto reductionOpUse : reductionOp->getUsers()) {
    if (auto convertReduction =
            mlir::dyn_cast<fir::ConvertOp>(reductionOpUse)) {
      for (auto convertReductionUse : convertReduction.getRes().getUsers()) {
        if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(convertReductionUse)) {
          if (storeOp.getMemref() == symVal)
            storeOp.erase();
        }
      }
    }
  }
}
