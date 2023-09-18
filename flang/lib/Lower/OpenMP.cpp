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
#include "DirectivesCommon.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using DeclareTargetCapturePair =
    std::pair<mlir::omp::DeclareTargetCaptureClause,
              Fortran::semantics::Symbol>;

//===----------------------------------------------------------------------===//
// Common helper functions
//===----------------------------------------------------------------------===//

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

static void genObjectList(const Fortran::parser::OmpObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          llvm::SmallVectorImpl<mlir::Value> &operands) {
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

static void gatherFuncAndVarSyms(
    const Fortran::parser::OmpObjectList &objList,
    mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {
  for (const Fortran::parser::OmpObject &ompObject : objList.v) {
    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const Fortran::parser::Name *name =
                      Fortran::semantics::getDesignatorNameIfDataRef(
                          designator)) {
                symbolAndClause.emplace_back(clause, *name->symbol);
              }
            },
            [&](const Fortran::parser::Name &name) {
              symbolAndClause.emplace_back(clause, *name.symbol);
            }},
        ompObject.u);
  }
}

//===----------------------------------------------------------------------===//
// DataSharingProcessor
//===----------------------------------------------------------------------===//

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
  void processStep2(mlir::Operation *op, bool isLoop);
};

void DataSharingProcessor::processStep1() {
  collectSymbolsForPrivatization();
  collectDefaultSymbols();
  privatize();
  defaultPrivatize();
  insertBarrier();
}

void DataSharingProcessor::processStep2(mlir::Operation *op, bool isLoop) {
  insPt = firOpBuilder.saveInsertionPoint();
  copyLastPrivatize(op);
  firOpBuilder.restoreInsertionPoint(insPt);

  if (isLoop) {
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
  for (const Fortran::semantics::Symbol *sym : privatizedSymbols)
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

  if (hasCollapse && hasLastPrivateOp)
    TODO(converter.getCurrentLocation(), "Collapse clause with lastprivate");
}

bool DataSharingProcessor ::needBarrier() {
  for (const Fortran::semantics::Symbol *sym : privatizedSymbols) {
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
      if (mlir::isa<mlir::omp::SectionOp>(op)) {
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
      } else if (mlir::isa<mlir::omp::WsLoopOp>(op)) {
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
  for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
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
  for (const Fortran::semantics::Symbol *sym : privatizedSymbols) {
    if (const auto *commonDet =
            sym->detailsIf<Fortran::semantics::CommonBlockDetails>()) {
      for (const auto &mem : commonDet->objects()) {
        cloneSymbol(&*mem);
        copyFirstPrivateSymbol(&*mem);
      }
    } else {
      cloneSymbol(sym);
      copyFirstPrivateSymbol(sym);
    }
  }
}

void DataSharingProcessor::copyLastPrivatize(mlir::Operation *op) {
  insertLastPrivateCompare(op);
  for (const Fortran::semantics::Symbol *sym : privatizedSymbols)
    if (const auto *commonDet =
            sym->detailsIf<Fortran::semantics::CommonBlockDetails>()) {
      for (const auto &mem : commonDet->objects()) {
        copyLastPrivateSymbol(&*mem, &lastPrivIP);
      }
    } else {
      copyLastPrivateSymbol(sym, &lastPrivIP);
    }
}

void DataSharingProcessor::defaultPrivatize() {
  for (const Fortran::semantics::Symbol *sym : defaultSymbols) {
    if (!symbolsInNestedRegions.contains(sym) &&
        !symbolsInParentRegions.contains(sym) &&
        !privatizedSymbols.contains(sym)) {
      cloneSymbol(sym);
      copyFirstPrivateSymbol(sym);
    }
  }
}

//===----------------------------------------------------------------------===//
// ClauseProcessor
//===----------------------------------------------------------------------===//

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
                  const Fortran::parser::OmpClauseList &clauses)
      : converter(converter), clauses(clauses) {}

  // 'Unique' clauses: They can appear at most once in the clause list.
  bool
  processCollapse(mlir::Location currentLocation,
                  Fortran::lower::pft::Evaluation &eval,
                  llvm::SmallVectorImpl<mlir::Value> &lowerBound,
                  llvm::SmallVectorImpl<mlir::Value> &upperBound,
                  llvm::SmallVectorImpl<mlir::Value> &step,
                  llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> &iv,
                  std::size_t &loopVarTypeSize) const;
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
  bool processDepend(llvm::SmallVectorImpl<mlir::Attribute> &dependTypeOperands,
                     llvm::SmallVectorImpl<mlir::Value> &dependOperands) const;
  bool
  processIf(Fortran::lower::StatementContext &stmtCtx,
            Fortran::parser::OmpIfClause::DirectiveNameModifier directiveName,
            mlir::Value &result) const;
  bool
  processLink(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;
  bool processMap(llvm::SmallVectorImpl<mlir::Value> &mapOperands,
                  llvm::SmallVectorImpl<mlir::IntegerAttr> &mapTypes) const;
  bool processReduction(
      mlir::Location currentLocation,
      llvm::SmallVectorImpl<mlir::Value> &reductionVars,
      llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols) const;
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

  // Call this method for these clauses that should be supported but are not
  // implemented yet. It triggers a compilation error if any of the given
  // clauses is found.
  template <typename... Ts>
  void processTODO(mlir::Location currentLocation,
                   llvm::omp::Directive directive) const;

private:
  using ClauseIterator = std::list<ClauseTy>::const_iterator;

  /// Utility to find a clause within a range in the clause list.
  template <typename T>
  static ClauseIterator findClause(ClauseIterator begin, ClauseIterator end) {
    for (ClauseIterator it = begin; it != end; ++it) {
      if (std::get_if<T>(&it->u))
        return it;
    }

    return end;
  }

  /// Return the first instance of the given clause found in the clause list or
  /// `nullptr` if not present. If more than one instance is expected, use
  /// `findRepeatableClause` instead.
  template <typename T>
  const T *
  findUniqueClause(const Fortran::parser::CharBlock **source = nullptr) const {
    ClauseIterator it = findClause<T>(clauses.v.begin(), clauses.v.end());
    if (it != clauses.v.end()) {
      if (source)
        *source = &it->source;
      return &std::get<T>(it->u);
    }
    return nullptr;
  }

  /// Call `callbackFn` for each occurrence of the given clause. Return `true`
  /// if at least one instance was found.
  template <typename T>
  bool findRepeatableClause(
      std::function<void(const T *, const Fortran::parser::CharBlock &source)>
          callbackFn) const {
    bool found = false;
    ClauseIterator nextIt, endIt = clauses.v.end();
    for (ClauseIterator it = clauses.v.begin(); it != endIt; it = nextIt) {
      nextIt = findClause<T>(it, endIt);

      if (nextIt != endIt) {
        callbackFn(&std::get<T>(nextIt->u), nextIt->source);
        found = true;
        ++nextIt;
      }
    }
    return found;
  }

  /// Set the `result` to a new `mlir::UnitAttr` if the clause is present.
  template <typename T>
  bool markClauseOccurrence(mlir::UnitAttr &result) const {
    if (findUniqueClause<T>()) {
      result = converter.getFirOpBuilder().getUnitAttr();
      return true;
    }
    return false;
  }

  Fortran::lower::AbstractConverter &converter;
  const Fortran::parser::OmpClauseList &clauses;
};

//===----------------------------------------------------------------------===//
// ClauseProcessor helper functions
//===----------------------------------------------------------------------===//

/// Check for unsupported map operand types.
static void checkMapType(mlir::Location location, mlir::Type type) {
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getElementType();
  if (auto boxType = type.dyn_cast_or_null<fir::BoxType>())
    if (!boxType.getElementType().isa<fir::PointerType>())
      TODO(location, "OMPD_target_data MapOperand BoxType");
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

static mlir::Value getReductionInitValue(mlir::Location loc, mlir::Type type,
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
    if (type.isa<mlir::FloatType>())
      return builder.create<mlir::arith::ConstantOp>(
          loc, type,
          builder.getFloatAttr(
              type, (double)getOperationIdentity(reductionOpName, loc)));

    if (type.isa<fir::LogicalType>()) {
      mlir::Value intConst = builder.create<mlir::arith::ConstantOp>(
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
static mlir::Value getReductionOperation(fir::FirOpBuilder &builder,
                                         mlir::Type type, mlir::Location loc,
                                         mlir::Value op1, mlir::Value op2) {
  assert(type.isIntOrIndexOrFloat() &&
         "only integer and float types are currently supported");
  if (type.isIntOrIndex())
    return builder.create<IntegerOp>(loc, op1, op2);
  return builder.create<FloatOp>(loc, op1, op2);
}

static mlir::omp::ReductionDeclareOp
createMinimalReductionDecl(fir::FirOpBuilder &builder,
                           llvm::StringRef reductionOpName, mlir::Type type,
                           mlir::Location loc) {
  mlir::ModuleOp module = builder.getModule();
  mlir::OpBuilder modBuilder(module.getBodyRegion());

  mlir::omp::ReductionDeclareOp decl =
      modBuilder.create<mlir::omp::ReductionDeclareOp>(loc, reductionOpName,
                                                       type);
  builder.createBlock(&decl.getInitializerRegion(),
                      decl.getInitializerRegion().end(), {type}, {loc});
  builder.setInsertionPointToEnd(&decl.getInitializerRegion().back());
  mlir::Value init = getReductionInitValue(loc, type, reductionOpName, builder);
  builder.create<mlir::omp::YieldOp>(loc, init);

  builder.createBlock(&decl.getReductionRegion(),
                      decl.getReductionRegion().end(), {type, type},
                      {loc, loc});

  return decl;
}

/// Creates an OpenMP reduction declaration and inserts it into the provided
/// symbol table. The declaration has a constant initializer with the neutral
/// value `initValue`, and the reduction combiner carried over from `reduce`.
/// TODO: Generalize this for non-integer types, add atomic region.
static mlir::omp::ReductionDeclareOp
createReductionDecl(fir::FirOpBuilder &builder, llvm::StringRef reductionOpName,
                    const Fortran::parser::ProcedureDesignator &procDesignator,
                    mlir::Type type, mlir::Location loc) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::ModuleOp module = builder.getModule();

  auto decl =
      module.lookupSymbol<mlir::omp::ReductionDeclareOp>(reductionOpName);
  if (decl)
    return decl;

  decl = createMinimalReductionDecl(builder, reductionOpName, type, loc);
  builder.setInsertionPointToEnd(&decl.getReductionRegion().back());
  mlir::Value op1 = decl.getReductionRegion().front().getArgument(0);
  mlir::Value op2 = decl.getReductionRegion().front().getArgument(1);

  mlir::Value reductionOp;
  if (const auto *name{
          Fortran::parser::Unwrap<Fortran::parser::Name>(procDesignator)}) {
    if (name->source == "max") {
      reductionOp =
          getReductionOperation<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp>(
              builder, type, loc, op1, op2);
    } else if (name->source == "min") {
      reductionOp =
          getReductionOperation<mlir::arith::MinimumFOp, mlir::arith::MinSIOp>(
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

  builder.create<mlir::omp::YieldOp>(loc, reductionOp);
  return decl;
}

/// Creates an OpenMP reduction declaration and inserts it into the provided
/// symbol table. The declaration has a constant initializer with the neutral
/// value `initValue`, and the reduction combiner carried over from `reduce`.
/// TODO: Generalize this for non-integer types, add atomic region.
static mlir::omp::ReductionDeclareOp createReductionDecl(
    fir::FirOpBuilder &builder, llvm::StringRef reductionOpName,
    Fortran::parser::DefinedOperator::IntrinsicOperator intrinsicOp,
    mlir::Type type, mlir::Location loc) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::ModuleOp module = builder.getModule();

  auto decl =
      module.lookupSymbol<mlir::omp::ReductionDeclareOp>(reductionOpName);
  if (decl)
    return decl;

  decl = createMinimalReductionDecl(builder, reductionOpName, type, loc);
  builder.setInsertionPointToEnd(&decl.getReductionRegion().back());
  mlir::Value op1 = decl.getReductionRegion().front().getArgument(0);
  mlir::Value op2 = decl.getReductionRegion().front().getArgument(1);

  mlir::Value reductionOp;
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
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value andiOp = builder.create<mlir::arith::AndIOp>(loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, andiOp);
    break;
  }
  case Fortran::parser::DefinedOperator::IntrinsicOperator::OR: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value oriOp = builder.create<mlir::arith::OrIOp>(loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, oriOp);
    break;
  }
  case Fortran::parser::DefinedOperator::IntrinsicOperator::EQV: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value cmpiOp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  case Fortran::parser::DefinedOperator::IntrinsicOperator::NEQV: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value cmpiOp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  default:
    TODO(loc, "Reduction of some intrinsic operators is not supported");
  }

  builder.create<mlir::omp::YieldOp>(loc, reductionOp);
  return decl;
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
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::OmpClause::If *ifClause,
    Fortran::parser::OmpIfClause::DirectiveNameModifier directiveName,
    mlir::Location clauseLocation) {
  // Only consider the clause if it's intended for the given directive.
  auto &directive = std::get<
      std::optional<Fortran::parser::OmpIfClause::DirectiveNameModifier>>(
      ifClause->v.t);
  if (directive && directive.value() != directiveName)
    return nullptr;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->v.t);
  mlir::Value ifVal = fir::getBase(
      converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
  return firOpBuilder.createConvert(clauseLocation, firOpBuilder.getI1Type(),
                                    ifVal);
}

/// Creates a reduction declaration and associates it with an OpenMP block
/// directive.
static void
addReductionDecl(mlir::Location currentLocation,
                 Fortran::lower::AbstractConverter &converter,
                 const Fortran::parser::OmpReductionClause &reduction,
                 llvm::SmallVectorImpl<mlir::Value> &reductionVars,
                 llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::omp::ReductionDeclareOp decl;
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
    for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
      if (const auto *name{
              Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
        if (const Fortran::semantics::Symbol * symbol{name->symbol}) {
          mlir::Value symVal = converter.getSymbolAddress(*symbol);
          if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>())
            symVal = declOp.getBase();
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
          reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
              firOpBuilder.getContext(), decl.getSymName()));
        }
      }
    }
  } else if (const auto *reductionIntrinsic =
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
      for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
        if (const auto *name{
                Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
          if (const Fortran::semantics::Symbol * symbol{name->symbol}) {
            mlir::Value symVal = converter.getSymbolAddress(*symbol);
            if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>())
              symVal = declOp.getBase();
            mlir::Type redType =
                symVal.getType().cast<fir::ReferenceType>().getEleTy();
            reductionVars.push_back(symVal);
            assert(redType.isIntOrIndexOrFloat() &&
                   "Unsupported reduction type");
            decl = createReductionDecl(
                firOpBuilder, getReductionName(intrinsicOp, redType),
                *reductionIntrinsic, redType, currentLocation);
            reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
                firOpBuilder.getContext(), decl.getSymName()));
          }
        }
      }
    }
  }
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
    Fortran::lower::StatementContext &stmtCtx,
    Fortran::parser::OmpIfClause::DirectiveNameModifier directiveName,
    mlir::Value &result) const {
  bool found = false;
  findRepeatableClause<ClauseTy::If>(
      [&](const ClauseTy::If *ifClause,
          const Fortran::parser::CharBlock &source) {
        mlir::Location clauseLocation = converter.genLocation(source);
        mlir::Value operand = getIfClauseOperand(converter, stmtCtx, ifClause,
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

bool ClauseProcessor::processMap(
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<mlir::IntegerAttr> &mapTypes) const {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  return findRepeatableClause<
      ClauseTy::Map>([&](const ClauseTy::Map *mapClause,
                         const Fortran::parser::CharBlock &source) {
    mlir::Location clauseLocation = converter.genLocation(source);
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
    // Check for unsupported map operand types.
    for (const Fortran::parser::OmpObject &ompObject :
         std::get<Fortran::parser::OmpObjectList>(mapClause->v.t).v) {
      if (Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(ompObject) ||
          Fortran::parser::Unwrap<Fortran::parser::StructureComponent>(
              ompObject))
        TODO(clauseLocation,
             "OMPD_target_data for Array Expressions or Structure Components");
    }
    genObjectList(std::get<Fortran::parser::OmpObjectList>(mapClause->v.t),
                  converter, mapOperand);

    for (mlir::Value mapOp : mapOperand) {
      checkMapType(mapOp.getLoc(), mapOp.getType());
      mapOperands.push_back(mapOp);
      mapTypes.push_back(mapTypeAttr);
    }
  });
}

bool ClauseProcessor::processReduction(
    mlir::Location currentLocation,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols) const {
  return findRepeatableClause<ClauseTy::Reduction>(
      [&](const ClauseTy::Reduction *reductionClause,
          const Fortran::parser::CharBlock &) {
        addReductionDecl(currentLocation, converter, reductionClause->v,
                         reductionVars, reductionDeclSymbols);
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

  for (ClauseIterator it = clauses.v.begin(); it != clauses.v.end(); ++it)
    (checkUnhandledClause(std::get_if<Ts>(&it->u)), ...);
}

//===----------------------------------------------------------------------===//
// Code generation helper functions
//===----------------------------------------------------------------------===//

static fir::GlobalOp globalInitialization(
    Fortran::lower::AbstractConverter &converter,
    fir::FirOpBuilder &firOpBuilder, const Fortran::semantics::Symbol &sym,
    const Fortran::lower::pft::Variable &var, mlir::Location currentLocation) {
  mlir::Type ty = converter.genType(sym);
  std::string globalName = converter.mangleName(sym);
  mlir::StringAttr linkage = firOpBuilder.createInternalLinkage();
  fir::GlobalOp global =
      firOpBuilder.createGlobal(currentLocation, ty, globalName, linkage);

  // Create default initialization for non-character scalar.
  if (Fortran::semantics::IsAllocatableOrObjectPointer(&sym)) {
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

static mlir::Operation *getCompareFromReductionOp(mlir::Operation *reductionOp,
                                                  mlir::Value loadVal) {
  for (mlir::Value reductionOperand : reductionOp->getOperands()) {
    if (mlir::Operation *compareOp = reductionOperand.getDefiningOp()) {
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

/// The COMMON block is a global structure. \p commonValue is the base address
/// of the COMMON block. As the offset from the symbol \p sym, generate the
/// COMMON block member value (commonValue + offset) for the symbol.
/// FIXME: Share the code with `instantiateCommon` in ConvertVariable.cpp.
static mlir::Value
genCommonBlockMember(Fortran::lower::AbstractConverter &converter,
                     const Fortran::semantics::Symbol &sym,
                     mlir::Value commonValue) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
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
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  // Get the original ThreadprivateOp corresponding to the symbol and use the
  // symbol value from that operation to create one ThreadprivateOp copy
  // operation inside the parallel region.
  auto genThreadprivateOp = [&](Fortran::lower::SymbolRef sym) -> mlir::Value {
    mlir::Value symOriThreadprivateValue = converter.getSymbolAddress(sym);
    mlir::Operation *op = symOriThreadprivateValue.getDefiningOp();
    if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(op))
      op = declOp.getMemref().getDefiningOp();
    assert(mlir::isa<mlir::omp::ThreadprivateOp>(op) &&
           "Threadprivate operation not created");
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
    const Fortran::semantics::Symbol *sym = threadprivateSyms[i];
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

static void resetBeforeTerminator(fir::FirOpBuilder &firOpBuilder,
                                  mlir::Operation *storeOp,
                                  mlir::Block &block) {
  if (storeOp)
    firOpBuilder.setInsertionPointAfter(storeOp);
  else
    firOpBuilder.setInsertionPointToStart(&block);
}

static mlir::Operation *
createAndSetPrivatizedLoopVar(Fortran::lower::AbstractConverter &converter,
                              mlir::Location loc, mlir::Value indexVal,
                              const Fortran::semantics::Symbol *sym) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  mlir::Type tempTy = converter.genType(*sym);
  mlir::Value temp = firOpBuilder.create<fir::AllocaOp>(
      loc, tempTy, /*pinned=*/true, /*lengthParams=*/mlir::ValueRange{},
      /*shapeParams*/ mlir::ValueRange{},
      llvm::ArrayRef<mlir::NamedAttribute>{
          Fortran::lower::getAdaptToByRefAttr(firOpBuilder)});
  converter.bindSymbol(*sym, temp);
  firOpBuilder.restoreInsertionPoint(insPt);
  mlir::Value cvtVal = firOpBuilder.createConvert(loc, tempTy, indexVal);
  mlir::Operation *storeOp = firOpBuilder.create<fir::StoreOp>(
      loc, cvtVal, converter.getSymbolAddress(*sym));
  return storeOp;
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
static void createBodyOfOp(
    Op &op, Fortran::lower::AbstractConverter &converter, mlir::Location &loc,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OmpClauseList *clauses = nullptr,
    const llvm::SmallVector<const Fortran::semantics::Symbol *> &args = {},
    bool outerCombined = false, DataSharingProcessor *dsp = nullptr) {
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
    llvm::SmallVector<mlir::Type> tiv;
    llvm::SmallVector<mlir::Location> locs;
    for (int i = 0; i < (int)args.size(); i++) {
      tiv.push_back(loopVarType);
      locs.push_back(loc);
    }
    firOpBuilder.createBlock(&op.getRegion(), {}, tiv, locs);
    int argIndex = 0;
    // The argument is not currently in memory, so make a temporary for the
    // argument, and store it there, then bind that location to the argument.
    for (const Fortran::semantics::Symbol *arg : args) {
      mlir::Value indexVal =
          fir::getBase(op.getRegion().front().getArgument(argIndex));
      storeOp = createAndSetPrivatizedLoopVar(converter, loc, indexVal, arg);
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
    Fortran::lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp,
                                            mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());

  // Insert the terminator.
  if constexpr (std::is_same_v<Op, mlir::omp::WsLoopOp> ||
                std::is_same_v<Op, mlir::omp::SimdLoopOp>) {
    mlir::ValueRange results;
    firOpBuilder.create<mlir::omp::YieldOp>(loc, results);
  } else {
    firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
  }
  // Reset the insert point to before the terminator.
  resetBeforeTerminator(firOpBuilder, storeOp, block);

  // Handle privatization. Do not privatize if this is the outer operation.
  if (clauses && !outerCombined) {
    constexpr bool is_loop = std::is_same_v<Op, mlir::omp::WsLoopOp> ||
                             std::is_same_v<Op, mlir::omp::SimdLoopOp>;
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

  if constexpr (std::is_same_v<Op, mlir::omp::ParallelOp>) {
    threadPrivatizeVars(converter, eval);
    if (clauses)
      ClauseProcessor(converter, *clauses).processCopyin();
  }
}

static void createBodyOfTargetDataOp(
    Fortran::lower::AbstractConverter &converter, mlir::omp::DataOp &dataOp,
    const llvm::SmallVector<mlir::Type> &useDeviceTypes,
    const llvm::SmallVector<mlir::Location> &useDeviceLocs,
    const llvm::SmallVector<const Fortran::semantics::Symbol *>
        &useDeviceSymbols,
    const mlir::Location &currentLocation) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Region &region = dataOp.getRegion();

  firOpBuilder.createBlock(&region, {}, useDeviceTypes, useDeviceLocs);
  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);
  firOpBuilder.setInsertionPointToStart(&region.front());

  unsigned argIndex = 0;
  for (const Fortran::semantics::Symbol *sym : useDeviceSymbols) {
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

template <typename OpTy, typename... Args>
static OpTy genOpWithBody(Fortran::lower::AbstractConverter &converter,
                          Fortran::lower::pft::Evaluation &eval,
                          mlir::Location currentLocation, bool outerCombined,
                          const Fortran::parser::OmpClauseList *clauseList,
                          Args &&...args) {
  auto op = converter.getFirOpBuilder().create<OpTy>(
      currentLocation, std::forward<Args>(args)...);
  createBodyOfOp<OpTy>(op, converter, currentLocation, eval, clauseList,
                       /*args=*/{}, outerCombined);
  return op;
}

static mlir::omp::MasterOp
genMasterOp(Fortran::lower::AbstractConverter &converter,
            Fortran::lower::pft::Evaluation &eval,
            mlir::Location currentLocation) {
  return genOpWithBody<mlir::omp::MasterOp>(converter, eval, currentLocation,
                                            /*outerCombined=*/false,
                                            /*clauseList=*/nullptr,
                                            /*resultTypes=*/mlir::TypeRange());
}

static mlir::omp::OrderedRegionOp
genOrderedRegionOp(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   mlir::Location currentLocation) {
  return genOpWithBody<mlir::omp::OrderedRegionOp>(
      converter, eval, currentLocation, /*outerCombined=*/false,
      /*clauseList=*/nullptr, /*simd=*/false);
}

static mlir::omp::ParallelOp
genParallelOp(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              mlir::Location currentLocation,
              const Fortran::parser::OmpClauseList &clauseList,
              bool outerCombined = false) {
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, numThreadsClauseOperand;
  mlir::omp::ClauseProcBindKindAttr procBindKindAttr;
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands,
      reductionVars;
  llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;

  ClauseProcessor cp(converter, clauseList);
  cp.processIf(stmtCtx,
               Fortran::parser::OmpIfClause::DirectiveNameModifier::Parallel,
               ifClauseOperand);
  cp.processNumThreads(stmtCtx, numThreadsClauseOperand);
  cp.processProcBind(procBindKindAttr);
  cp.processDefault();
  cp.processAllocate(allocatorOperands, allocateOperands);
  cp.processReduction(currentLocation, reductionVars, reductionDeclSymbols);

  return genOpWithBody<mlir::omp::ParallelOp>(
      converter, eval, currentLocation, outerCombined, &clauseList,
      /*resultTypes=*/mlir::TypeRange(), ifClauseOperand,
      numThreadsClauseOperand, allocateOperands, allocatorOperands,
      reductionVars,
      reductionDeclSymbols.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 reductionDeclSymbols),
      procBindKindAttr);
}

static mlir::omp::SingleOp
genSingleOp(Fortran::lower::AbstractConverter &converter,
            Fortran::lower::pft::Evaluation &eval,
            mlir::Location currentLocation,
            const Fortran::parser::OmpClauseList &beginClauseList,
            const Fortran::parser::OmpClauseList &endClauseList) {
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands;
  mlir::UnitAttr nowaitAttr;

  ClauseProcessor cp(converter, beginClauseList);
  cp.processAllocate(allocatorOperands, allocateOperands);
  cp.processTODO<Fortran::parser::OmpClause::Copyprivate>(
      currentLocation, llvm::omp::Directive::OMPD_single);

  ClauseProcessor(converter, endClauseList).processNowait(nowaitAttr);

  return genOpWithBody<mlir::omp::SingleOp>(
      converter, eval, currentLocation, /*outerCombined=*/false,
      &beginClauseList, allocateOperands, allocatorOperands, nowaitAttr);
}

static mlir::omp::TaskOp
genTaskOp(Fortran::lower::AbstractConverter &converter,
          Fortran::lower::pft::Evaluation &eval, mlir::Location currentLocation,
          const Fortran::parser::OmpClauseList &clauseList) {
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, finalClauseOperand, priorityClauseOperand;
  mlir::UnitAttr untiedAttr, mergeableAttr;
  llvm::SmallVector<mlir::Attribute> dependTypeOperands;
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands,
      dependOperands;

  ClauseProcessor cp(converter, clauseList);
  cp.processIf(stmtCtx,
               Fortran::parser::OmpIfClause::DirectiveNameModifier::Task,
               ifClauseOperand);
  cp.processAllocate(allocatorOperands, allocateOperands);
  cp.processDefault();
  cp.processFinal(stmtCtx, finalClauseOperand);
  cp.processUntied(untiedAttr);
  cp.processMergeable(mergeableAttr);
  cp.processPriority(stmtCtx, priorityClauseOperand);
  cp.processDepend(dependTypeOperands, dependOperands);
  cp.processTODO<Fortran::parser::OmpClause::InReduction,
                 Fortran::parser::OmpClause::Detach,
                 Fortran::parser::OmpClause::Affinity>(
      currentLocation, llvm::omp::Directive::OMPD_task);

  return genOpWithBody<mlir::omp::TaskOp>(
      converter, eval, currentLocation, /*outerCombined=*/false, &clauseList,
      ifClauseOperand, finalClauseOperand, untiedAttr, mergeableAttr,
      /*in_reduction_vars=*/mlir::ValueRange(),
      /*in_reductions=*/nullptr, priorityClauseOperand,
      dependTypeOperands.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 dependTypeOperands),
      dependOperands, allocateOperands, allocatorOperands);
}

static mlir::omp::TaskGroupOp
genTaskGroupOp(Fortran::lower::AbstractConverter &converter,
               Fortran::lower::pft::Evaluation &eval,
               mlir::Location currentLocation,
               const Fortran::parser::OmpClauseList &clauseList) {
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands;
  ClauseProcessor cp(converter, clauseList);
  cp.processAllocate(allocatorOperands, allocateOperands);
  cp.processTODO<Fortran::parser::OmpClause::TaskReduction>(
      currentLocation, llvm::omp::Directive::OMPD_taskgroup);
  return genOpWithBody<mlir::omp::TaskGroupOp>(
      converter, eval, currentLocation, /*outerCombined=*/false, &clauseList,
      /*task_reduction_vars=*/mlir::ValueRange(),
      /*task_reductions=*/nullptr, allocateOperands, allocatorOperands);
}

static mlir::omp::DataOp
genDataOp(Fortran::lower::AbstractConverter &converter,
          mlir::Location currentLocation,
          const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, deviceOperand;
  llvm::SmallVector<mlir::Value> mapOperands, devicePtrOperands,
      deviceAddrOperands;
  llvm::SmallVector<mlir::IntegerAttr> mapTypes;
  llvm::SmallVector<mlir::Type> useDeviceTypes;
  llvm::SmallVector<mlir::Location> useDeviceLocs;
  llvm::SmallVector<const Fortran::semantics::Symbol *> useDeviceSymbols;

  ClauseProcessor cp(converter, clauseList);
  cp.processIf(stmtCtx,
               Fortran::parser::OmpIfClause::DirectiveNameModifier::TargetData,
               ifClauseOperand);
  cp.processDevice(stmtCtx, deviceOperand);
  cp.processUseDevicePtr(devicePtrOperands, useDeviceTypes, useDeviceLocs,
                         useDeviceSymbols);
  cp.processUseDeviceAddr(deviceAddrOperands, useDeviceTypes, useDeviceLocs,
                          useDeviceSymbols);
  cp.processMap(mapOperands, mapTypes);

  llvm::SmallVector<mlir::Attribute> mapTypesAttr(mapTypes.begin(),
                                                  mapTypes.end());
  mlir::ArrayAttr mapTypesArrayAttr =
      mlir::ArrayAttr::get(firOpBuilder.getContext(), mapTypesAttr);

  auto dataOp = converter.getFirOpBuilder().create<mlir::omp::DataOp>(
      currentLocation, ifClauseOperand, deviceOperand, devicePtrOperands,
      deviceAddrOperands, mapOperands, mapTypesArrayAttr);
  createBodyOfTargetDataOp(converter, dataOp, useDeviceTypes, useDeviceLocs,
                           useDeviceSymbols, currentLocation);
  return dataOp;
}

template <typename OpTy>
static OpTy
genEnterExitDataOp(Fortran::lower::AbstractConverter &converter,
                   mlir::Location currentLocation,
                   const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, deviceOperand;
  mlir::UnitAttr nowaitAttr;
  llvm::SmallVector<mlir::Value> mapOperands;
  llvm::SmallVector<mlir::IntegerAttr> mapTypes;

  Fortran::parser::OmpIfClause::DirectiveNameModifier directiveName;
  llvm::omp::Directive directive;
  if constexpr (std::is_same_v<OpTy, mlir::omp::EnterDataOp>) {
    directiveName =
        Fortran::parser::OmpIfClause::DirectiveNameModifier::TargetEnterData;
    directive = llvm::omp::Directive::OMPD_target_enter_data;
  } else if constexpr (std::is_same_v<OpTy, mlir::omp::ExitDataOp>) {
    directiveName =
        Fortran::parser::OmpIfClause::DirectiveNameModifier::TargetExitData;
    directive = llvm::omp::Directive::OMPD_target_exit_data;
  } else {
    return nullptr;
  }

  ClauseProcessor cp(converter, clauseList);
  cp.processIf(stmtCtx, directiveName, ifClauseOperand);
  cp.processDevice(stmtCtx, deviceOperand);
  cp.processNowait(nowaitAttr);
  cp.processMap(mapOperands, mapTypes);
  cp.processTODO<Fortran::parser::OmpClause::Depend>(currentLocation,
                                                     directive);

  llvm::SmallVector<mlir::Attribute> mapTypesAttr(mapTypes.begin(),
                                                  mapTypes.end());
  mlir::ArrayAttr mapTypesArrayAttr =
      mlir::ArrayAttr::get(firOpBuilder.getContext(), mapTypesAttr);

  return firOpBuilder.create<OpTy>(currentLocation, ifClauseOperand,
                                   deviceOperand, nowaitAttr, mapOperands,
                                   mapTypesArrayAttr);
}

static mlir::omp::TargetOp
genTargetOp(Fortran::lower::AbstractConverter &converter,
            Fortran::lower::pft::Evaluation &eval,
            mlir::Location currentLocation,
            const Fortran::parser::OmpClauseList &clauseList,
            bool outerCombined = false) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, deviceOperand, threadLimitOperand;
  mlir::UnitAttr nowaitAttr;
  llvm::SmallVector<mlir::Value> mapOperands;
  llvm::SmallVector<mlir::IntegerAttr> mapTypes;

  ClauseProcessor cp(converter, clauseList);
  cp.processIf(stmtCtx,
               Fortran::parser::OmpIfClause::DirectiveNameModifier::Target,
               ifClauseOperand);
  cp.processDevice(stmtCtx, deviceOperand);
  cp.processThreadLimit(stmtCtx, threadLimitOperand);
  cp.processNowait(nowaitAttr);
  cp.processMap(mapOperands, mapTypes);
  cp.processTODO<Fortran::parser::OmpClause::Private,
                 Fortran::parser::OmpClause::Depend,
                 Fortran::parser::OmpClause::Firstprivate,
                 Fortran::parser::OmpClause::IsDevicePtr,
                 Fortran::parser::OmpClause::HasDeviceAddr,
                 Fortran::parser::OmpClause::Reduction,
                 Fortran::parser::OmpClause::InReduction,
                 Fortran::parser::OmpClause::Allocate,
                 Fortran::parser::OmpClause::UsesAllocators,
                 Fortran::parser::OmpClause::Defaultmap>(
      currentLocation, llvm::omp::Directive::OMPD_target);

  llvm::SmallVector<mlir::Attribute> mapTypesAttr(mapTypes.begin(),
                                                  mapTypes.end());
  mlir::ArrayAttr mapTypesArrayAttr =
      mlir::ArrayAttr::get(firOpBuilder.getContext(), mapTypesAttr);

  return genOpWithBody<mlir::omp::TargetOp>(
      converter, eval, currentLocation, outerCombined, &clauseList,
      ifClauseOperand, deviceOperand, threadLimitOperand, nowaitAttr,
      mapOperands, mapTypesArrayAttr);
}

static mlir::omp::TeamsOp
genTeamsOp(Fortran::lower::AbstractConverter &converter,
           Fortran::lower::pft::Evaluation &eval,
           mlir::Location currentLocation,
           const Fortran::parser::OmpClauseList &clauseList,
           bool outerCombined = false) {
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value numTeamsClauseOperand, ifClauseOperand, threadLimitClauseOperand;
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands,
      reductionVars;
  llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;

  ClauseProcessor cp(converter, clauseList);
  cp.processIf(stmtCtx,
               Fortran::parser::OmpIfClause::DirectiveNameModifier::Teams,
               ifClauseOperand);
  cp.processAllocate(allocatorOperands, allocateOperands);
  cp.processDefault();
  cp.processNumTeams(stmtCtx, numTeamsClauseOperand);
  cp.processThreadLimit(stmtCtx, threadLimitClauseOperand);
  cp.processTODO<Fortran::parser::OmpClause::Reduction>(
      currentLocation, llvm::omp::Directive::OMPD_teams);

  return genOpWithBody<mlir::omp::TeamsOp>(
      converter, eval, currentLocation, outerCombined, &clauseList,
      /*num_teams_lower=*/nullptr, numTeamsClauseOperand, ifClauseOperand,
      threadLimitClauseOperand, allocateOperands, allocatorOperands,
      reductionVars,
      reductionDeclSymbols.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 reductionDeclSymbols));
}

/// Extract the list of function and variable symbols affected by the given
/// 'declare target' directive and return the intended device type for them.
static mlir::omp::DeclareTargetDeviceType getDeclareTargetInfo(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclareTargetConstruct &declareTargetConstruct,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {

  // The default capture type
  mlir::omp::DeclareTargetDeviceType deviceType =
      mlir::omp::DeclareTargetDeviceType::any;
  const auto &spec = std::get<Fortran::parser::OmpDeclareTargetSpecifier>(
      declareTargetConstruct.t);

  if (const auto *objectList{
          Fortran::parser::Unwrap<Fortran::parser::OmpObjectList>(spec.u)}) {
    // Case: declare target(func, var1, var2)
    gatherFuncAndVarSyms(*objectList, mlir::omp::DeclareTargetCaptureClause::to,
                         symbolAndClause);
  } else if (const auto *clauseList{
                 Fortran::parser::Unwrap<Fortran::parser::OmpClauseList>(
                     spec.u)}) {
    if (clauseList->v.empty()) {
      // Case: declare target, implicit capture of function
      symbolAndClause.emplace_back(
          mlir::omp::DeclareTargetCaptureClause::to,
          eval.getOwningProcedure()->getSubprogramSymbol());
    }

    ClauseProcessor cp(converter, *clauseList);
    cp.processTo(symbolAndClause);
    cp.processLink(symbolAndClause);
    cp.processDeviceType(deviceType);
    cp.processTODO<Fortran::parser::OmpClause::Indirect>(
        converter.getCurrentLocation(),
        llvm::omp::Directive::OMPD_declare_target);
  }

  return deviceType;
}

static std::optional<mlir::omp::DeclareTargetDeviceType>
getDeclareTargetFunctionDevice(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclareTargetConstruct
        &declareTargetConstruct) {
  llvm::SmallVector<DeclareTargetCapturePair, 0> symbolAndClause;
  mlir::omp::DeclareTargetDeviceType deviceType = getDeclareTargetInfo(
      converter, eval, declareTargetConstruct, symbolAndClause);

  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(
        converter.mangleName(std::get<Fortran::semantics::Symbol>(symClause)));

    if (mlir::isa<mlir::func::FuncOp>(op))
      return deviceType;
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// genOMP() Code generation helper functions
//===----------------------------------------------------------------------===//

static void
genOmpSimpleStandalone(Fortran::lower::AbstractConverter &converter,
                       Fortran::lower::pft::Evaluation &eval,
                       const Fortran::parser::OpenMPSimpleStandaloneConstruct
                           &simpleStandaloneConstruct) {
  const auto &directive =
      std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
          simpleStandaloneConstruct.t);
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(simpleStandaloneConstruct.t);
  mlir::Location currentLocation = converter.genLocation(directive.source);

  switch (directive.v) {
  default:
    break;
  case llvm::omp::Directive::OMPD_barrier:
    firOpBuilder.create<mlir::omp::BarrierOp>(currentLocation);
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    ClauseProcessor(converter, opClauseList)
        .processTODO<Fortran::parser::OmpClause::Depend,
                     Fortran::parser::OmpClause::Nowait>(
            currentLocation, llvm::omp::Directive::OMPD_taskwait);
    firOpBuilder.create<mlir::omp::TaskwaitOp>(currentLocation);
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    firOpBuilder.create<mlir::omp::TaskyieldOp>(currentLocation);
    break;
  case llvm::omp::Directive::OMPD_target_data:
    genDataOp(converter, currentLocation, opClauseList);
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    genEnterExitDataOp<mlir::omp::EnterDataOp>(converter, currentLocation,
                                               opClauseList);
    break;
  case llvm::omp::Directive::OMPD_target_exit_data:
    genEnterExitDataOp<mlir::omp::ExitDataOp>(converter, currentLocation,
                                              opClauseList);
    break;
  case llvm::omp::Directive::OMPD_target_update:
    TODO(currentLocation, "OMPD_target_update");
  case llvm::omp::Directive::OMPD_ordered:
    TODO(currentLocation, "OMPD_ordered");
  }
}

static void
genOmpFlush(Fortran::lower::AbstractConverter &converter,
            Fortran::lower::pft::Evaluation &eval,
            const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
  llvm::SmallVector<mlir::Value, 4> operandRange;
  if (const auto &ompObjectList =
          std::get<std::optional<Fortran::parser::OmpObjectList>>(
              flushConstruct.t))
    genObjectList(*ompObjectList, converter, operandRange);
  const auto &memOrderClause =
      std::get<std::optional<std::list<Fortran::parser::OmpMemoryOrderClause>>>(
          flushConstruct.t);
  if (memOrderClause && memOrderClause->size() > 0)
    TODO(converter.getCurrentLocation(), "Handle OmpMemoryOrderClause");
  converter.getFirOpBuilder().create<mlir::omp::FlushOp>(
      converter.getCurrentLocation(), operandRange);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPSimpleStandaloneConstruct
                  &simpleStandaloneConstruct) {
            genOmpSimpleStandalone(converter, eval, simpleStandaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
            genOmpFlush(converter, eval, flushConstruct);
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

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Value> lowerBound, upperBound, step, linearVars,
      linearStepVars, reductionVars;
  mlir::Value scheduleChunkClauseOperand;
  mlir::IntegerAttr orderedClauseOperand;
  mlir::omp::ClauseOrderKindAttr orderClauseOperand;
  mlir::omp::ClauseScheduleKindAttr scheduleValClauseOperand;
  mlir::omp::ScheduleModifierAttr scheduleModClauseOperand;
  mlir::UnitAttr nowaitClauseOperand, scheduleSimdClauseOperand;
  llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;
  Fortran::lower::StatementContext stmtCtx;
  std::size_t loopVarTypeSize;
  llvm::SmallVector<const Fortran::semantics::Symbol *> iv;

  const auto &beginLoopDirective =
      std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t);
  const auto &loopOpClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginLoopDirective.t);
  mlir::Location currentLocation =
      converter.genLocation(beginLoopDirective.source);
  const auto ompDirective =
      std::get<Fortran::parser::OmpLoopDirective>(beginLoopDirective.t).v;

  bool validDirective = false;
  if (llvm::omp::topTaskloopSet.test(ompDirective)) {
    validDirective = true;
    TODO(currentLocation, "Taskloop construct");
  } else {
    // Create omp.{target, teams, distribute, parallel} nested operations
    if ((llvm::omp::allTargetSet & llvm::omp::loopConstructSet)
            .test(ompDirective)) {
      validDirective = true;
      genTargetOp(converter, eval, currentLocation, loopOpClauseList,
                  /*outerCombined=*/true);
    }
    if ((llvm::omp::allTeamsSet & llvm::omp::loopConstructSet)
            .test(ompDirective)) {
      validDirective = true;
      genTeamsOp(converter, eval, currentLocation, loopOpClauseList,
                 /*outerCombined=*/true);
    }
    if (llvm::omp::allDistributeSet.test(ompDirective)) {
      validDirective = true;
      TODO(currentLocation, "Distribute construct");
    }
    if ((llvm::omp::allParallelSet & llvm::omp::loopConstructSet)
            .test(ompDirective)) {
      validDirective = true;
      genParallelOp(converter, eval, currentLocation, loopOpClauseList,
                    /*outerCombined=*/true);
    }
  }
  if ((llvm::omp::allDoSet | llvm::omp::allSimdSet).test(ompDirective))
    validDirective = true;

  if (!validDirective) {
    TODO(currentLocation, "Unhandled loop directive (" +
                              llvm::omp::getOpenMPDirectiveName(ompDirective) +
                              ")");
  }

  DataSharingProcessor dsp(converter, loopOpClauseList, eval);
  dsp.processStep1();

  ClauseProcessor cp(converter, loopOpClauseList);
  cp.processCollapse(currentLocation, eval, lowerBound, upperBound, step, iv,
                     loopVarTypeSize);
  cp.processScheduleChunk(stmtCtx, scheduleChunkClauseOperand);
  cp.processReduction(currentLocation, reductionVars, reductionDeclSymbols);
  cp.processTODO<Fortran::parser::OmpClause::Linear,
                 Fortran::parser::OmpClause::Order>(currentLocation,
                                                    ompDirective);

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
  if (llvm::omp::allSimdSet.test(ompDirective)) {
    llvm::SmallVector<mlir::Value> alignedVars, nontemporalVars;
    mlir::Value ifClauseOperand;
    mlir::IntegerAttr simdlenClauseOperand, safelenClauseOperand;
    cp.processIf(stmtCtx,
                 Fortran::parser::OmpIfClause::DirectiveNameModifier::Simd,
                 ifClauseOperand);
    cp.processSimdlen(simdlenClauseOperand);
    cp.processSafelen(safelenClauseOperand);
    cp.processTODO<Fortran::parser::OmpClause::Aligned,
                   Fortran::parser::OmpClause::Allocate,
                   Fortran::parser::OmpClause::Nontemporal>(currentLocation,
                                                            ompDirective);

    mlir::TypeRange resultType;
    auto simdLoopOp = firOpBuilder.create<mlir::omp::SimdLoopOp>(
        currentLocation, resultType, lowerBound, upperBound, step, alignedVars,
        /*alignment_values=*/nullptr, ifClauseOperand, nontemporalVars,
        orderClauseOperand, simdlenClauseOperand, safelenClauseOperand,
        /*inclusive=*/firOpBuilder.getUnitAttr());
    createBodyOfOp<mlir::omp::SimdLoopOp>(
        simdLoopOp, converter, currentLocation, eval, &loopOpClauseList, iv,
        /*outer=*/false, &dsp);
    return;
  }

  auto wsLoopOp = firOpBuilder.create<mlir::omp::WsLoopOp>(
      currentLocation, lowerBound, upperBound, step, linearVars, linearStepVars,
      reductionVars,
      reductionDeclSymbols.empty()
          ? nullptr
          : mlir::ArrayAttr::get(firOpBuilder.getContext(),
                                 reductionDeclSymbols),
      scheduleValClauseOperand, scheduleChunkClauseOperand,
      /*schedule_modifiers=*/nullptr,
      /*simd_modifier=*/nullptr, nowaitClauseOperand, orderedClauseOperand,
      orderClauseOperand,
      /*inclusive=*/firOpBuilder.getUnitAttr());

  // Handle attribute based clauses.
  if (cp.processOrdered(orderedClauseOperand))
    wsLoopOp.setOrderedValAttr(orderedClauseOperand);

  if (cp.processSchedule(scheduleValClauseOperand, scheduleModClauseOperand,
                         scheduleSimdClauseOperand)) {
    wsLoopOp.setScheduleValAttr(scheduleValClauseOperand);
    wsLoopOp.setScheduleModifierAttr(scheduleModClauseOperand);
    wsLoopOp.setSimdModifierAttr(scheduleSimdClauseOperand);
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
    if (ClauseProcessor(converter, clauseList)
            .processNowait(nowaitClauseOperand))
      wsLoopOp.setNowaitAttr(nowaitClauseOperand);
  }

  createBodyOfOp<mlir::omp::WsLoopOp>(wsLoopOp, converter, currentLocation,
                                      eval, &loopOpClauseList, iv,
                                      /*outer=*/false, &dsp);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &endBlockDirective =
      std::get<Fortran::parser::OmpEndBlockDirective>(blockConstruct.t);
  const auto &directive =
      std::get<Fortran::parser::OmpBlockDirective>(beginBlockDirective.t);
  const auto &beginClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginBlockDirective.t);
  const auto &endClauseList =
      std::get<Fortran::parser::OmpClauseList>(endBlockDirective.t);

  for (const Fortran::parser::OmpClause &clause : beginClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (!std::get_if<Fortran::parser::OmpClause::If>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::NumThreads>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::ProcBind>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Allocate>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Default>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Final>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Untied>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Mergeable>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Priority>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Reduction>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Depend>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Private>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Firstprivate>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Copyin>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Shared>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Threads>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Map>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::UseDevicePtr>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::UseDeviceAddr>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::ThreadLimit>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::NumTeams>(&clause.u)) {
      TODO(clauseLocation, "OpenMP Block construct clause");
    }
  }

  for (const auto &clause : endClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (!std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u))
      TODO(clauseLocation, "OpenMP Block construct clause");
  }

  mlir::Location currentLocation = converter.genLocation(directive.source);
  switch (directive.v) {
  case llvm::omp::Directive::OMPD_master:
    genMasterOp(converter, eval, currentLocation);
    break;
  case llvm::omp::Directive::OMPD_ordered:
    genOrderedRegionOp(converter, eval, currentLocation);
    break;
  case llvm::omp::Directive::OMPD_parallel:
    genParallelOp(converter, eval, currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_single:
    genSingleOp(converter, eval, currentLocation, beginClauseList,
                endClauseList);
    break;
  case llvm::omp::Directive::OMPD_target:
    genTargetOp(converter, eval, currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_target_data:
    genDataOp(converter, currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_task:
    genTaskOp(converter, eval, currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_taskgroup:
    genTaskGroupOp(converter, eval, currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_teams:
    genTeamsOp(converter, eval, currentLocation, beginClauseList,
               /*outerCombined=*/false);
    break;
  case llvm::omp::Directive::OMPD_workshare:
    TODO(currentLocation, "Workshare construct");
    break;
  default: {
    // Codegen for combined directives
    bool combinedDirective = false;
    if ((llvm::omp::allTargetSet & llvm::omp::blockConstructSet)
            .test(directive.v)) {
      genTargetOp(converter, eval, currentLocation, beginClauseList,
                  /*outerCombined=*/true);
      combinedDirective = true;
    }
    if ((llvm::omp::allTeamsSet & llvm::omp::blockConstructSet)
            .test(directive.v)) {
      genTeamsOp(converter, eval, currentLocation, beginClauseList);
      combinedDirective = true;
    }
    if ((llvm::omp::allParallelSet & llvm::omp::blockConstructSet)
            .test(directive.v)) {
      bool outerCombined =
          directive.v != llvm::omp::Directive::OMPD_target_parallel;
      genParallelOp(converter, eval, currentLocation, beginClauseList,
                    outerCombined);
      combinedDirective = true;
    }
    if ((llvm::omp::workShareSet & llvm::omp::blockConstructSet)
            .test(directive.v)) {
      TODO(currentLocation, "Workshare construct");
      combinedDirective = true;
    }
    if (!combinedDirective)
      TODO(currentLocation, "Unhandled block directive (" +
                                llvm::omp::getOpenMPDirectiveName(directive.v) +
                                ")");
    break;
  }
  }
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPCriticalConstruct &criticalConstruct) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::IntegerAttr hintClauseOp;
  std::string name;
  const Fortran::parser::OmpCriticalDirective &cd =
      std::get<Fortran::parser::OmpCriticalDirective>(criticalConstruct.t);
  if (std::get<std::optional<Fortran::parser::Name>>(cd.t).has_value()) {
    name =
        std::get<std::optional<Fortran::parser::Name>>(cd.t).value().ToString();
  }

  const auto &clauseList = std::get<Fortran::parser::OmpClauseList>(cd.t);
  ClauseProcessor(converter, clauseList).processHint(hintClauseOp);

  mlir::omp::CriticalOp criticalOp = [&]() {
    if (name.empty()) {
      return firOpBuilder.create<mlir::omp::CriticalOp>(
          currentLocation, mlir::FlatSymbolRefAttr());
    }
    mlir::ModuleOp module = firOpBuilder.getModule();
    mlir::OpBuilder modBuilder(module.getBodyRegion());
    auto global = module.lookupSymbol<mlir::omp::CriticalDeclareOp>(name);
    if (!global)
      global = modBuilder.create<mlir::omp::CriticalDeclareOp>(
          currentLocation,
          mlir::StringAttr::get(firOpBuilder.getContext(), name), hintClauseOp);
    return firOpBuilder.create<mlir::omp::CriticalOp>(
        currentLocation, mlir::FlatSymbolRefAttr::get(firOpBuilder.getContext(),
                                                      global.getSymName()));
  }();
  createBodyOfOp<mlir::omp::CriticalOp>(criticalOp, converter, currentLocation,
                                        eval);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPSectionConstruct &sectionConstruct) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
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
  createBodyOfOp<mlir::omp::SectionOp>(sectionOp, converter, currentLocation,
                                       eval, &sectionsClauseList);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPSectionsConstruct &sectionsConstruct) {
  mlir::Location currentLocation = converter.getCurrentLocation();
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands;
  mlir::UnitAttr nowaitClauseOperand;
  const auto &beginSectionsDirective =
      std::get<Fortran::parser::OmpBeginSectionsDirective>(sectionsConstruct.t);
  const auto &sectionsClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginSectionsDirective.t);

  // Process clauses before optional omp.parallel, so that new variables are
  // allocated outside of the parallel region
  ClauseProcessor cp(converter, sectionsClauseList);
  cp.processSectionsReduction(currentLocation);
  cp.processAllocate(allocatorOperands, allocateOperands);

  llvm::omp::Directive dir =
      std::get<Fortran::parser::OmpSectionsDirective>(beginSectionsDirective.t)
          .v;

  // Parallel wrapper of PARALLEL SECTIONS construct
  if (dir == llvm::omp::Directive::OMPD_parallel_sections) {
    genParallelOp(converter, eval, currentLocation, sectionsClauseList,
                  /*outerCombined=*/true);
  } else {
    const auto &endSectionsDirective =
        std::get<Fortran::parser::OmpEndSectionsDirective>(sectionsConstruct.t);
    const auto &endSectionsClauseList =
        std::get<Fortran::parser::OmpClauseList>(endSectionsDirective.t);
    ClauseProcessor(converter, endSectionsClauseList)
        .processNowait(nowaitClauseOperand);
  }

  // SECTIONS construct
  genOpWithBody<mlir::omp::SectionsOp>(converter, eval, currentLocation,
                                       /*outerCombined=*/false,
                                       /*clauseList=*/nullptr,
                                       /*reduction_vars=*/mlir::ValueRange(),
                                       /*reductions=*/nullptr, allocateOperands,
                                       allocatorOperands, nowaitClauseOperand);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OmpAtomicRead &atomicRead) {
            Fortran::lower::genOmpAccAtomicRead<
                Fortran::parser::OmpAtomicRead,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicRead);
          },
          [&](const Fortran::parser::OmpAtomicWrite &atomicWrite) {
            Fortran::lower::genOmpAccAtomicWrite<
                Fortran::parser::OmpAtomicWrite,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicWrite);
          },
          [&](const Fortran::parser::OmpAtomic &atomicConstruct) {
            Fortran::lower::genOmpAtomic<Fortran::parser::OmpAtomic,
                                         Fortran::parser::OmpAtomicClauseList>(
                converter, atomicConstruct);
          },
          [&](const Fortran::parser::OmpAtomicUpdate &atomicUpdate) {
            Fortran::lower::genOmpAccAtomicUpdate<
                Fortran::parser::OmpAtomicUpdate,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicUpdate);
          },
          [&](const Fortran::parser::OmpAtomicCapture &atomicCapture) {
            Fortran::lower::genOmpAccAtomicCapture<
                Fortran::parser::OmpAtomicCapture,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicCapture);
          },
      },
      atomicConstruct.u);
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPDeclareTargetConstruct
                       &declareTargetConstruct) {
  llvm::SmallVector<DeclareTargetCapturePair, 0> symbolAndClause;
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  mlir::omp::DeclareTargetDeviceType deviceType = getDeclareTargetInfo(
      converter, eval, declareTargetConstruct, symbolAndClause);

  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(
        converter.mangleName(std::get<Fortran::semantics::Symbol>(symClause)));
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

    auto declareTargetOp =
        llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(op);
    if (!declareTargetOp)
      fir::emitFatalError(
          converter.getCurrentLocation(),
          "Attempt to apply declare target on unsupported operation");

    // The function or global already has a declare target applied to it, very
    // likely through implicit capture (usage in another declare target
    // function/subroutine). It should be marked as any if it has been assigned
    // both host and nohost, else we skip, as there is no change
    if (declareTargetOp.isDeclareTarget()) {
      if (declareTargetOp.getDeclareTargetDeviceType() != deviceType)
        declareTargetOp.setDeclareTarget(
            mlir::omp::DeclareTargetDeviceType::any,
            std::get<mlir::omp::DeclareTargetCaptureClause>(symClause));
      continue;
    }

    declareTargetOp.setDeclareTarget(
        deviceType, std::get<mlir::omp::DeclareTargetCaptureClause>(symClause));
  }
}

//===----------------------------------------------------------------------===//
// Public functions
//===----------------------------------------------------------------------===//

void Fortran::lower::genOpenMPTerminator(fir::FirOpBuilder &builder,
                                         mlir::Operation *op,
                                         mlir::Location loc) {
  if (mlir::isa<mlir::omp::WsLoopOp, mlir::omp::ReductionDeclareOp,
                mlir::omp::AtomicUpdateOp, mlir::omp::SimdLoopOp>(op))
    builder.create<mlir::omp::YieldOp>(loc);
  else
    builder.create<mlir::omp::TerminatorOp>(loc);
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
            genOMP(converter, eval, declareTargetConstruct);
          },
          [&](const Fortran::parser::OpenMPRequiresConstruct
                  &requiresConstruct) {
            // Requires directives are gathered and processed in semantics and
            // then combined in the lowering bridge before triggering codegen
            // just once. Hence, there is no need to lower each individual
            // occurrence here.
          },
          [&](const Fortran::parser::OpenMPThreadprivate &threadprivate) {
            // The directive is lowered when instantiating the variable to
            // support the case of threadprivate variable declared in module.
          },
      },
      ompDeclConstruct.u);
}

int64_t Fortran::lower::getCollapseValue(
    const Fortran::parser::OmpClauseList &clauseList) {
  for (const Fortran::parser::OmpClause &clause : clauseList.v) {
    if (const auto &collapseClause =
            std::get_if<Fortran::parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      return Fortran::evaluate::ToInt64(*expr).value();
    }
  }
  return 1;
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

    // The symbol may be use-associated multiple times, and nothing needs to be
    // done after the original symbol is mapped to the threadprivatized value
    // for the first time. Use the threadprivatized value directly.
    mlir::Operation *op;
    if (auto declOp = symValue.getDefiningOp<hlfir::DeclareOp>())
      op = declOp.getMemref().getDefiningOp();
    else
      op = symValue.getDefiningOp();
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

  for (const Fortran::parser::OmpClause &clause : clauseList.v) {
    if (const auto &reductionClause =
            std::get_if<Fortran::parser::OmpClause::Reduction>(&clause.u)) {
      const auto &redOperator{std::get<Fortran::parser::OmpReductionOperator>(
          reductionClause->v.t)};
      const auto &objectList{
          std::get<Fortran::parser::OmpObjectList>(reductionClause->v.t)};
      if (const auto *reductionOp =
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
        for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
          if (const auto *name{
                  Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
            if (const Fortran::semantics::Symbol * symbol{name->symbol}) {
              mlir::Value reductionVal = converter.getSymbolAddress(*symbol);
              if (auto declOp = reductionVal.getDefiningOp<hlfir::DeclareOp>())
                reductionVal = declOp.getBase();
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
                  } else if (mlir::Operation *reductionOp =
                                 findReductionChain(loadVal, &reductionVal)) {
                    updateReduction(reductionOp, firOpBuilder, loadVal,
                                    reductionVal);
                  }
                }
              }
            }
          }
        }
      } else if (const auto *reductionIntrinsic =
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
          for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
            if (const auto *name{Fortran::parser::Unwrap<Fortran::parser::Name>(
                    ompObject)}) {
              if (const Fortran::semantics::Symbol * symbol{name->symbol}) {
                mlir::Value reductionVal = converter.getSymbolAddress(*symbol);
                if (auto declOp =
                        reductionVal.getDefiningOp<hlfir::DeclareOp>())
                  reductionVal = declOp.getBase();
                for (const mlir::OpOperand &reductionValUse :
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
    if (mlir::Operation *reductionOp = loadOperand.getOwner()) {
      if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(reductionOp)) {
        for (mlir::OpOperand &convertOperand : convertOp.getRes().getUses()) {
          if (mlir::Operation *reductionOp = convertOperand.getOwner())
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
        if (auto assign =
                mlir::dyn_cast<hlfir::AssignOp>(reductionOperand.getOwner())) {
          if (assign.getLhs() == *reductionVal) {
            assign.erase();
            return reductionOp;
          }
        }
      }
    }
  }
  return nullptr;
}

// for a logical operator 'op' reduction X = X op Y
// This function returns the operation responsible for converting Y from
// fir.logical<4> to i1
fir::ConvertOp
Fortran::lower::getConvertFromReductionOp(mlir::Operation *reductionOp,
                                          mlir::Value loadVal) {
  for (mlir::Value reductionOperand : reductionOp->getOperands()) {
    if (auto convertOp =
            mlir::dyn_cast<fir::ConvertOp>(reductionOperand.getDefiningOp())) {
      if (convertOp.getOperand() == loadVal)
        continue;
      return convertOp;
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

void Fortran::lower::removeStoreOp(mlir::Operation *reductionOp,
                                   mlir::Value symVal) {
  for (mlir::Operation *reductionOpUse : reductionOp->getUsers()) {
    if (auto convertReduction =
            mlir::dyn_cast<fir::ConvertOp>(reductionOpUse)) {
      for (mlir::Operation *convertReductionUse :
           convertReduction.getRes().getUsers()) {
        if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(convertReductionUse)) {
          if (storeOp.getMemref() == symVal)
            storeOp.erase();
        }
        if (auto assignOp =
                mlir::dyn_cast<hlfir::AssignOp>(convertReductionUse)) {
          if (assignOp.getLhs() == symVal)
            assignOp.erase();
        }
      }
    }
  }
}

bool Fortran::lower::isOpenMPTargetConstruct(
    const Fortran::parser::OpenMPConstruct &omp) {
  llvm::omp::Directive dir = llvm::omp::Directive::OMPD_unknown;
  if (const auto *block =
          std::get_if<Fortran::parser::OpenMPBlockConstruct>(&omp.u)) {
    const auto &begin =
        std::get<Fortran::parser::OmpBeginBlockDirective>(block->t);
    dir = std::get<Fortran::parser::OmpBlockDirective>(begin.t).v;
  } else if (const auto *loop =
                 std::get_if<Fortran::parser::OpenMPLoopConstruct>(&omp.u)) {
    const auto &begin =
        std::get<Fortran::parser::OmpBeginLoopDirective>(loop->t);
    dir = std::get<Fortran::parser::OmpLoopDirective>(begin.t).v;
  }
  return llvm::omp::allTargetSet.test(dir);
}

bool Fortran::lower::isOpenMPDeviceDeclareTarget(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &ompDecl) {
  return std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct &ompReq) {
            mlir::omp::DeclareTargetDeviceType targetType =
                getDeclareTargetFunctionDevice(converter, eval, ompReq)
                    .value_or(mlir::omp::DeclareTargetDeviceType::host);
            return targetType != mlir::omp::DeclareTargetDeviceType::host;
          },
          [&](const auto &) { return false; },
      },
      ompDecl.u);
}

void Fortran::lower::genOpenMPRequires(
    mlir::Operation *mod, const Fortran::semantics::Symbol *symbol) {
  using MlirRequires = mlir::omp::ClauseRequires;
  using SemaRequires = Fortran::semantics::WithOmpDeclarative::RequiresFlag;

  if (auto offloadMod =
          llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(mod)) {
    Fortran::semantics::WithOmpDeclarative::RequiresFlags semaFlags;
    if (symbol) {
      Fortran::common::visit(
          [&](const auto &details) {
            if constexpr (std::is_base_of_v<
                              Fortran::semantics::WithOmpDeclarative,
                              std::decay_t<decltype(details)>>) {
              if (details.has_ompRequires())
                semaFlags = *details.ompRequires();
            }
          },
          symbol->details());
    }

    MlirRequires mlirFlags = MlirRequires::none;
    if (semaFlags.test(SemaRequires::ReverseOffload))
      mlirFlags = mlirFlags | MlirRequires::reverse_offload;
    if (semaFlags.test(SemaRequires::UnifiedAddress))
      mlirFlags = mlirFlags | MlirRequires::unified_address;
    if (semaFlags.test(SemaRequires::UnifiedSharedMemory))
      mlirFlags = mlirFlags | MlirRequires::unified_shared_memory;
    if (semaFlags.test(SemaRequires::DynamicAllocators))
      mlirFlags = mlirFlags | MlirRequires::dynamic_allocators;

    offloadMod.setRequires(mlirFlags);
  }
}
