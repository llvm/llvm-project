//===-- Lower/OpenMP/DataSharingProcessor.h ---------------------*- C++ -*-===//
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
#ifndef FORTRAN_LOWER_DATASHARINGPROCESSOR_H
#define FORTRAN_LOWER_DATASHARINGPROCESSOR_H

#include "Clauses.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/symbol.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace mlir {
namespace omp {
struct PrivateClauseOps;
} // namespace omp
} // namespace mlir

namespace Fortran {
namespace lower {
namespace omp {

class DataSharingProcessor {
private:
  bool hasLastPrivateOp;
  mlir::OpBuilder::InsertPoint lastPrivIP;
  mlir::OpBuilder::InsertPoint insPt;
  mlir::Value loopIV;
  // Symbols in private, firstprivate, and/or lastprivate clauses.
  llvm::SetVector<const semantics::Symbol *> privatizedSymbols;
  llvm::SetVector<const semantics::Symbol *> defaultSymbols;
  llvm::SetVector<const semantics::Symbol *> implicitSymbols;
  llvm::DenseMap<const semantics::Symbol *, mlir::omp::PrivateClauseOp>
      symToPrivatizer;
  lower::AbstractConverter &converter;
  semantics::SemanticsContext &semaCtx;
  fir::FirOpBuilder &firOpBuilder;
  omp::List<omp::Clause> clauses;
  lower::pft::Evaluation &eval;
  bool useDelayedPrivatization;
  lower::SymMap *symTable;

  bool needBarrier();
  void collectSymbols(semantics::Symbol::Flag flag,
                      llvm::SetVector<const semantics::Symbol *> &symbols);
  void collectSymbolsInNestedRegions(
      lower::pft::Evaluation &eval, semantics::Symbol::Flag flag,
      llvm::SetVector<const semantics::Symbol *> &symbolsInNestedRegions);
  void collectOmpObjectListSymbol(
      const omp::ObjectList &objects,
      llvm::SetVector<const semantics::Symbol *> &symbolSet);
  void collectSymbolsForPrivatization();
  void insertBarrier();
  void collectDefaultSymbols();
  void collectImplicitSymbols();
  void privatize(mlir::omp::PrivateClauseOps *clauseOps,
                 llvm::SmallVectorImpl<const semantics::Symbol *> *privateSyms);
  void defaultPrivatize(
      mlir::omp::PrivateClauseOps *clauseOps,
      llvm::SmallVectorImpl<const semantics::Symbol *> *privateSyms);
  void implicitPrivatize(
      mlir::omp::PrivateClauseOps *clauseOps,
      llvm::SmallVectorImpl<const semantics::Symbol *> *privateSyms);
  void
  doPrivatize(const semantics::Symbol *sym,
              mlir::omp::PrivateClauseOps *clauseOps,
              llvm::SmallVectorImpl<const semantics::Symbol *> *privateSyms);
  void copyLastPrivatize(mlir::Operation *op);
  void insertLastPrivateCompare(mlir::Operation *op);
  void cloneSymbol(const semantics::Symbol *sym);
  void
  copyFirstPrivateSymbol(const semantics::Symbol *sym,
                         mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr);
  void copyLastPrivateSymbol(const semantics::Symbol *sym,
                             mlir::OpBuilder::InsertPoint *lastPrivIP);
  void insertDeallocs();

public:
  DataSharingProcessor(lower::AbstractConverter &converter,
                       semantics::SemanticsContext &semaCtx,
                       const List<Clause> &clauses,
                       lower::pft::Evaluation &eval,
                       bool useDelayedPrivatization = false,
                       lower::SymMap *symTable = nullptr)
      : hasLastPrivateOp(false), converter(converter), semaCtx(semaCtx),
        firOpBuilder(converter.getFirOpBuilder()), clauses(clauses), eval(eval),
        useDelayedPrivatization(useDelayedPrivatization), symTable(symTable) {}

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
  void processStep1(
      mlir::omp::PrivateClauseOps *clauseOps = nullptr,
      llvm::SmallVectorImpl<const semantics::Symbol *> *privateSyms = nullptr);
  void processStep2(mlir::Operation *op, bool isLoop);

  void setLoopIV(mlir::Value iv) {
    assert(!loopIV && "Loop iteration variable already set");
    loopIV = iv;
  }
};

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_DATASHARINGPROCESSOR_H
