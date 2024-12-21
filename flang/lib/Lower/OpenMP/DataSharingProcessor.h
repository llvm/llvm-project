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
  /// A symbol visitor that keeps track of the currently active OpenMPConstruct
  /// at any point in time. This is used to track Symbol definition scopes in
  /// order to tell which OMP scope defined vs. references a certain Symbol.
  struct OMPConstructSymbolVisitor {
    template <typename T>
    bool Pre(const T &) {
      return true;
    }
    template <typename T>
    void Post(const T &) {}

    bool Pre(const parser::OpenMPConstruct &omp) {
      // Skip constructs that may not have privatizations.
      if (!std::holds_alternative<parser::OpenMPCriticalConstruct>(omp.u))
        currentConstruct = &omp;
      return true;
    }

    void Post(const parser::OpenMPConstruct &omp) {
      currentConstruct = nullptr;
    }

    void Post(const parser::Name &name) {
      symDefMap.try_emplace(name.symbol, currentConstruct);
    }

    const parser::OpenMPConstruct *currentConstruct = nullptr;
    llvm::DenseMap<semantics::Symbol *, const parser::OpenMPConstruct *>
        symDefMap;

    /// Given a \p symbol and an \p eval, returns true if eval is the OMP
    /// construct that defines symbol.
    bool isSymbolDefineBy(const semantics::Symbol *symbol,
                          lower::pft::Evaluation &eval) const;
  };

  mlir::OpBuilder::InsertPoint lastPrivIP;
  llvm::SmallVector<mlir::Value> loopIVs;
  // Symbols in private, firstprivate, and/or lastprivate clauses.
  llvm::SetVector<const semantics::Symbol *> explicitlyPrivatizedSymbols;
  llvm::SetVector<const semantics::Symbol *> defaultSymbols;
  llvm::SetVector<const semantics::Symbol *> implicitSymbols;
  llvm::SetVector<const semantics::Symbol *> preDeterminedSymbols;
  llvm::SetVector<const semantics::Symbol *> allPrivatizedSymbols;

  llvm::DenseMap<const semantics::Symbol *, mlir::omp::PrivateClauseOp>
      symToPrivatizer;
  lower::AbstractConverter &converter;
  semantics::SemanticsContext &semaCtx;
  fir::FirOpBuilder &firOpBuilder;
  omp::List<omp::Clause> clauses;
  lower::pft::Evaluation &eval;
  bool shouldCollectPreDeterminedSymbols;
  bool useDelayedPrivatization;
  bool callsInitClone = false;
  lower::SymMap &symTable;
  OMPConstructSymbolVisitor visitor;

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
  void collectPreDeterminedSymbols();
  void privatize(mlir::omp::PrivateClauseOps *clauseOps);
  void doPrivatize(const semantics::Symbol *sym,
                   mlir::omp::PrivateClauseOps *clauseOps);
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
                       bool shouldCollectPreDeterminedSymbols,
                       bool useDelayedPrivatization, lower::SymMap &symTable);

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
  void processStep1(mlir::omp::PrivateClauseOps *clauseOps = nullptr);
  void processStep2(mlir::Operation *op, bool isLoop);

  void pushLoopIV(mlir::Value iv) { loopIVs.push_back(iv); }

  const llvm::SetVector<const semantics::Symbol *> &
  getAllSymbolsToPrivatize() const {
    return allPrivatizedSymbols;
  }

  llvm::ArrayRef<const semantics::Symbol *> getDelayedPrivSymbols() const {
    return useDelayedPrivatization
               ? allPrivatizedSymbols.getArrayRef()
               : llvm::ArrayRef<const semantics::Symbol *>();
  }
};

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_DATASHARINGPROCESSOR_H
