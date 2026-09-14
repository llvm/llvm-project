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

#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/symbol.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include <variant>

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
    OMPConstructSymbolVisitor(
        semantics::SemanticsContext &ctx,
        llvm::ArrayRef<const semantics::Symbol *> metadirectiveLoopIVs)
        : metadirectiveLoopIVs(metadirectiveLoopIVs.begin(),
                               metadirectiveLoopIVs.end()),
          isMetadirectiveLoop(!metadirectiveLoopIVs.empty()),
          version(ctx.langOptions().getOpenMPVersion()) {}
    template <typename T>
    bool Pre(const T &) {
      return true;
    }
    template <typename T>
    void Post(const T &) {}

    bool Pre(const parser::LoopControl::Bounds &bounds) {
      if (isMetadirectiveLoop)
        if (const semantics::Symbol *symbol = bounds.Name().thing.symbol)
          metadirectiveLoopIVs.insert(symbol);
      return true;
    }

    bool Pre(const parser::OpenMPConstruct &omp) {
      // Skip constructs that may not have privatizations.
      if (isOpenMPPrivatizingConstruct(omp, version))
        constructs.push_back(&omp);
      return true;
    }

    void Post(const parser::OpenMPConstruct &omp) {
      if (isOpenMPPrivatizingConstruct(omp, version))
        constructs.pop_back();
    }

    void Post(const parser::Name &name) {
      auto current = !constructs.empty() ? constructs.back() : ConstructPtr();
      symDefMap.try_emplace(name.symbol, current);
    }

    bool Pre(const parser::DeclarationConstruct &decl) {
      constructs.push_back(&decl);
      return true;
    }

    void Post(const parser::DeclarationConstruct &decl) {
      constructs.pop_back();
    }

    /// Given a \p symbol and an \p eval, returns true if eval is the OMP
    /// construct that defines symbol.
    bool isSymbolDefineBy(const semantics::Symbol *symbol,
                          lower::pft::Evaluation &eval) const;

    // Given a \p symbol, returns true if it is defined by a nested
    // `DeclarationConstruct`.
    bool
    isSymbolDefineByNestedDeclaration(const semantics::Symbol *symbol) const;

  private:
    using ConstructPtr = std::variant<const parser::OpenMPConstruct *,
                                      const parser::DeclarationConstruct *>;
    llvm::SmallVector<ConstructPtr> constructs;
    llvm::DenseMap<semantics::Symbol *, ConstructPtr> symDefMap;
    llvm::SmallPtrSet<const semantics::Symbol *, 4> metadirectiveLoopIVs;
    bool isMetadirectiveLoop;

    llvm::omp::Version version;
  };

  mlir::OpBuilder::InsertPoint lastPrivIP;
  llvm::SmallVector<mlir::Value> loopIVs;
  // Symbols in private, firstprivate, and/or lastprivate clauses.
  llvm::SetVector<const semantics::Symbol *> explicitlyPrivatizedSymbols;
  llvm::SetVector<const semantics::Symbol *> defaultSymbols;
  llvm::SetVector<const semantics::Symbol *> allPrivatizedSymbols;
  llvm::SetVector<const semantics::Symbol *> conditionalLastPrivatizedSymbols;
  // When true, conditional-lastprivate list items get an ordinary private copy
  // (their in-loop working value) plus a separate reduction struct as the
  // conditional-last accumulator.  Used by worksharing loops, where a
  // nonmonotonic schedule can execute chunks out of order.  When false the list
  // item is bound directly to the reduction struct (used by sections, which are
  // lexically ordered and never need the private copy).
  bool conditionalLpUsesPrivateCopy = false;

  lower::AbstractConverter &converter;
  semantics::SemanticsContext &semaCtx;
  fir::FirOpBuilder &firOpBuilder;
  omp::List<omp::Clause> clauses;
  lower::pft::Evaluation &eval;
  bool shouldCollectPreDeterminedSymbols;
  bool useDelayedPrivatization;
  bool forceHeapAllocationForPrivateDynamicArrays = false;
  llvm::SmallPtrSet<const semantics::Symbol *, 16> mightHaveReadHostSym;
  lower::SymMap &symTable;
  bool isTargetPrivatization;
  bool isMetadirectiveLoop;
  OMPConstructSymbolVisitor visitor;

  bool needBarrier();
  void collectPrivatizedSymbols(
      std::optional<semantics::Symbol::Flag> flag,
      const llvm::SetVector<const semantics::Symbol *> &allSymbols,
      const llvm::SetVector<const semantics::Symbol *> &symbolsInNestedRegions,
      llvm::SetVector<const semantics::Symbol *> *symbols = nullptr);
  void
  collectSymbols(semantics::Symbol::Flag flag,
                 llvm::SetVector<const semantics::Symbol *> *symbols = nullptr);
  void collectSymbolsInNestedRegions(
      lower::pft::Evaluation &eval, semantics::Symbol::Flag flag,
      llvm::SetVector<const semantics::Symbol *> &symbolsInNestedRegions);
  void collectOmpObjectListSymbol(
      const omp::ObjectList &objects,
      llvm::SetVector<const semantics::Symbol *> &symbolSet);
  void collectSymbolsForPrivatization();
  void insertBarrier(mlir::omp::PrivateClauseOps *clauseOps);
  void collectDefaultSymbols();
  void collectImplicitSymbols();
  void collectPreDeterminedSymbols();
  void collectIndirectReferences();
  void privatize(mlir::omp::PrivateClauseOps *clauseOps,
                 std::optional<llvm::omp::Directive> dir = std::nullopt);
  void copyLastPrivatize(mlir::Operation *op);
  void insertLastPrivateCompare(mlir::Operation *op);
  void cloneSymbol(const semantics::Symbol *sym);
  void
  copyFirstPrivateSymbol(const semantics::Symbol *sym,
                         mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr);
  void copyLastPrivateSymbol(const semantics::Symbol *sym,
                             mlir::OpBuilder::InsertPoint *lastPrivIP);
  void insertDeallocs();

  static bool isOpenMPPrivatizingConstruct(const parser::OpenMPConstruct &omp,
                                           llvm::omp::Version version);
  bool isOpenMPPrivatizingEvaluation(const pft::Evaluation &eval) const;

public:
  DataSharingProcessor(
      lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
      const List<Clause> &clauses, lower::pft::Evaluation &eval,
      bool shouldCollectPreDeterminedSymbols, bool useDelayedPrivatization,
      lower::SymMap &symTable, bool isTargetPrivatization = false,
      llvm::ArrayRef<const semantics::Symbol *> metadirectiveLoopIVs = {});

  DataSharingProcessor(lower::AbstractConverter &converter,
                       semantics::SemanticsContext &semaCtx,
                       lower::pft::Evaluation &eval,
                       bool useDelayedPrivatization, lower::SymMap &symTable,
                       bool isTargetPrivatization = false);

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
  void processStep1(mlir::omp::PrivateClauseOps *clauseOps = nullptr,
                    std::optional<llvm::omp::Directive> dir = std::nullopt);
  void processStep2(mlir::Operation *op, bool isLoop);

  void pushLoopIV(mlir::Value iv) { loopIVs.push_back(iv); }

  void setForceHeapAllocationForPrivateDynamicArrays(bool value = true) {
    forceHeapAllocationForPrivateDynamicArrays = value;
  }

  const llvm::SetVector<const semantics::Symbol *> &
  getAllSymbolsToPrivatize() const {
    return allPrivatizedSymbols;
  }

  llvm::ArrayRef<const semantics::Symbol *> getDelayedPrivSymbols() const {
    return useDelayedPrivatization
               ? allPrivatizedSymbols.getArrayRef()
               : llvm::ArrayRef<const semantics::Symbol *>();
  }

  void privatizeSymbol(const semantics::Symbol *symToPrivatize,
                       mlir::omp::PrivateClauseOps *clauseOps,
                       std::optional<llvm::omp::Directive> dir = std::nullopt);

  const llvm::SetVector<const semantics::Symbol *> &
  getConditionalLastprivateSymbols() const {
    return conditionalLastPrivatizedSymbols;
  }

  void setConditionalLpUsesPrivateCopy(bool v) {
    conditionalLpUsesPrivateCopy = v;
  }
};

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_DATASHARINGPROCESSOR_H
