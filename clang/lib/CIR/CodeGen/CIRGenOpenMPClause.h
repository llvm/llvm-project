//===--- CIRGenOpenMPClause.h - OpenMP clause processor ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCLAUSE_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCLAUSE_H

#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "mlir/Dialect/OpenMP/OpenMPClauseOperands.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

namespace clang::CIRGen {

class CIRGenFunction;

/// Processes OpenMP clauses for a directive, writing results into the
/// auto-generated ClauseOps from the OMP dialect.
class OpenMPClauseProcessor {
  CIRGenFunction &cgf;
  CIRGenModule &cgm;
  CIRGenBuilderTy &builder;
  mlir::Location loc;
  llvm::ArrayRef<const OMPClause *> clauses;

public:
  OpenMPClauseProcessor(CIRGenFunction &cgf, CIRGenModule &cgm,
                        CIRGenBuilderTy &builder, mlir::Location loc,
                        llvm::ArrayRef<const OMPClause *> clauses)
      : cgf(cgf), cgm(cgm), builder(builder), loc(loc), clauses(clauses) {}

  bool processProcBind(mlir::omp::ProcBindClauseOps &result) const;

  /// Process map clauses. The optional \p mapSyms parameter collects the
  /// VarDecls corresponding to each map operand.
  bool
  processMap(mlir::omp::MapClauseOps &result,
             llvm::SmallVectorImpl<const VarDecl *> *mapSyms = nullptr) const;

  /// Emit an errorNYI for each clause of the given types if present.
  template <typename... ClauseTypes>
  void processTODO(llvm::omp::Directive directive) const;

private:
  template <typename ClauseType>
  void processTODOClause(llvm::omp::Directive directive) const;
};

template <typename ClauseType>
void OpenMPClauseProcessor::processTODOClause(
    llvm::omp::Directive directive) const {
  for (const OMPClause *c : clauses) {
    if (isa<ClauseType>(c)) {
      std::string msg =
          ("OpenMP " + llvm::omp::getOpenMPDirectiveName(directive) + " " +
           llvm::omp::getOpenMPClauseName(c->getClauseKind()) + " clause")
              .str();
      cgm.errorNYI(c->getBeginLoc(), msg);
    }
  }
}

template <typename... ClauseTypes>
void OpenMPClauseProcessor::processTODO(llvm::omp::Directive directive) const {
  (processTODOClause<ClauseTypes>(directive), ...);
}

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCLAUSE_H
