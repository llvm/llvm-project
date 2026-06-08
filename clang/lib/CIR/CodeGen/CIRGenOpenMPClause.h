//===--- CIRGenOpenMPClause.h - OpenMP clause emitter -----------*- C++ -*-===//
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

#include <type_traits>

namespace clang::CIRGen {

class CIRGenFunction;

/// A type-only list of OpenMP clause AST node types.
/// Note: The clause AST classes do not have a default constructor, so a
/// std::tuple is not practical.
template <typename... Clauses> struct OpenMPClauseList {};

/// Emits OpenMP clauses for a directive, writing results into the
/// auto-generated ClauseOps from the OMP dialect.
class OpenMPClauseEmitter {
  CIRGenFunction &cgf;
  CIRGenModule &cgm;
  CIRGenBuilderTy &builder;
  mlir::Location loc;
  llvm::ArrayRef<const OMPClause *> clauses;

public:
  OpenMPClauseEmitter(CIRGenFunction &cgf, CIRGenModule &cgm,
                      CIRGenBuilderTy &builder, mlir::Location loc,
                      llvm::ArrayRef<const OMPClause *> clauses)
      : cgf(cgf), cgm(cgm), builder(builder), loc(loc), clauses(clauses) {}

  bool emitProcBind(mlir::omp::ProcBindClauseOps &result) const;

  /// Emit map clauses. The optional \p mapSyms parameter collects the
  /// VarDecls corresponding to each map operand.
  bool emitMap(mlir::omp::MapClauseOps &result,
               llvm::SmallVectorImpl<const VarDecl *> *mapSyms = nullptr) const;

  /// Verify the clauses of a directive to make sure all legal cases are either
  /// implemented or give a NYI error. If the clause is neither, then
  /// an unknown clause error will be emitted.
  template <typename... SupportedClauses, typename... NYIClauses>
  void emitNYI(OpenMPClauseList<NYIClauses...> nyi,
               llvm::omp::Directive directive) const;

private:
  /// True if T is the same type as any of Ts.
  template <typename T, typename... Ts>
  static constexpr bool isAnyOf = (std::is_same_v<T, Ts> || ...);
};

template <typename... SupportedClauses, typename... NYIClauses>
void OpenMPClauseEmitter::emitNYI(OpenMPClauseList<NYIClauses...>,
                                  llvm::omp::Directive directive) const {
  static_assert(
      (!isAnyOf<NYIClauses, SupportedClauses...> && ...),
      "the supported and not-yet-implemented clause lists must be disjoint");

  for (const OMPClause *c : clauses) {
    if ((isa<NYIClauses>(c) || ...)) {
      std::string msg =
          ("OpenMP " + llvm::omp::getOpenMPDirectiveName(directive) + " " +
           llvm::omp::getOpenMPClauseName(c->getClauseKind()) + " clause")
              .str();
      cgm.errorNYI(c->getBeginLoc(), msg);
    } else if (!(isa<SupportedClauses>(c) || ...)) {
      // Unknown/illegal clause encountered
      llvm_unreachable("unexpected OpenMP clause");
    }
  }
}

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCLAUSE_H
