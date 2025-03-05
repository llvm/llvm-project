//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal per-function state used for AST-to-ClangIR code gen
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_CODEGEN_CIRGENFUNCTION_H
#define CLANG_LIB_CIR_CODEGEN_CIRGENFUNCTION_H

#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "CIRGenTypeCache.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/TypeEvaluationKind.h"

#include "llvm/ADT/ScopedHashTable.h"

namespace {
class ScalarExprEmitter;
} // namespace

namespace clang::CIRGen {

class CIRGenFunction : public CIRGenTypeCache {
public:
  CIRGenModule &cgm;

private:
  friend class ::ScalarExprEmitter;
  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  CIRGenBuilderTy &builder;

public:
  clang::QualType fnRetTy;

  /// This is the current function or global initializer that is generated code
  /// for.
  mlir::Operation *curFn = nullptr;

  clang::ASTContext &getContext() const { return cgm.getASTContext(); }

  CIRGenBuilderTy &getBuilder() { return builder; }

  CIRGenModule &getCIRGenModule() { return cgm; }
  const CIRGenModule &getCIRGenModule() const { return cgm; }

  mlir::Type convertTypeForMem(QualType T);

  mlir::Type convertType(clang::QualType T);
  mlir::Type convertType(const TypeDecl *T) {
    return convertType(getContext().getTypeDeclType(T));
  }

  ///  Return the cir::TypeEvaluationKind of QualType \c type.
  static cir::TypeEvaluationKind getEvaluationKind(clang::QualType type);

  static bool hasScalarEvaluationKind(clang::QualType type) {
    return getEvaluationKind(type) == cir::TEK_Scalar;
  }

  CIRGenFunction(CIRGenModule &cgm, CIRGenBuilderTy &builder,
                 bool suppressNewContext = false);
  ~CIRGenFunction();

  CIRGenTypes &getTypes() const { return cgm.getTypes(); }

  mlir::MLIRContext &getMLIRContext() { return cgm.getMLIRContext(); }

  /// Use to track source locations across nested visitor traversals.
  /// Always use a `SourceLocRAIIObject` to change currSrcLoc.
  std::optional<mlir::Location> currSrcLoc;
  class SourceLocRAIIObject {
    CIRGenFunction &cgf;
    std::optional<mlir::Location> oldLoc;

  public:
    SourceLocRAIIObject(CIRGenFunction &cgf, mlir::Location value) : cgf(cgf) {
      if (cgf.currSrcLoc)
        oldLoc = cgf.currSrcLoc;
      cgf.currSrcLoc = value;
    }

    /// Can be used to restore the state early, before the dtor
    /// is run.
    void restore() { cgf.currSrcLoc = oldLoc; }
    ~SourceLocRAIIObject() { restore(); }
  };

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(clang::SourceLocation srcLoc);
  mlir::Location getLoc(clang::SourceRange srcLoc);
  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  const clang::LangOptions &getLangOpts() const { return cgm.getLangOpts(); }

  void finishFunction(SourceLocation endLoc);
  mlir::LogicalResult emitFunctionBody(const clang::Stmt *body);

  // Build CIR for a statement. useCurrentScope should be true if no
  // new scopes need be created when finding a compound statement.
  mlir::LogicalResult
  emitStmt(const clang::Stmt *s, bool useCurrentScope,
           llvm::ArrayRef<const Attr *> attrs = std::nullopt);

  mlir::LogicalResult emitSimpleStmt(const clang::Stmt *s,
                                     bool useCurrentScope);

  void emitCompoundStmt(const clang::CompoundStmt &s);

  void emitCompoundStmtWithoutScope(const clang::CompoundStmt &s);

  mlir::LogicalResult emitReturnStmt(const clang::ReturnStmt &s);

  /// Emit the computation of the specified expression of scalar type.
  mlir::Value emitScalarExpr(const clang::Expr *e);
  cir::FuncOp generateCode(clang::GlobalDecl gd, cir::FuncOp fn,
                           cir::FuncType funcType);

  /// Emit code for the start of a function.
  /// \param loc       The location to be associated with the function.
  /// \param startLoc  The location of the function body.
  void startFunction(clang::GlobalDecl gd, clang::QualType retTy,
                     cir::FuncOp fn, cir::FuncType funcType,
                     clang::SourceLocation loc, clang::SourceLocation startLoc);
};

} // namespace clang::CIRGen

#endif
