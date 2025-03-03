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
#include "CIRGenCall.h"
#include "CIRGenModule.h"
#include "CIRGenTypeCache.h"
#include "CIRGenValue.h"

#include "Address.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
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

  using DeclMapTy = llvm::DenseMap<const clang::Decl *, Address>;
  /// This keeps track of the CIR allocas or globals for local C
  /// declarations.
  DeclMapTy LocalDeclMap;

  clang::ASTContext &getContext() const { return cgm.getASTContext(); }

  CIRGenBuilderTy &getBuilder() { return builder; }

  CIRGenModule &getCIRGenModule() { return cgm; }
  const CIRGenModule &getCIRGenModule() const { return cgm; }

  mlir::Block *getCurFunctionEntryBlock() {
    auto fn = mlir::dyn_cast<cir::FuncOp>(curFn);
    assert(fn && "other callables NYI");
    return &fn.getRegion().front();
  }

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

private:
  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(mlir::Value addrVal, const clang::Decl *var,
                              clang::QualType ty, mlir::Location loc,
                              clang::CharUnits alignment, bool isParam = false);

public:
  mlir::Value emitAlloca(llvm::StringRef name, mlir::Type ty,
                         mlir::Location loc, clang::CharUnits alignment);

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

  mlir::LogicalResult emitDeclStmt(const clang::DeclStmt &s);

  mlir::LogicalResult emitReturnStmt(const clang::ReturnStmt &s);

  /// Given an expression that represents a value lvalue, this method emits
  /// the address of the lvalue, then loads the result as an rvalue,
  /// returning the rvalue.
  RValue emitLoadOfLValue(LValue lv, SourceLocation loc);

  /// EmitLoadOfScalar - Load a scalar value from an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.  The l-value must be a simple
  /// l-value.
  mlir::Value emitLoadOfScalar(LValue lvalue, SourceLocation loc);

  /// Emit code to compute a designator that specifies the location
  /// of the expression.
  /// FIXME: document this function better.
  LValue emitLValue(const clang::Expr *e);

  void emitDecl(const clang::Decl &d);

  LValue emitDeclRefLValue(const clang::DeclRefExpr *e);

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void emitAutoVarDecl(const clang::VarDecl &d);

  void emitAutoVarAlloca(const clang::VarDecl &d);
  void emitAutoVarInit(const clang::VarDecl &d);
  void emitAutoVarCleanups(const clang::VarDecl &d);

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void emitVarDecl(const clang::VarDecl &d);

  /// Set the address of a local variable.
  void setAddrOfLocalVar(const clang::VarDecl *vd, Address addr) {
    assert(!LocalDeclMap.count(vd) && "Decl already exists in LocalDeclMap!");
    LocalDeclMap.insert({vd, addr});
    // TODO: Add symbol table support
  }

  /// Emit the computation of the specified expression of scalar type.
  mlir::Value emitScalarExpr(const clang::Expr *e);
  cir::FuncOp generateCode(clang::GlobalDecl gd, cir::FuncOp fn,
                           cir::FuncType funcType);

  clang::QualType buildFunctionArgList(clang::GlobalDecl gd,
                                       FunctionArgList &args);

  /// Emit code for the start of a function.
  /// \param loc       The location to be associated with the function.
  /// \param startLoc  The location of the function body.
  void startFunction(clang::GlobalDecl gd, clang::QualType retTy,
                     cir::FuncOp fn, cir::FuncType funcType,
                     FunctionArgList args, clang::SourceLocation loc,
                     clang::SourceLocation startLoc);

  Address createTempAlloca(mlir::Type ty, CharUnits align, mlir::Location loc,
                           const Twine &name = "tmp");
};
} // namespace clang::CIRGen

#endif
