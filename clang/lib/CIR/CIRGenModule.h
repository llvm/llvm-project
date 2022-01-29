//===--- CIRGenModule.h - Per-Module state for CIR gen ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
#define LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H

#include "CIRGenFunction.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceManager.h"

#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/Dialect/CIR/IR/CIRAttrs.h"
#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace cir {

/// Implementation of a CIR/MLIR emission from Clang AST.
///
/// This will emit operations that are specific to C(++)/ObjC(++) language,
/// preserving the semantics of the language and (hopefully) allow to perform
/// accurate analysis and transformation based on these high level semantics.
class CIRGenModule {
public:
  CIRGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx);
  CIRGenModule(CIRGenModule &) = delete;
  CIRGenModule &operator=(CIRGenModule &) = delete;
  ~CIRGenModule() = default;

  using SymTableTy = llvm::ScopedHashTable<const clang::Decl *, mlir::Value>;
  using SymTableScopeTy =
      llvm::ScopedHashTableScope<const clang::Decl *, mlir::Value>;

private:
  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated,
  /// the scope is destroyed and the mappings created in this scope are
  /// dropped.
  SymTableTy symbolTable;

  /// Hold Clang AST information.
  clang::ASTContext &astCtx;

  /// Per-function codegen information. Updated everytime buildCIR is called
  /// for FunctionDecls's.
  CIRGenFunction *CurCGF = nullptr;

  /// Per-module type mapping from clang AST to CIR.
  std::unique_ptr<CIRGenTypes> genTypes;

  /// Use to track source locations across nested visitor traversals.
  /// Always use a `SourceLocRAIIObject` to change currSrcLoc.
  std::optional<mlir::Location> currSrcLoc;
  class SourceLocRAIIObject {
    CIRGenModule &P;
    std::optional<mlir::Location> OldVal;

  public:
    SourceLocRAIIObject(CIRGenModule &p, mlir::Location Value) : P(p) {
      if (P.currSrcLoc)
        OldVal = P.currSrcLoc;
      P.currSrcLoc = Value;
    }

    /// Can be used to restore the state early, before the dtor
    /// is run.
    void restore() { P.currSrcLoc = OldVal; }
    ~SourceLocRAIIObject() { restore(); }
  };

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(const Decl *var, QualType T, mlir::Location loc,
                              CharUnits alignment, mlir::Value &addr,
                              bool IsParam = false);

public:
  mlir::ModuleOp getModule() { return theModule; }
  mlir::OpBuilder &getBuilder() { return builder; }
  clang::ASTContext &getASTContext() { return astCtx; }

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(SourceLocation SLoc);

  mlir::Location getLoc(SourceRange SLoc);

  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  struct AutoVarEmission {
    const VarDecl *Variable;
    /// The address of the alloca for languages with explicit address space
    /// (e.g. OpenCL) or alloca casted to generic pointer for address space
    /// agnostic languages (e.g. C++). Invalid if the variable was emitted
    /// as a global constant.
    Address Addr;

    /// True if the variable is of aggregate type and has a constant
    /// initializer.
    bool IsConstantAggregate;

    struct Invalid {};
    AutoVarEmission(Invalid) : Variable(nullptr), Addr(Address::invalid()) {}

    AutoVarEmission(const VarDecl &variable)
        : Variable(&variable), Addr(Address::invalid()),
          IsConstantAggregate(false) {}

    static AutoVarEmission invalid() { return AutoVarEmission(Invalid()); }
    /// Returns the raw, allocated address, which is not necessarily
    /// the address of the object itself. It is casted to default
    /// address space for address space agnostic languages.
    Address getAllocatedAddress() const { return Addr; }
  };

  /// Determine whether an object of this type can be emitted
  /// as a constant.
  ///
  /// If ExcludeCtor is true, the duration when the object's constructor runs
  /// will not be considered. The caller will need to verify that the object is
  /// not written to during its construction.
  /// FIXME: in LLVM codegen path this is part of CGM, which doesn't seem
  /// like necessary, since (1) it doesn't use CGM at all and (2) is AST type
  /// query specific.
  bool isTypeConstant(QualType Ty, bool ExcludeCtor);

  /// Emit the alloca and debug information for a
  /// local variable.  Does not emit initialization or destruction.
  AutoVarEmission buildAutoVarAlloca(const VarDecl &D);

  /// Determine whether the given initializer is trivial in the sense
  /// that it requires no code to be generated.
  bool isTrivialInitializer(const Expr *Init);

  // TODO: this can also be abstrated into common AST helpers
  bool hasBooleanRepresentation(QualType Ty);

  mlir::Value buildToMemory(mlir::Value Value, QualType Ty);

  void buildStoreOfScalar(mlir::Value value, LValue lvalue,
                          const Decl *InitDecl);

  void buildStoreOfScalar(mlir::Value Value, Address Addr, bool Volatile,
                          QualType Ty, LValueBaseInfo BaseInfo,
                          const Decl *InitDecl, bool isNontemporal);

  /// Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void buldStoreThroughLValue(RValue Src, LValue Dst, const Decl *InitDecl);

  void buildScalarInit(const Expr *init, const ValueDecl *D, LValue lvalue);

  /// Emit an expression as an initializer for an object (variable, field, etc.)
  /// at the given location.  The expression is not necessarily the normal
  /// initializer for the object, and the address is not necessarily
  /// its normal location.
  ///
  /// \param init the initializing expression
  /// \param D the object to act as if we're initializing
  /// \param lvalue the lvalue to initialize
  void buildExprAsInit(const Expr *init, const ValueDecl *D, LValue lvalue);

  void buildAutoVarInit(const AutoVarEmission &emission);

  void buildAutoVarCleanups(const AutoVarEmission &emission);

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void buildAutoVarDecl(const VarDecl &D);

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void buildVarDecl(const VarDecl &D);

  void buildDecl(const Decl &D);

  /// Emit the computation of the specified expression of scalar type,
  /// ignoring the result.
  mlir::Value buildScalarExpr(const Expr *E);

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  mlir::Value buildScalarConversion(mlir::Value Src, QualType SrcTy,
                                    QualType DstTy, SourceLocation Loc);

  mlir::LogicalResult buildReturnStmt(const ReturnStmt &S);

  mlir::LogicalResult buildDeclStmt(const DeclStmt &S);

  mlir::LogicalResult buildSimpleStmt(const Stmt *S, bool useCurrentScope);

  LValue buildDeclRefLValue(const DeclRefExpr *E);

  /// Emit code to compute the specified expression which
  /// can have any type.  The result is returned as an RValue struct.
  /// TODO: if this is an aggregate expression, add a AggValueSlot to indicate
  /// where the result should be returned.
  RValue buildAnyExpr(const Expr *E);

  LValue buildBinaryOperatorLValue(const BinaryOperator *E);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// Return the best known alignment for an unknown pointer to a
  /// particular class.
  CharUnits getClassPointerAlignment(const CXXRecordDecl *RD);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// TODO: Add TBAAAccessInfo
  CharUnits getNaturalPointeeTypeAlignment(QualType T,
                                           LValueBaseInfo *BaseInfo);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// TODO: Add TBAAAccessInfo
  CharUnits getNaturalTypeAlignment(QualType T, LValueBaseInfo *BaseInfo,
                                    bool forPointeeType);

  /// Given an expression of pointer type, try to
  /// derive a more accurate bound on the alignment of the pointer.
  Address buildPointerWithAlignment(const Expr *E, LValueBaseInfo *BaseInfo);

  LValue buildUnaryOpLValue(const UnaryOperator *E);

  /// Emit code to compute a designator that specifies the location
  /// of the expression.
  /// FIXME: document this function better.
  LValue buildLValue(const Expr *E);

  /// EmitIgnoredExpr - Emit code to compute the specified expression,
  /// ignoring the result.
  void buildIgnoredExpr(const Expr *E);

  /// If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return false.  If it
  /// constant folds return true and set the boolean result in Result.
  bool ConstantFoldsToSimpleInteger(const Expr *Cond, bool &ResultBool,
                                    bool AllowLabels);

  /// Return true if the statement contains a label in it.  If
  /// this statement is not executed normally, it not containing a label means
  /// that we can just remove the code.
  bool ContainsLabel(const Stmt *S, bool IgnoreCaseStmts = false);

  /// If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return false.  If it
  /// constant folds return true and set the folded value.
  bool ConstantFoldsToSimpleInteger(const Expr *Cond, llvm::APSInt &ResultInt,
                                    bool AllowLabels);

  /// Perform the usual unary conversions on the specified
  /// expression and compare the result against zero, returning an Int1Ty value.
  mlir::Value evaluateExprAsBool(const Expr *E);

  /// Emit an if on a boolean condition to the specified blocks.
  /// FIXME: Based on the condition, this might try to simplify the codegen of
  /// the conditional based on the branch. TrueCount should be the number of
  /// times we expect the condition to evaluate to true based on PGO data. We
  /// might decide to leave this as a separate pass (see EmitBranchOnBoolExpr
  /// for extra ideas).
  mlir::LogicalResult buildIfOnBoolExpr(const Expr *cond, mlir::Location loc,
                                        const Stmt *thenS, const Stmt *elseS);

  mlir::LogicalResult buildIfStmt(const IfStmt &S);

  // Build CIR for a statement. useCurrentScope should be true if no
  // new scopes need be created when finding a compound statement.
  mlir::LogicalResult buildStmt(const Stmt *S, bool useCurrentScope);

  mlir::LogicalResult buildFunctionBody(const Stmt *Body);

  mlir::LogicalResult buildCompoundStmt(const CompoundStmt &S);

  mlir::LogicalResult buildCompoundStmtWithoutScope(const CompoundStmt &S);

  void buildTopLevelDecl(Decl *decl);

  // Emit a new function and add it to the MLIR module.
  mlir::FuncOp buildFunction(const FunctionDecl *FD);

  mlir::Type getCIRType(const QualType &type);

  void verifyModule();
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
