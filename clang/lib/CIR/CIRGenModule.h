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
#include "llvm/ADT/SmallPtrSet.h"

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
  CIRGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx,
               const clang::CodeGenOptions &CGO);
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

  const clang::CodeGenOptions &codeGenOpts;

  /// Per-module type mapping from clang AST to CIR.
  std::unique_ptr<CIRGenTypes> genTypes;

  /// -------
  /// Goto
  /// -------

  /// A jump destination is an abstract label, branching to which may
  /// require a jump out through normal cleanups.
  struct JumpDest {
    JumpDest() = default;
    JumpDest(mlir::Block *Block) : Block(Block) {}

    bool isValid() const { return Block != nullptr; }
    mlir::Block *getBlock() const { return Block; }
    mlir::Block *Block = nullptr;
  };

  /// Track mlir Blocks for each C/C++ label.
  llvm::DenseMap<const clang::LabelDecl *, JumpDest> LabelMap;
  JumpDest &getJumpDestForLabel(const clang::LabelDecl *D);

  /// -------
  /// Lexical Scope: to be read as in the meaning in CIR, a scope is always
  /// related with initialization and destruction of objects.
  /// -------

  // Represents a cir.scope, cir.if, and then/else regions. I.e. lexical
  // scopes that require cleanups.
  struct LexicalScopeContext {
    unsigned Depth = 0;
    LexicalScopeContext(mlir::OpBuilder &builder, mlir::Location b,
                        mlir::Location e)
        : BeginLoc(b), EndLoc(e) {
      {
        // Create the cleanup block but dont hook it up around just
        // yet.
        mlir::OpBuilder::InsertionGuard guard(builder);
        CleanupBlock = builder.createBlock(builder.getBlock()->getParent());
      }
      assert(builder.getInsertionBlock() && "Should be valid");
    }
    ~LexicalScopeContext() = default;

    // Block containing cleanup code for things initialized in this
    // lexical context (scope).
    mlir::Block *CleanupBlock = nullptr;

    // Goto's introduced in this scope but didn't get fixed.
    llvm::SmallVector<std::pair<mlir::Operation *, const clang::LabelDecl *>, 4>
        PendingGotos;

    // Labels solved inside this scope.
    llvm::SmallPtrSet<const clang::LabelDecl *, 4> SolvedLabels;

    mlir::Location BeginLoc, EndLoc;
  };

  class LexicalScopeGuard {
    CIRGenModule &CGM;
    LexicalScopeContext *OldVal = nullptr;

  public:
    LexicalScopeGuard(CIRGenModule &c, LexicalScopeContext *L) : CGM(c) {
      if (CGM.currLexScope) {
        OldVal = CGM.currLexScope;
        L->Depth++;
      }
      CGM.currLexScope = L;
    }

    LexicalScopeGuard(const LexicalScopeGuard &) = delete;
    LexicalScopeGuard &operator=(const LexicalScopeGuard &) = delete;
    LexicalScopeGuard &operator=(LexicalScopeGuard &&other) = delete;

    void cleanup();
    void restore() { CGM.currLexScope = OldVal; }
    ~LexicalScopeGuard() {
      cleanup();
      restore();
    }
  };

  LexicalScopeContext *currLexScope = nullptr;

  /// -------
  /// Source Location tracking
  /// -------

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

  /// -------
  /// Declaring variables
  /// -------

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(const clang::Decl *var, clang::QualType T,
                              mlir::Location loc, clang::CharUnits alignment,
                              mlir::Value &addr, bool IsParam = false);

public:
  mlir::ModuleOp getModule() { return theModule; }
  mlir::OpBuilder &getBuilder() { return builder; }
  clang::ASTContext &getASTContext() { return astCtx; }

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(clang::SourceLocation SLoc);

  mlir::Location getLoc(clang::SourceRange SLoc);

  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  struct AutoVarEmission {
    const clang::VarDecl *Variable;
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

    AutoVarEmission(const clang::VarDecl &variable)
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
  bool isTypeConstant(clang::QualType Ty, bool ExcludeCtor);

  /// Emit the alloca and debug information for a
  /// local variable.  Does not emit initialization or destruction.
  AutoVarEmission buildAutoVarAlloca(const clang::VarDecl &D);

  /// Determine whether the given initializer is trivial in the sense
  /// that it requires no code to be generated.
  bool isTrivialInitializer(const clang::Expr *Init);

  // TODO: this can also be abstrated into common AST helpers
  bool hasBooleanRepresentation(clang::QualType Ty);

  mlir::Value buildToMemory(mlir::Value Value, clang::QualType Ty);

  void buildStoreOfScalar(mlir::Value value, LValue lvalue,
                          const clang::Decl *InitDecl);

  void buildStoreOfScalar(mlir::Value Value, Address Addr, bool Volatile,
                          clang::QualType Ty, LValueBaseInfo BaseInfo,
                          const clang::Decl *InitDecl, bool isNontemporal);

  /// Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void buldStoreThroughLValue(RValue Src, LValue Dst,
                              const clang::Decl *InitDecl);

  void buildScalarInit(const clang::Expr *init, const clang::ValueDecl *D,
                       LValue lvalue);

  /// Emit an expression as an initializer for an object (variable, field, etc.)
  /// at the given location.  The expression is not necessarily the normal
  /// initializer for the object, and the address is not necessarily
  /// its normal location.
  ///
  /// \param init the initializing expression
  /// \param D the object to act as if we're initializing
  /// \param lvalue the lvalue to initialize
  void buildExprAsInit(const clang::Expr *init, const clang::ValueDecl *D,
                       LValue lvalue);

  void buildAutoVarInit(const AutoVarEmission &emission);

  void buildAutoVarCleanups(const AutoVarEmission &emission);

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void buildAutoVarDecl(const clang::VarDecl &D);

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void buildVarDecl(const clang::VarDecl &D);

  void buildDecl(const clang::Decl &D);

  /// Emit the computation of the specified expression of scalar type,
  /// ignoring the result.
  mlir::Value buildScalarExpr(const clang::Expr *E);

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  mlir::Value buildScalarConversion(mlir::Value Src, clang::QualType SrcTy,
                                    clang::QualType DstTy,
                                    clang::SourceLocation Loc);

  mlir::LogicalResult buildBranchThroughCleanup(JumpDest &Dest,
                                                clang::LabelDecl *L,
                                                mlir::Location Loc);

  mlir::LogicalResult buildReturnStmt(const clang::ReturnStmt &S);

  mlir::LogicalResult buildLabel(const clang::LabelDecl *D);
  mlir::LogicalResult buildLabelStmt(const clang::LabelStmt &S);

  mlir::LogicalResult buildGotoStmt(const clang::GotoStmt &S);

  mlir::LogicalResult buildDeclStmt(const clang::DeclStmt &S);

  mlir::LogicalResult buildSimpleStmt(const clang::Stmt *S,
                                      bool useCurrentScope);

  LValue buildDeclRefLValue(const clang::DeclRefExpr *E);

  /// Emit code to compute the specified expression which
  /// can have any type.  The result is returned as an RValue struct.
  /// TODO: if this is an aggregate expression, add a AggValueSlot to indicate
  /// where the result should be returned.
  RValue buildAnyExpr(const clang::Expr *E);

  LValue buildBinaryOperatorLValue(const clang::BinaryOperator *E);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// Return the best known alignment for an unknown pointer to a
  /// particular class.
  clang::CharUnits getClassPointerAlignment(const clang::CXXRecordDecl *RD);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// TODO: Add TBAAAccessInfo
  clang::CharUnits getNaturalPointeeTypeAlignment(clang::QualType T,
                                                  LValueBaseInfo *BaseInfo);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// TODO: Add TBAAAccessInfo
  clang::CharUnits getNaturalTypeAlignment(clang::QualType T,
                                           LValueBaseInfo *BaseInfo,
                                           bool forPointeeType);

  /// Given an expression of pointer type, try to
  /// derive a more accurate bound on the alignment of the pointer.
  Address buildPointerWithAlignment(const clang::Expr *E,
                                    LValueBaseInfo *BaseInfo);

  LValue buildUnaryOpLValue(const clang::UnaryOperator *E);

  /// Emit code to compute a designator that specifies the location
  /// of the expression.
  /// FIXME: document this function better.
  LValue buildLValue(const clang::Expr *E);

  /// EmitIgnoredExpr - Emit code to compute the specified expression,
  /// ignoring the result.
  void buildIgnoredExpr(const clang::Expr *E);

  /// If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return false.  If it
  /// constant folds return true and set the boolean result in Result.
  bool ConstantFoldsToSimpleInteger(const clang::Expr *Cond, bool &ResultBool,
                                    bool AllowLabels);

  /// Return true if the statement contains a label in it.  If
  /// this statement is not executed normally, it not containing a label means
  /// that we can just remove the code.
  bool ContainsLabel(const clang::Stmt *S, bool IgnoreCaseStmts = false);

  /// If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return false.  If it
  /// constant folds return true and set the folded value.
  bool ConstantFoldsToSimpleInteger(const clang::Expr *Cond,
                                    llvm::APSInt &ResultInt, bool AllowLabels);

  /// Perform the usual unary conversions on the specified
  /// expression and compare the result against zero, returning an Int1Ty value.
  mlir::Value evaluateExprAsBool(const clang::Expr *E);

  /// Emit an if on a boolean condition to the specified blocks.
  /// FIXME: Based on the condition, this might try to simplify the codegen of
  /// the conditional based on the branch. TrueCount should be the number of
  /// times we expect the condition to evaluate to true based on PGO data. We
  /// might decide to leave this as a separate pass (see EmitBranchOnBoolExpr
  /// for extra ideas).
  mlir::LogicalResult buildIfOnBoolExpr(const clang::Expr *cond,
                                        mlir::Location loc,
                                        const clang::Stmt *thenS,
                                        const clang::Stmt *elseS);

  mlir::LogicalResult buildIfStmt(const clang::IfStmt &S);

  // Build CIR for a statement. useCurrentScope should be true if no
  // new scopes need be created when finding a compound statement.
  mlir::LogicalResult buildStmt(const clang::Stmt *S, bool useCurrentScope);

  mlir::LogicalResult buildFunctionBody(const clang::Stmt *Body);

  mlir::LogicalResult buildCompoundStmt(const clang::CompoundStmt &S);

  mlir::LogicalResult
  buildCompoundStmtWithoutScope(const clang::CompoundStmt &S);

  void buildTopLevelDecl(clang::Decl *decl);

  // Emit a new function and add it to the MLIR module.
  mlir::FuncOp buildFunction(const clang::FunctionDecl *FD);

  mlir::Type getCIRType(const clang::QualType &type);

  void verifyModule();
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
