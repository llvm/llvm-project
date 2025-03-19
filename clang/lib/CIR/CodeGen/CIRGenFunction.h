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
#include "clang/CIR/MissingFeatures.h"
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

  /// Sanitizers enabled for this function.
  clang::SanitizerSet sanOpts;

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
  void declare(mlir::Value addrVal, const clang::Decl *var, clang::QualType ty,
               mlir::Location loc, clang::CharUnits alignment,
               bool isParam = false);

public:
  mlir::Value emitAlloca(llvm::StringRef name, mlir::Type ty,
                         mlir::Location loc, clang::CharUnits alignment);

  mlir::Value createDummyValue(mlir::Location loc, clang::QualType qt);

private:
  // Track current variable initialization (if there's one)
  const clang::VarDecl *currVarDecl = nullptr;
  class VarDeclContext {
    CIRGenFunction &p;
    const clang::VarDecl *oldVal = nullptr;

  public:
    VarDeclContext(CIRGenFunction &p, const VarDecl *value) : p(p) {
      if (p.currVarDecl)
        oldVal = p.currVarDecl;
      p.currVarDecl = value;
    }

    /// Can be used to restore the state early, before the dtor
    /// is run.
    void restore() { p.currVarDecl = oldVal; }
    ~VarDeclContext() { restore(); }
  };

public:
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

  /// Emit code to compute the specified expression which can have any type. The
  /// result is returned as an RValue struct. If this is an aggregate
  /// expression, the aggloc/agglocvolatile arguments indicate where the result
  /// should be returned.
  RValue emitAnyExpr(const clang::Expr *e);

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

  /// Emit code to compute the specified expression,
  /// ignoring the result.
  void emitIgnoredExpr(const clang::Expr *e);

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

  void emitScalarInit(const clang::Expr *init, mlir::Location loc,
                      LValue lvalue, bool capturedByInit = false);

  LValue emitDeclRefLValue(const clang::DeclRefExpr *e);
  LValue emitUnaryOpLValue(const clang::UnaryOperator *e);

  /// Determine whether the given initializer is trivial in the sense
  /// that it requires no code to be generated.
  bool isTrivialInitializer(const Expr *init);

  /// Emit an expression as an initializer for an object (variable, field, etc.)
  /// at the given location.  The expression is not necessarily the normal
  /// initializer for the object, and the address is not necessarily
  /// its normal location.
  ///
  /// \param init the initializing expression
  /// \param d the object to act as if we're initializing
  /// \param lvalue the lvalue to initialize
  /// \param capturedByInit true if \p d is a __block variable whose address is
  /// potentially changed by the initializer
  void emitExprAsInit(const clang::Expr *init, const clang::ValueDecl *d,
                      LValue lvalue, bool capturedByInit = false);

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void emitAutoVarDecl(const clang::VarDecl &d);

  struct AutoVarEmission {
    const clang::VarDecl *Variable;
    /// The address of the alloca for languages with explicit address space
    /// (e.g. OpenCL) or alloca casted to generic pointer for address space
    /// agnostic languages (e.g. C++). Invalid if the variable was emitted
    /// as a global constant.
    Address Addr;

    /// True if the variable is of aggregate type and has a constant
    /// initializer.
    bool IsConstantAggregate = false;

    /// True if the variable is a __block variable that is captured by an
    /// escaping block.
    bool IsEscapingByRef = false;

    mlir::Value NRVOFlag{};

    struct Invalid {};
    AutoVarEmission(Invalid) : Variable(nullptr), Addr(Address::invalid()) {}

    AutoVarEmission(const clang::VarDecl &variable)
        : Variable(&variable), Addr(Address::invalid()) {}

    static AutoVarEmission invalid() { return AutoVarEmission(Invalid()); }

    bool wasEmittedAsGlobal() const { return !Addr.isValid(); }

    /// Returns the raw, allocated address, which is not necessarily
    /// the address of the object itself. It is casted to default
    /// address space for address space agnostic languages.
    Address getAllocatedAddress() const { return Addr; }

    /// Returns the address of the object within this declaration.
    /// Note that this does not chase the forwarding pointer for
    /// __block decls.
    Address getObjectAddress(CIRGenFunction &CGF) const {
      if (!IsEscapingByRef)
        return Addr;

      assert(!cir::MissingFeatures::opAllocaEscapeByReference());
      return Address::invalid();
    }
  };

  AutoVarEmission emitAutoVarAlloca(const clang::VarDecl &d);
  void emitAutoVarInit(const AutoVarEmission &emission);
  void emitAutoVarCleanups(const AutoVarEmission &emission);

  void emitStoreOfScalar(mlir::Value value, Address addr, bool isVolatile,
                         clang::QualType ty, bool isInit = false,
                         bool isNontemporal = false);
  void emitStoreOfScalar(mlir::Value value, LValue lvalue, bool isInit);

  /// Given a value and its clang type, returns the value casted to its memory
  /// representation.
  /// Note: CIR defers most of the special casting to the final lowering passes
  /// to conserve the high level information.
  mlir::Value emitToMemory(mlir::Value Value, clang::QualType Ty);

  /// Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void emitStoreThroughLValue(RValue Src, LValue Dst, bool isInit = false);

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void emitVarDecl(const clang::VarDecl &d);

  /// Set the address of a local variable.
  void setAddrOfLocalVar(const clang::VarDecl *vd, Address addr) {
    assert(!LocalDeclMap.count(vd) && "Decl already exists in LocalDeclMap!");
    LocalDeclMap.insert({vd, addr});
    // TODO: Add symbol table support
  }

  mlir::Value emitScalarPrePostIncDec(const UnaryOperator *e, LValue lv,
                                      bool isInc, bool isPre);

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
