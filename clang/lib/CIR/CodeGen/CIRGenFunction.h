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
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/TypeEvaluationKind.h"

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
  /// The GlobalDecl for the current function being compiled or the global
  /// variable currently being initialized.
  clang::GlobalDecl curGD;

  /// The compiler-generated variable that holds the return value.
  std::optional<mlir::Value> fnRetAlloca;

  /// The function for which code is currently being generated.
  cir::FuncOp curFn;

  using DeclMapTy = llvm::DenseMap<const clang::Decl *, Address>;
  /// This keeps track of the CIR allocas or globals for local C
  /// declarations.
  DeclMapTy localDeclMap;

  clang::ASTContext &getContext() const { return cgm.getASTContext(); }

  CIRGenBuilderTy &getBuilder() { return builder; }

  CIRGenModule &getCIRGenModule() { return cgm; }
  const CIRGenModule &getCIRGenModule() const { return cgm; }

  mlir::Block *getCurFunctionEntryBlock() { return &curFn.getRegion().front(); }

  /// Sanitizers enabled for this function.
  clang::SanitizerSet sanOpts;

  /// Whether or not a Microsoft-style asm block has been processed within
  /// this fuction. These can potentially set the return value.
  bool sawAsmBlock = false;

  mlir::Type convertTypeForMem(QualType t);

  mlir::Type convertType(clang::QualType t);
  mlir::Type convertType(const TypeDecl *t) {
    return convertType(getContext().getTypeDeclType(t));
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
  mlir::Value createDummyValue(mlir::Location loc, clang::QualType qt);

  void emitNullInitialization(mlir::Location loc, Address destPtr, QualType ty);

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

  /// An abstract representation of regular/ObjC call/message targets.
  class AbstractCallee {
    /// The function declaration of the callee.
    [[maybe_unused]] const clang::Decl *calleeDecl;

  public:
    AbstractCallee() : calleeDecl(nullptr) {}
    AbstractCallee(const clang::FunctionDecl *fd) : calleeDecl(fd) {}
  };

  void finishFunction(SourceLocation endLoc);

  /// Determine whether the given initializer is trivial in the sense
  /// that it requires no code to be generated.
  bool isTrivialInitializer(const Expr *init);

  /// If the specified expression does not fold to a constant, or if it does but
  /// contains a label, return false.  If it constant folds return true and set
  /// the boolean result in Result.
  bool constantFoldsToBool(const clang::Expr *cond, bool &resultBool,
                           bool allowLabels = false);
  bool constantFoldsToSimpleInteger(const clang::Expr *cond,
                                    llvm::APSInt &resultInt,
                                    bool allowLabels = false);

  /// Return true if the statement contains a label in it.  If
  /// this statement is not executed normally, it not containing a label means
  /// that we can just remove the code.
  bool containsLabel(const clang::Stmt *s, bool ignoreCaseStmts = false);

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

  /// Perform the usual unary conversions on the specified expression and
  /// compare the result against zero, returning an Int1Ty value.
  mlir::Value evaluateExprAsBool(const clang::Expr *e);

  /// Set the address of a local variable.
  void setAddrOfLocalVar(const clang::VarDecl *vd, Address addr) {
    assert(!localDeclMap.count(vd) && "Decl already exists in LocalDeclMap!");
    localDeclMap.insert({vd, addr});
    // TODO: Add symbol table support
  }

  /// Construct an address with the natural alignment of T. If a pointer to T
  /// is expected to be signed, the pointer passed to this function must have
  /// been signed, and the returned Address will have the pointer authentication
  /// information needed to authenticate the signed pointer.
  Address makeNaturalAddressForPointer(mlir::Value ptr, QualType t,
                                       CharUnits alignment,
                                       bool forPointeeType = false,
                                       LValueBaseInfo *baseInfo = nullptr) {
    if (alignment.isZero())
      alignment = cgm.getNaturalTypeAlignment(t, baseInfo);
    return Address(ptr, convertTypeForMem(t), alignment);
  }

  LValue makeAddrLValue(Address addr, QualType ty,
                        AlignmentSource source = AlignmentSource::Type) {
    return makeAddrLValue(addr, ty, LValueBaseInfo(source));
  }

  LValue makeAddrLValue(Address addr, QualType ty, LValueBaseInfo baseInfo) {
    return LValue::makeAddr(addr, ty, baseInfo);
  }

  cir::FuncOp generateCode(clang::GlobalDecl gd, cir::FuncOp fn,
                           cir::FuncType funcType);

  clang::QualType buildFunctionArgList(clang::GlobalDecl gd,
                                       FunctionArgList &args);

  /// Emit code for the start of a function.
  /// \param loc       The location to be associated with the function.
  /// \param startLoc  The location of the function body.
  void startFunction(clang::GlobalDecl gd, clang::QualType returnType,
                     cir::FuncOp fn, cir::FuncType funcType,
                     FunctionArgList args, clang::SourceLocation loc,
                     clang::SourceLocation startLoc);

  /// Represents a scope, including function bodies, compound statements, and
  /// the substatements of if/while/do/for/switch/try statements.  This class
  /// handles any automatic cleanup, along with the return value.
  struct LexicalScope {
  private:
    // TODO(CIR): This will live in the base class RunCleanupScope once that
    // class is upstreamed.
    CIRGenFunction &cgf;

    // Points to the scope entry block. This is useful, for instance, for
    // helping to insert allocas before finalizing any recursive CodeGen from
    // switches.
    mlir::Block *entryBlock;

    LexicalScope *parentScope = nullptr;

    // Only Regular is used at the moment. Support for other kinds will be
    // added as the relevant statements/expressions are upstreamed.
    enum Kind {
      Regular,   // cir.if, cir.scope, if_regions
      Ternary,   // cir.ternary
      Switch,    // cir.switch
      Try,       // cir.try
      GlobalInit // cir.global initialization code
    };
    Kind scopeKind = Kind::Regular;

    // The scope return value.
    mlir::Value retVal = nullptr;

    mlir::Location beginLoc;
    mlir::Location endLoc;

  public:
    unsigned depth = 0;

    LexicalScope(CIRGenFunction &cgf, mlir::Location loc, mlir::Block *eb)
        : cgf(cgf), entryBlock(eb), parentScope(cgf.curLexScope), beginLoc(loc),
          endLoc(loc) {

      assert(entryBlock && "LexicalScope requires an entry block");
      cgf.curLexScope = this;
      if (parentScope)
        ++depth;

      if (const auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
        assert(fusedLoc.getLocations().size() == 2 && "too many locations");
        beginLoc = fusedLoc.getLocations()[0];
        endLoc = fusedLoc.getLocations()[1];
      }
    }

    void setRetVal(mlir::Value v) { retVal = v; }

    void cleanup();
    void restore() { cgf.curLexScope = parentScope; }

    ~LexicalScope() {
      assert(!cir::MissingFeatures::generateDebugInfo());
      cleanup();
      restore();
    }

    // ---
    // Kind
    // ---
    bool isGlobalInit() { return scopeKind == Kind::GlobalInit; }
    bool isRegular() { return scopeKind == Kind::Regular; }
    bool isSwitch() { return scopeKind == Kind::Switch; }
    bool isTernary() { return scopeKind == Kind::Ternary; }
    bool isTry() { return scopeKind == Kind::Try; }

    void setAsGlobalInit() { scopeKind = Kind::GlobalInit; }
    void setAsSwitch() { scopeKind = Kind::Switch; }
    void setAsTernary() { scopeKind = Kind::Ternary; }

    // ---
    // Return handling.
    // ---

  private:
    // `returnBlock`, `returnLoc`, and all the functions that deal with them
    // will change and become more complicated when `switch` statements are
    // upstreamed.  `case` statements within the `switch` are in the same scope
    // but have their own regions.  Therefore the LexicalScope will need to
    // keep track of multiple return blocks.
    mlir::Block *returnBlock = nullptr;
    std::optional<mlir::Location> returnLoc;

    // See the comment on `getOrCreateRetBlock`.
    mlir::Block *createRetBlock(CIRGenFunction &cgf, mlir::Location loc) {
      assert(returnBlock == nullptr && "only one return block per scope");
      // Create the cleanup block but don't hook it up just yet.
      mlir::OpBuilder::InsertionGuard guard(cgf.builder);
      returnBlock =
          cgf.builder.createBlock(cgf.builder.getBlock()->getParent());
      updateRetLoc(returnBlock, loc);
      return returnBlock;
    }

    cir::ReturnOp emitReturn(mlir::Location loc);
    void emitImplicitReturn();

  public:
    mlir::Block *getRetBlock() { return returnBlock; }
    mlir::Location getRetLoc(mlir::Block *b) { return *returnLoc; }
    void updateRetLoc(mlir::Block *b, mlir::Location loc) { returnLoc = loc; }

    // Create the return block for this scope, or return the existing one.
    // This get-or-create logic is necessary to handle multiple return
    // statements within the same scope, which can happen if some of them are
    // dead code or if there is a `goto` into the middle of the scope.
    mlir::Block *getOrCreateRetBlock(CIRGenFunction &cgf, mlir::Location loc) {
      if (returnBlock == nullptr) {
        returnBlock = createRetBlock(cgf, loc);
        return returnBlock;
      }
      updateRetLoc(returnBlock, loc);
      return returnBlock;
    }

    mlir::Block *getEntryBlock() { return entryBlock; }
  };

  LexicalScope *curLexScope = nullptr;

  /// ----------------------
  /// CIR emit functions
  /// ----------------------
private:
  void emitAndUpdateRetAlloca(clang::QualType type, mlir::Location loc,
                              clang::CharUnits alignment);

public:
  mlir::Value emitAlloca(llvm::StringRef name, mlir::Type ty,
                         mlir::Location loc, clang::CharUnits alignment,
                         bool insertIntoFnEntryBlock,
                         mlir::Value arraySize = nullptr);
  mlir::Value emitAlloca(llvm::StringRef name, mlir::Type ty,
                         mlir::Location loc, clang::CharUnits alignment,
                         mlir::OpBuilder::InsertPoint ip,
                         mlir::Value arraySize = nullptr);

  void emitAggExpr(const clang::Expr *e, AggValueSlot slot);

  /// Emit code to compute the specified expression which can have any type. The
  /// result is returned as an RValue struct. If this is an aggregate
  /// expression, the aggloc/agglocvolatile arguments indicate where the result
  /// should be returned.
  RValue emitAnyExpr(const clang::Expr *e);

  LValue emitArraySubscriptExpr(const clang::ArraySubscriptExpr *e);

  AutoVarEmission emitAutoVarAlloca(const clang::VarDecl &d);

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void emitAutoVarDecl(const clang::VarDecl &d);

  void emitAutoVarCleanups(const AutoVarEmission &emission);
  void emitAutoVarInit(const AutoVarEmission &emission);

  LValue emitBinaryOperatorLValue(const BinaryOperator *e);

  mlir::LogicalResult emitBreakStmt(const clang::BreakStmt &s);

  RValue emitCall(const CIRGenFunctionInfo &funcInfo,
                  const CIRGenCallee &callee, cir::CIRCallOpInterface *callOp,
                  mlir::Location loc);
  RValue emitCall(clang::QualType calleeTy, const CIRGenCallee &callee,
                  const clang::CallExpr *e);
  RValue emitCallExpr(const clang::CallExpr *e);
  CIRGenCallee emitCallee(const clang::Expr *e);

  mlir::LogicalResult emitContinueStmt(const clang::ContinueStmt &s);
  mlir::LogicalResult emitDoStmt(const clang::DoStmt &s);

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

  mlir::LogicalResult emitFunctionBody(const clang::Stmt *body);

  mlir::Value emitPromotedScalarExpr(const Expr *e, QualType promotionType);

  /// Emit the computation of the specified expression of scalar type.
  mlir::Value emitScalarExpr(const clang::Expr *e);

  mlir::Value emitScalarPrePostIncDec(const UnaryOperator *e, LValue lv,
                                      bool isInc, bool isPre);

  /// Build a debug stoppoint if we are emitting debug info.
  void emitStopPoint(const Stmt *s);

  // Build CIR for a statement. useCurrentScope should be true if no
  // new scopes need be created when finding a compound statement.
  mlir::LogicalResult
  emitStmt(const clang::Stmt *s, bool useCurrentScope,
           llvm::ArrayRef<const Attr *> attrs = std::nullopt);

  mlir::LogicalResult emitSimpleStmt(const clang::Stmt *s,
                                     bool useCurrentScope);

  mlir::LogicalResult emitForStmt(const clang::ForStmt &s);

  void emitCompoundStmt(const clang::CompoundStmt &s);

  void emitCompoundStmtWithoutScope(const clang::CompoundStmt &s);

  void emitDecl(const clang::Decl &d);
  mlir::LogicalResult emitDeclStmt(const clang::DeclStmt &s);
  LValue emitDeclRefLValue(const clang::DeclRefExpr *e);

  /// Emit an `if` on a boolean condition to the specified blocks.
  /// FIXME: Based on the condition, this might try to simplify the codegen of
  /// the conditional based on the branch.
  /// In the future, we may apply code generation simplifications here,
  /// similar to those used in classic LLVM  codegen
  /// See `EmitBranchOnBoolExpr` for inspiration.
  mlir::LogicalResult emitIfOnBoolExpr(const clang::Expr *cond,
                                       const clang::Stmt *thenS,
                                       const clang::Stmt *elseS);
  cir::IfOp emitIfOnBoolExpr(const clang::Expr *cond,
                             BuilderCallbackRef thenBuilder,
                             mlir::Location thenLoc,
                             BuilderCallbackRef elseBuilder,
                             std::optional<mlir::Location> elseLoc = {});

  mlir::Value emitOpOnBoolExpr(mlir::Location loc, const clang::Expr *cond);

  mlir::LogicalResult emitIfStmt(const clang::IfStmt &s);

  /// Emit code to compute the specified expression,
  /// ignoring the result.
  void emitIgnoredExpr(const clang::Expr *e);

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

  /// Given an expression with a pointer type, emit the value and compute our
  /// best estimate of the alignment of the pointee.
  ///
  /// One reasonable way to use this information is when there's a language
  /// guarantee that the pointer must be aligned to some stricter value, and
  /// we're simply trying to ensure that sufficiently obvious uses of under-
  /// aligned objects don't get miscompiled; for example, a placement new
  /// into the address of a local variable.  In such a case, it's quite
  /// reasonable to just ignore the returned alignment when it isn't from an
  /// explicit source.
  Address emitPointerWithAlignment(const clang::Expr *expr,
                                   LValueBaseInfo *baseInfo);

  mlir::LogicalResult emitReturnStmt(const clang::ReturnStmt &s);

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  mlir::Value emitScalarConversion(mlir::Value src, clang::QualType srcType,
                                   clang::QualType dstType,
                                   clang::SourceLocation loc);

  void emitScalarInit(const clang::Expr *init, mlir::Location loc,
                      LValue lvalue, bool capturedByInit = false);

  void emitStoreOfScalar(mlir::Value value, Address addr, bool isVolatile,
                         clang::QualType ty, bool isInit = false,
                         bool isNontemporal = false);
  void emitStoreOfScalar(mlir::Value value, LValue lvalue, bool isInit);

  /// Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void emitStoreThroughLValue(RValue src, LValue dst, bool isInit = false);

  mlir::Value emitStoreThroughBitfieldLValue(RValue src, LValue dstresult);

  /// Given a value and its clang type, returns the value casted to its memory
  /// representation.
  /// Note: CIR defers most of the special casting to the final lowering passes
  /// to conserve the high level information.
  mlir::Value emitToMemory(mlir::Value value, clang::QualType ty);

  LValue emitUnaryOpLValue(const clang::UnaryOperator *e);

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void emitVarDecl(const clang::VarDecl &d);

  mlir::LogicalResult emitWhileStmt(const clang::WhileStmt &s);

  /// Given an assignment `*lhs = rhs`, emit a test that checks if \p rhs is
  /// nonnull, if 1\p LHS is marked _Nonnull.
  void emitNullabilityCheck(LValue lhs, mlir::Value rhs,
                            clang::SourceLocation loc);

  /// ----------------------
  /// CIR build helpers
  /// -----------------
public:
  Address createTempAlloca(mlir::Type ty, CharUnits align, mlir::Location loc,
                           const Twine &name, bool insertIntoFnEntryBlock);

  //===--------------------------------------------------------------------===//
  //                         OpenACC Emission
  //===--------------------------------------------------------------------===//
private:
  template <typename Op>
  mlir::LogicalResult
  emitOpenACCOp(mlir::Location start, OpenACCDirectiveKind dirKind,
                SourceLocation dirLoc,
                llvm::ArrayRef<const OpenACCClause *> clauses);
  // Function to do the basic implementation of an operation with an Associated
  // Statement.  Models AssociatedStmtConstruct.
  template <typename Op, typename TermOp>
  mlir::LogicalResult emitOpenACCOpAssociatedStmt(
      mlir::Location start, mlir::Location end, OpenACCDirectiveKind dirKind,
      SourceLocation dirLoc, llvm::ArrayRef<const OpenACCClause *> clauses,
      const Stmt *associatedStmt);

public:
  mlir::LogicalResult
  emitOpenACCComputeConstruct(const OpenACCComputeConstruct &s);
  mlir::LogicalResult emitOpenACCLoopConstruct(const OpenACCLoopConstruct &s);
  mlir::LogicalResult
  emitOpenACCCombinedConstruct(const OpenACCCombinedConstruct &s);
  mlir::LogicalResult emitOpenACCDataConstruct(const OpenACCDataConstruct &s);
  mlir::LogicalResult
  emitOpenACCEnterDataConstruct(const OpenACCEnterDataConstruct &s);
  mlir::LogicalResult
  emitOpenACCExitDataConstruct(const OpenACCExitDataConstruct &s);
  mlir::LogicalResult
  emitOpenACCHostDataConstruct(const OpenACCHostDataConstruct &s);
  mlir::LogicalResult emitOpenACCWaitConstruct(const OpenACCWaitConstruct &s);
  mlir::LogicalResult emitOpenACCInitConstruct(const OpenACCInitConstruct &s);
  mlir::LogicalResult
  emitOpenACCShutdownConstruct(const OpenACCShutdownConstruct &s);
  mlir::LogicalResult emitOpenACCSetConstruct(const OpenACCSetConstruct &s);
  mlir::LogicalResult
  emitOpenACCUpdateConstruct(const OpenACCUpdateConstruct &s);
  mlir::LogicalResult
  emitOpenACCAtomicConstruct(const OpenACCAtomicConstruct &s);
  mlir::LogicalResult emitOpenACCCacheConstruct(const OpenACCCacheConstruct &s);

  void emitOpenACCDeclare(const OpenACCDeclareDecl &d);
  void emitOpenACCRoutine(const OpenACCRoutineDecl &d);
};

} // namespace clang::CIRGen

#endif
