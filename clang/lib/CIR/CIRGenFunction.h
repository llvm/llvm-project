//===-- CIRGenFunction.h - Per-Function state for CIR gen -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-function state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H
#define LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H

#include "CIRGenCall.h"
#include "CIRGenModule.h"
#include "CIRGenValue.h"

#include "clang/AST/BaseSubobject.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/TargetInfo.h"

#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"

namespace clang {
class Expr;
} // namespace clang

namespace mlir {
namespace func {
class CallOp;
}
} // namespace mlir

namespace cir {

// FIXME: for now we are reusing this from lib/Clang/CodeGenFunction.h, which
// isn't available in the include dir. Same for getEvaluationKind below.
enum TypeEvaluationKind { TEK_Scalar, TEK_Complex, TEK_Aggregate };

class CIRGenFunction {
  CIRGenModule &CGM;
  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  mlir::OpBuilder &builder;

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
  private:
    // Block containing cleanup code for things initialized in this
    // lexical context (scope).
    mlir::Block *CleanupBlock = nullptr;

    // Points to scope entry block. This is useful, for instance, for
    // helping to insert allocas before finalizing any recursive codegen
    // from switches.
    mlir::Block *EntryBlock;

    // FIXME: perhaps we can use some info encoded in operations.
    enum Kind {
      Regular, // cir.if, cir.scope, if_regions
      Switch   // cir.switch
    } ScopeKind = Regular;

  public:
    unsigned Depth = 0;
    bool HasReturn = false;
    LexicalScopeContext(mlir::Location b, mlir::Location e, mlir::Block *eb)
        : EntryBlock(eb), BeginLoc(b), EndLoc(e) {}
    ~LexicalScopeContext() = default;

    // ---
    // Kind
    // ---
    bool isRegular() { return ScopeKind == Kind::Regular; }
    bool isSwitch() { return ScopeKind == Kind::Switch; }
    void setAsSwitch() { ScopeKind = Kind::Switch; }

    // ---
    // Goto handling
    // ---

    // Lazy create cleanup block or return what's available.
    mlir::Block *getOrCreateCleanupBlock(mlir::OpBuilder &builder) {
      if (CleanupBlock)
        return getCleanupBlock(builder);
      return createCleanupBlock(builder);
    }

    mlir::Block *getCleanupBlock(mlir::OpBuilder &builder) {
      return CleanupBlock;
    }
    mlir::Block *createCleanupBlock(mlir::OpBuilder &builder) {
      {
        // Create the cleanup block but dont hook it up around just yet.
        mlir::OpBuilder::InsertionGuard guard(builder);
        CleanupBlock = builder.createBlock(builder.getBlock()->getParent());
      }
      assert(builder.getInsertionBlock() && "Should be valid");
      return CleanupBlock;
    }

    // Goto's introduced in this scope but didn't get fixed.
    llvm::SmallVector<std::pair<mlir::Operation *, const clang::LabelDecl *>, 4>
        PendingGotos;

    // Labels solved inside this scope.
    llvm::SmallPtrSet<const clang::LabelDecl *, 4> SolvedLabels;

    // ---
    // Return handling
    // ---

  private:
    // On switches we need one return block per region, since cases don't
    // have their own scopes but are distinct regions nonetheless.
    llvm::SmallVector<mlir::Block *> RetBlocks;
    llvm::SmallVector<std::optional<mlir::Location>> RetLocs;
    unsigned int CurrentSwitchRegionIdx = -1;

    // There's usually only one ret block per scope, but this needs to be
    // get or create because of potential unreachable return statements, note
    // that for those, all source location maps to the first one found.
    mlir::Block *createRetBlock(CIRGenFunction &CGF, mlir::Location loc) {
      assert((isSwitch() || RetBlocks.size() == 0) &&
             "only switches can hold more than one ret block");

      // Create the cleanup block but dont hook it up around just yet.
      mlir::OpBuilder::InsertionGuard guard(CGF.builder);
      auto *b = CGF.builder.createBlock(CGF.builder.getBlock()->getParent());
      RetBlocks.push_back(b);
      RetLocs.push_back(loc);
      return b;
    }

  public:
    void updateCurrentSwitchCaseRegion() { CurrentSwitchRegionIdx++; }
    llvm::ArrayRef<mlir::Block *> getRetBlocks() { return RetBlocks; }
    llvm::ArrayRef<std::optional<mlir::Location>> getRetLocs() {
      return RetLocs;
    }

    mlir::Block *getOrCreateRetBlock(CIRGenFunction &CGF, mlir::Location loc) {
      unsigned int regionIdx = 0;
      if (isSwitch())
        regionIdx = CurrentSwitchRegionIdx;
      if (regionIdx >= RetBlocks.size())
        return createRetBlock(CGF, loc);
      return &*RetBlocks.back();
    }

    // ---
    // Scope entry block tracking
    // ---
    mlir::Block *getEntryBlock() { return EntryBlock; }

    mlir::Location BeginLoc, EndLoc;
  };

  class LexicalScopeGuard {
    CIRGenFunction &CGF;
    LexicalScopeContext *OldVal = nullptr;

  public:
    LexicalScopeGuard(CIRGenFunction &c, LexicalScopeContext *L) : CGF(c) {
      if (CGF.currLexScope) {
        OldVal = CGF.currLexScope;
        L->Depth++;
      }
      CGF.currLexScope = L;
    }

    LexicalScopeGuard(const LexicalScopeGuard &) = delete;
    LexicalScopeGuard &operator=(const LexicalScopeGuard &) = delete;
    LexicalScopeGuard &operator=(LexicalScopeGuard &&other) = delete;

    void cleanup();
    void restore() { CGF.currLexScope = OldVal; }
    ~LexicalScopeGuard() {
      cleanup();
      restore();
    }
  };

  LexicalScopeContext *currLexScope = nullptr;

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(const clang::Decl *var, clang::QualType ty,
                              mlir::Location loc, clang::CharUnits alignment,
                              mlir::Value &addr, bool isParam = false);
  mlir::Value buildAlloca(llvm::StringRef name, mlir::cir::InitStyle initStyle,
                          clang::QualType ty, mlir::Location loc,
                          clang::CharUnits alignment);
  void buildAndUpdateRetAlloca(clang::QualType ty, mlir::Location loc,
                               clang::CharUnits alignment);

  /// -------
  /// Source Location tracking
  /// -------

  /// Use to track source locations across nested visitor traversals.
  /// Always use a `SourceLocRAIIObject` to change currSrcLoc.
  std::optional<mlir::Location> currSrcLoc;
  class SourceLocRAIIObject {
    CIRGenFunction &P;
    std::optional<mlir::Location> OldVal;

  public:
    SourceLocRAIIObject(CIRGenFunction &p, mlir::Location Value) : P(p) {
      if (P.currSrcLoc)
        OldVal = P.currSrcLoc;
      P.currSrcLoc = Value;
    }

    /// Can be used to restore the state early, before the dtor
    /// is run.
    void restore() { P.currSrcLoc = OldVal; }
    ~SourceLocRAIIObject() { restore(); }
  };

public:
  using SymTableScopeTy =
      llvm::ScopedHashTableScope<const clang::Decl *, mlir::Value>;

  enum class EvaluationOrder {
    ///! No langauge constraints on evaluation order.
    Default,
    ///! Language semantics requrie left-to-right evaluation
    ForceLeftToRight,
    ///! Language semantics require right-to-left evaluation.
    ForceRightToLeft
  };

  /// Situations in which we might emit a check for the suitability of a pointer
  /// or glvalue. Needs to be kept in sync with ubsan_handlers.cpp in
  /// compiler-rt.
  enum TypeCheckKind {
    /// Checking hte operand of a load. Must be suitably sized and aligned.
    TCK_Load,
    /// Checking the destination of a store. Must be suitably sized and aligned.
    TCK_Store,
    /// Checking the bound value in a reference binding. Must be suitably sized
    /// and aligned, but is not required to refer to an object (until the
    /// reference is used), per core issue 453.
    TCK_ReferenceBinding,
    /// Checking the object expression in a non-static data member access. Must
    /// be an object within its lifetime.
    TCK_MemberAccess,
    /// Checking the 'this' pointer for a call to a non-static member function.
    /// Must be an object within its lifetime.
    TCK_MemberCall,
    /// Checking the 'this' pointer for a constructor call.
    TCK_ConstructorCall,
  };

  /// CurGD - The GlobalDecl for the current function being compiled.
  clang::GlobalDecl CurGD;

  /// ReturnValue - The temporary alloca to hold the return value. This is
  /// invalid iff the function has no return value.
  Address ReturnValue = Address::invalid();

  /// Counts of the number return expressions in the function.
  unsigned NumReturnExprs = 0;

  clang::QualType FnRetQualTy;
  std::optional<mlir::Type> FnRetCIRTy;
  std::optional<mlir::Value> FnRetAlloca;

  /// CXXThisDecl - When generating code for a C++ member function, this will
  /// hold the implicit 'this' declaration.
  clang::ImplicitParamDecl *CXXABIThisDecl = nullptr;
  mlir::Operation *CXXABIThisValue = nullptr;
  mlir::Operation *CXXThisValue = nullptr;
  clang::CharUnits CXXABIThisAlignment;
  clang::CharUnits CXXThisAlignment;

  /// The value of 'this' to sue when evaluating CXXDefaultInitExprs within this
  /// expression.
  Address CXXDefaultInitExprThis = Address::invalid();

  // CurFuncDecl - Holds the Decl for the current outermost non-closure context
  const clang::Decl *CurFuncDecl;
  /// CurCodeDecl - This is the inner-most code context, which includes blocks.
  const clang::Decl *CurCodeDecl;
  const CIRGenFunctionInfo *CurFnInfo;
  clang::QualType FnRetTy;
  mlir::FuncOp CurFn = nullptr;

  /// CXXStructorImplicitParamDecl - When generating code for a constructor or
  /// destructor, this will hold the implicit argument (e.g. VTT).
  clang::ImplicitParamDecl *CXXStructorImplicitParamDecl = nullptr;

  // The CallExpr within the current statement that the musttail attribute
  // applies to. nullptr if there is no 'musttail' on the current statement.
  const clang::CallExpr *MustTailCall = nullptr;

  clang::ASTContext &getContext() const;

  mlir::OpBuilder &getBuilder() { return builder; }

  /// Sanitizers enabled for this function.
  clang::SanitizerSet SanOpts;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated,
  /// the scope is destroyed and the mappings created in this scope are
  /// dropped.
  using SymTableTy = llvm::ScopedHashTable<const clang::Decl *, mlir::Value>;
  SymTableTy symbolTable;
  /// True if we need to emit the life-time markers. This is initially set in
  /// the constructor, but could be overwrriten to true if this is a coroutine.
  bool ShouldEmitLifetimeMarkers;

  using DeclMapTy = llvm::DenseMap<const clang::Decl *, Address>;
  /// LocalDeclMap - This keeps track of the CIR allocas or globals for local C
  /// delcs.
  DeclMapTy LocalDeclMap;

  /// DidCallStackSave - Whether llvm.stacksave has been called. Used to avoid
  /// calling llvm.stacksave for multiple VLAs in the same scope.
  /// TODO: Translate to MLIR
  bool DidCallStackSave = false;

  /// Whether we processed a Microsoft-style asm block during CIRGen. These can
  /// potentially set the return value.
  bool SawAsmBlock = false;

  /// In C++, whether we are code generating a thunk. This controls whether we
  /// should emit cleanups.
  bool CurFuncIsThunk = false;

  ///  Return the TypeEvaluationKind of QualType \c T.
  static TypeEvaluationKind getEvaluationKind(clang::QualType T);

  static bool hasScalarEvaluationKind(clang::QualType T) {
    return getEvaluationKind(T) == TEK_Scalar;
  }

  static bool hasAggregateEvaluationKind(clang::QualType T) {
    return getEvaluationKind(T) == TEK_Aggregate;
  }

  CIRGenFunction(CIRGenModule &CGM, mlir::OpBuilder &builder,
                 bool suppressNewContext = false);

  CIRGenTypes &getTypes() const { return CGM.getTypes(); }

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(clang::SourceLocation SLoc);

  mlir::Location getLoc(clang::SourceRange SLoc);

  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  const clang::LangOptions &getLangOpts() const { return CGM.getLangOpts(); }

  // TODO: This is currently just a dumb stub. But we want to be able to clearly
  // assert where we arne't doing things that we know we should and will crash
  // as soon as we add a DebugInfo type to this class.
  std::nullptr_t *getDebugInfo() { return nullptr; }

  /// Set the address of a local variable.
  void setAddrOfLocalVar(const clang::VarDecl *VD, Address Addr) {
    assert(!LocalDeclMap.count(VD) && "Decl already exists in LocalDeclMap!");
    LocalDeclMap.insert({VD, Addr});
  }

  /// Whether any type-checking sanitizers are enabled. If \c false, calls to
  /// buildTypeCheck can be skipped.
  bool sanitizePerformTypeCheck() const;

  void buildTypeCheck(TypeCheckKind TCK, clang::SourceLocation Loc,
                      mlir::Value V, clang::QualType Type,
                      clang::CharUnits Alignment = clang::CharUnits::Zero(),
                      clang::SanitizerSet SkippedChecks = clang::SanitizerSet(),
                      std::optional<mlir::Value> ArraySize = std::nullopt);

  void buildAggExpr(const clang::Expr *E, AggValueSlot Slot);

  void buildCXXConstructExpr(const clang::CXXConstructExpr *E,
                             AggValueSlot Dest);

  void buildCXXConstructorCall(const clang::CXXConstructorDecl *D,
                               clang::CXXCtorType Type, bool ForVirtualBase,
                               bool Delegating, AggValueSlot ThisAVS,
                               const clang::CXXConstructExpr *E);

  void buildCXXConstructorCall(const clang::CXXConstructorDecl *D,
                               clang::CXXCtorType Type, bool ForVirtualBase,
                               bool Delegating, Address This, CallArgList &Args,
                               AggValueSlot::Overlap_t Overlap,
                               clang::SourceLocation Loc,
                               bool NewPointerIsChecked);

  RValue buildCXXMemberCallExpr(const clang::CXXMemberCallExpr *E,
                                ReturnValueSlot ReturnValue);
  RValue buildCXXMemberOrOperatorMemberCallExpr(
      const clang::CallExpr *CE, const clang::CXXMethodDecl *MD,
      ReturnValueSlot ReturnValue, bool HasQualifier,
      clang::NestedNameSpecifier *Qualifier, bool IsArrow,
      const clang::Expr *Base);

  mlir::Operation *createLoad(const clang::VarDecl *VD, const char *Name);

  // Wrapper for function prototype sources. Wraps either a FunctionProtoType or
  // an ObjCMethodDecl.
  struct PrototypeWrapper {
    llvm::PointerUnion<const clang::FunctionProtoType *,
                       const clang::ObjCMethodDecl *>
        P;

    PrototypeWrapper(const clang::FunctionProtoType *FT) : P(FT) {}
    PrototypeWrapper(const clang::ObjCMethodDecl *MD) : P(MD) {}
  };

  bool LValueIsSuitableForInlineAtomic(LValue Src);

  /// An abstract representation of regular/ObjC call/message targets.
  class AbstractCallee {
    /// The function declaration of the callee.
    const clang::Decl *CalleeDecl;

  public:
    AbstractCallee() : CalleeDecl(nullptr) {}
    AbstractCallee(const clang::FunctionDecl *FD) : CalleeDecl(FD) {}
    AbstractCallee(const clang::ObjCMethodDecl *OMD) : CalleeDecl(OMD) {}
    bool hasFunctionDecl() const {
      return llvm::isa_and_nonnull<clang::FunctionDecl>(CalleeDecl);
    }
    const clang::Decl *getDecl() const { return CalleeDecl; }
    unsigned getNumParams() const {
      if (const auto *FD = llvm::dyn_cast<clang::FunctionDecl>(CalleeDecl))
        return FD->getNumParams();
      return llvm::cast<clang::ObjCMethodDecl>(CalleeDecl)->param_size();
    }
    const clang::ParmVarDecl *getParamDecl(unsigned I) const {
      if (const auto *FD = llvm::dyn_cast<clang::FunctionDecl>(CalleeDecl))
        return FD->getParamDecl(I);
      return *(llvm::cast<clang::ObjCMethodDecl>(CalleeDecl)->param_begin() +
               I);
    }
  };

  RValue convertTempToRValue(Address addr, clang::QualType type,
                             clang::SourceLocation Loc);

  mlir::Value buildLoadOfScalar(Address Addr, bool Volatile, clang::QualType Ty,
                                clang::SourceLocation Loc,
                                LValueBaseInfo BaseInfo,
                                bool isNontemporal = false);

  /// buildLoadOfScalar - Load a scalar value from an address, taking care to
  /// appropriately convert form the memory representation to the CIR value
  /// representation. The l-value must be a simple l-value.
  mlir::Value buildLoadOfScalar(LValue lvalue, clang::SourceLocation Loc);

  void buildCallArgs(
      CallArgList &Args, PrototypeWrapper Prototype,
      llvm::iterator_range<clang::CallExpr::const_arg_iterator> ArgRange,
      AbstractCallee AC = AbstractCallee(), unsigned ParamsToSkip = 0,
      EvaluationOrder Order = EvaluationOrder::Default);

  /// buildCall - Generate a call of the given function, expecting the given
  /// result type, and using the given argument list which specifies both the
  /// LLVM arguments and the types they were derived from.
  RValue buildCall(const CIRGenFunctionInfo &CallInfo,
                   const CIRGenCallee &Callee, ReturnValueSlot ReturnValue,
                   const CallArgList &Args, mlir::func::CallOp &callOrInvoke,
                   bool IsMustTail, clang::SourceLocation Loc);
  RValue buildCall(clang::QualType FnType, const CIRGenCallee &Callee,
                   const clang::CallExpr *E, ReturnValueSlot returnValue,
                   mlir::Value Chain = nullptr);

  RValue buildCallExpr(const clang::CallExpr *E,
                       ReturnValueSlot ReturnValue = ReturnValueSlot());

  void buildCallArg(CallArgList &args, const clang::Expr *E,
                    clang::QualType ArgType);

  /// buildAnyExprToTemp - Similarly to buildAnyExpr(), however, the result will
  /// always be accessible even if no aggregate location is provided.
  RValue buildAnyExprToTemp(const clang::Expr *E);

  CIRGenCallee buildCallee(const clang::Expr *E);

  /// buildAnyExpr - Emit code to compute the specified expression which can
  /// have any type. The result is returned as an RValue struct. If this is an
  /// aggregate expression, the aggloc/agglocvolatile arguments indicate where
  /// the result should be returned.
  RValue buildAnyExpr(const clang::Expr *E,
                      AggValueSlot aggSlot = AggValueSlot::ignored(),
                      bool ignoreResult = false);

  mlir::LogicalResult buildFunctionBody(const clang::Stmt *Body);

  // Build CIR for a statement. useCurrentScope should be true if no
  // new scopes need be created when finding a compound statement.
  mlir::LogicalResult buildStmt(const clang::Stmt *S, bool useCurrentScope);

  mlir::LogicalResult buildSimpleStmt(const clang::Stmt *S,
                                      bool useCurrentScope);

  mlir::LogicalResult buildForStmt(const clang::ForStmt &S);
  mlir::LogicalResult buildWhileStmt(const clang::WhileStmt &S);
  mlir::LogicalResult buildDoStmt(const clang::DoStmt &S);
  mlir::LogicalResult buildSwitchStmt(const clang::SwitchStmt &S);

  mlir::LogicalResult buildCompoundStmt(const clang::CompoundStmt &S);

  mlir::LogicalResult
  buildCompoundStmtWithoutScope(const clang::CompoundStmt &S);

  /// Emit code to compute the specified expression,
  /// ignoring the result.
  void buildIgnoredExpr(const clang::Expr *E);

  LValue buildArraySubscriptExpr(const clang::ArraySubscriptExpr *E,
                                 bool Accessed = false);

  mlir::LogicalResult buildDeclStmt(const clang::DeclStmt &S);

  /// Get an appropriate 'undef' rvalue for the given type.
  /// TODO: What's the equivalent for MLIR? Currently we're only using this for
  /// void types so it just returns RValue::get(nullptr) but it'll need
  /// addressed later.
  RValue GetUndefRValue(clang::QualType Ty);

  mlir::Value buildFromMemory(mlir::Value Value, clang::QualType Ty);

  mlir::Type convertType(clang::QualType T);

  mlir::LogicalResult buildIfStmt(const clang::IfStmt &S);

  mlir::LogicalResult buildReturnStmt(const clang::ReturnStmt &S);

  mlir::LogicalResult buildGotoStmt(const clang::GotoStmt &S);

  mlir::LogicalResult buildLabel(const clang::LabelDecl *D);
  mlir::LogicalResult buildLabelStmt(const clang::LabelStmt &S);

  mlir::LogicalResult buildBreakStmt(const clang::BreakStmt &S);
  mlir::LogicalResult buildContinueStmt(const clang::ContinueStmt &S);

  /// Emit code to compute a designator that specifies the location
  /// of the expression.
  /// FIXME: document this function better.
  LValue buildLValue(const clang::Expr *E);

  void buildDecl(const clang::Decl &D);

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

  /// Emit the computation of the specified expression of scalar type,
  /// ignoring the result.
  mlir::Value buildScalarExpr(const clang::Expr *E);

  mlir::Type getCIRType(const clang::QualType &type);

  mlir::LogicalResult buildCaseStmt(const clang::CaseStmt &S,
                                    mlir::Type condType,
                                    mlir::cir::CaseAttr &caseEntry);

  mlir::LogicalResult buildDefaultStmt(const clang::DefaultStmt &S,
                                       mlir::Type condType,
                                       mlir::cir::CaseAttr &caseEntry);

  mlir::FuncOp generateCode(clang::GlobalDecl GD, mlir::FuncOp Fn,
                            const CIRGenFunctionInfo &FnInfo);

  clang::QualType buildFunctionArgList(clang::GlobalDecl GD,
                                       FunctionArgList &Args);

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

  /// Emit the alloca and debug information for a
  /// local variable.  Does not emit initialization or destruction.
  AutoVarEmission buildAutoVarAlloca(const clang::VarDecl &D);

  void buildAutoVarInit(const AutoVarEmission &emission);

  void buildAutoVarCleanups(const AutoVarEmission &emission);

  void buildStoreOfScalar(mlir::Value value, LValue lvalue,
                          const clang::Decl *InitDecl);

  void buildStoreOfScalar(mlir::Value Value, Address Addr, bool Volatile,
                          clang::QualType Ty, LValueBaseInfo BaseInfo,
                          const clang::Decl *InitDecl, bool isNontemporal);

  mlir::Value buildToMemory(mlir::Value Value, clang::QualType Ty);

  /// Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void buldStoreThroughLValue(RValue Src, LValue Dst,
                              const clang::Decl *InitDecl);

  mlir::LogicalResult buildBranchThroughCleanup(JumpDest &Dest,
                                                clang::LabelDecl *L,
                                                mlir::Location Loc);

  void buildScalarInit(const clang::Expr *init, const clang::ValueDecl *D,
                       LValue lvalue);

  LValue buildDeclRefLValue(const clang::DeclRefExpr *E);

  LValue buildBinaryOperatorLValue(const clang::BinaryOperator *E);

  LValue buildUnaryOpLValue(const clang::UnaryOperator *E);

  /// Given an expression of pointer type, try to
  /// derive a more accurate bound on the alignment of the pointer.
  Address buildPointerWithAlignment(const clang::Expr *E,
                                    LValueBaseInfo *BaseInfo);

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

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void buildAutoVarDecl(const clang::VarDecl &D);

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void buildVarDecl(const clang::VarDecl &D);

  /// Perform the usual unary conversions on the specified
  /// expression and compare the result against zero, returning an Int1Ty value.
  mlir::Value evaluateExprAsBool(const clang::Expr *E);

  void buildCtorPrologue(const clang::CXXConstructorDecl *CD,
                         clang::CXXCtorType Type, FunctionArgList &Args);

  void buildConstructorBody(FunctionArgList &Args);

  static bool
  IsConstructorDelegationValid(const clang::CXXConstructorDecl *Ctor);

  struct VPtr {
    clang::BaseSubobject Base;
    const clang::CXXRecordDecl *NearestVBase;
    clang::CharUnits OffsetFromNearestVBase;
    const clang::CXXRecordDecl *VTableClass;
  };

  using VisitedVirtualBasesSetTy =
      llvm::SmallPtrSet<const clang::CXXRecordDecl *, 4>;

  using VPtrsVector = llvm::SmallVector<VPtr, 4>;
  VPtrsVector getVTablePointers(const clang::CXXRecordDecl *VTableClass);
  void getVTablePointers(clang::BaseSubobject Base,
                         const clang::CXXRecordDecl *NearestVBase,
                         clang::CharUnits OffsetFromNearestVBase,
                         bool BaseIsNonVirtualPrimaryBase,
                         const clang::CXXRecordDecl *VTableClass,
                         VisitedVirtualBasesSetTy &VBases, VPtrsVector &vptrs);

  /// A scoep within which we are constructing the fields of an object which
  /// might use a CXXDefaultInitExpr. This stashes away a 'this' value to use if
  /// we need to evaluate the CXXDefaultInitExpr within the evaluation.
  class FieldConstructionScope {
  public:
    FieldConstructionScope(CIRGenFunction &CGF, Address This)
        : CGF(CGF), OldCXXDefaultInitExprThis(CGF.CXXDefaultInitExprThis) {
      CGF.CXXDefaultInitExprThis = This;
    }
    ~FieldConstructionScope() {
      CGF.CXXDefaultInitExprThis = OldCXXDefaultInitExprThis;
    }

  private:
    CIRGenFunction &CGF;
    Address OldCXXDefaultInitExprThis;
  };

  LValue MakeNaturalAlignPointeeAddrLValue(mlir::Operation *Op,
                                           clang::QualType T);

  /// LoadCXXThis - Load the value for 'this'. This function is only valid while
  /// generating code for an C++ member function.
  mlir::Operation *LoadCXXThis() {
    assert(CXXThisValue && "no 'this' value for this function");
    return CXXThisValue;
  }
  Address LoadCXXThisAddress();

  /// Emit code for the start of a function.
  /// \param Loc       The location to be associated with the function.
  /// \param StartLoc  The location of the function body.
  void StartFunction(clang::GlobalDecl GD, clang::QualType RetTy,
                     mlir::FuncOp Fn, const CIRGenFunctionInfo &FnInfo,
                     const FunctionArgList &Args, clang::SourceLocation Loc,
                     clang::SourceLocation StartLoc);

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  mlir::Value buildScalarConversion(mlir::Value Src, clang::QualType SrcTy,
                                    clang::QualType DstTy,
                                    clang::SourceLocation Loc);

  LValue makeAddrLValue(Address Addr, clang::QualType T,
                        LValueBaseInfo BaseInfo) {
    return LValue::makeAddr(Addr, T, getContext(), BaseInfo);
  }

  LValue makeAddrLValue(Address Addr, clang::QualType T,
                        AlignmentSource Source = AlignmentSource::Type) {
    return LValue::makeAddr(Addr, T, getContext(), LValueBaseInfo(Source));
  }

  void initializeVTablePointers(const clang::CXXRecordDecl *RD);

  LValue buildLValueForField(LValue Base, const clang::FieldDecl *Field);

  /// buildLValueForFieldInitialization - like buildLValueForField, excpet that
  /// if the Field is a reference, this will return the address of the reference
  /// and not the address of the value stored in the reference.
  LValue buildLValueForFieldInitialization(LValue Base,
                                           const clang::FieldDecl *Field);

  void buildInitializerForField(clang::FieldDecl *Field, LValue LHS,
                                clang::Expr *Init);

  /// Determine whether the given initializer is trivial in the sense
  /// that it requires no code to be generated.
  bool isTrivialInitializer(const clang::Expr *Init);

  // TODO: this can also be abstrated into common AST helpers
  bool hasBooleanRepresentation(clang::QualType Ty);

  /// GetAddrOfLocalVar - Return the address of a local variable.
  Address GetAddrOfLocalVar(const clang::VarDecl *VD) {
    auto it = LocalDeclMap.find(VD);
    assert(it != LocalDeclMap.end() &&
           "Invalid argument to GetAddrOfLocalVar(), no decl!");
    return it->second;
  }

  void buildDelegateCXXConstructorCall(const clang::CXXConstructorDecl *Ctor,
                                       clang::CXXCtorType CtorType,
                                       const FunctionArgList &Args,
                                       clang::SourceLocation Loc);

  /// buildDelegatingCallArg - We are performing a delegate call; that is, the
  /// current function is delegating to another one. Produce a r-value suitable
  /// for passing the given parameter.
  void buildDelegateCallArg(CallArgList &args, const clang::VarDecl *param,
                            clang::SourceLocation loc);

  /// ShouldInstrumentFunction - Return true if the current function should be
  /// instrumented with __cyg_profile_func_* calls
  bool ShouldInstrumentFunction();
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H
