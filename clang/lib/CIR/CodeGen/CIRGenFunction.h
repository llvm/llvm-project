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

#include "CIRGenBuilder.h"
#include "CIRGenCall.h"
#include "CIRGenDebugInfo.h"
#include "CIRGenModule.h"
#include "CIRGenTBAA.h"
#include "CIRGenTypeCache.h"
#include "CIRGenValue.h"
#include "EHScopeStack.h"

#include "clang/AST/BaseSubobject.h"
#include "clang/AST/CurrentSourceLocExprScope.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/TypeEvaluationKind.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace clang {
class Expr;
} // namespace clang

namespace mlir {
namespace func {
class CallOp;
}
} // namespace mlir

namespace {
class ScalarExprEmitter;
class AggExprEmitter;
} // namespace

namespace clang::CIRGen {

struct CGCoroData;

class CIRGenFunction : public CIRGenTypeCache {
public:
  CIRGenModule &CGM;

private:
  friend class ::ScalarExprEmitter;
  friend class ::AggExprEmitter;

  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  CIRGenBuilderTy &builder;

  /// -------
  /// Goto
  /// -------

  /// A jump destination is an abstract label, branching to which may
  /// require a jump out through normal cleanups.
  struct JumpDest {
    JumpDest() = default;
    JumpDest(mlir::Block *block, EHScopeStack::stable_iterator depth = {},
             unsigned index = 0)
        : block(block) {}

    bool isValid() const { return block != nullptr; }
    mlir::Block *getBlock() const { return block; }
    EHScopeStack::stable_iterator getScopeDepth() const { return scopeDepth; }
    unsigned getDestIndex() const { return index; }

    // This should be used cautiously.
    void setScopeDepth(EHScopeStack::stable_iterator depth) {
      scopeDepth = depth;
    }

  private:
    mlir::Block *block = nullptr;
    EHScopeStack::stable_iterator scopeDepth;
    unsigned index;
  };

  /// Track mlir Blocks for each C/C++ label.
  llvm::DenseMap<const clang::LabelDecl *, JumpDest> LabelMap;
  JumpDest &getJumpDestForLabel(const clang::LabelDecl *D);

  // ---------------------
  // Opaque value handling
  // ---------------------

  /// Keeps track of the current set of opaque value expressions.
  llvm::DenseMap<const OpaqueValueExpr *, LValue> OpaqueLValues;
  llvm::DenseMap<const OpaqueValueExpr *, RValue> OpaqueRValues;

  // This keeps track of the associated size for each VLA type.
  // We track this by the size expression rather than the type itself because
  // in certain situations, like a const qualifier applied to an VLA typedef,
  // multiple VLA types can share the same size expression.
  // FIXME: Maybe this could be a stack of maps that is pushed/popped as we
  // enter/leave scopes.
  llvm::DenseMap<const Expr *, mlir::Value> VLASizeMap;

  /// Add OpenCL kernel arg metadata and the kernel attribute metadata to
  /// the function metadata.
  void emitKernelMetadata(const FunctionDecl *FD, cir::FuncOp Fn);

public:
  /// A non-RAII class containing all the information about a bound
  /// opaque value.  OpaqueValueMapping, below, is a RAII wrapper for
  /// this which makes individual mappings very simple; using this
  /// class directly is useful when you have a variable number of
  /// opaque values or don't want the RAII functionality for some
  /// reason.
  class OpaqueValueMappingData {
    const OpaqueValueExpr *OpaqueValue;
    bool BoundLValue;

    OpaqueValueMappingData(const OpaqueValueExpr *ov, bool boundLValue)
        : OpaqueValue(ov), BoundLValue(boundLValue) {}

  public:
    OpaqueValueMappingData() : OpaqueValue(nullptr) {}

    static bool shouldBindAsLValue(const Expr *expr) {
      // gl-values should be bound as l-values for obvious reasons.
      // Records should be bound as l-values because IR generation
      // always keeps them in memory.  Expressions of function type
      // act exactly like l-values but are formally required to be
      // r-values in C.
      return expr->isGLValue() || expr->getType()->isFunctionType() ||
             hasAggregateEvaluationKind(expr->getType());
    }

    static OpaqueValueMappingData
    bind(CIRGenFunction &CGF, const OpaqueValueExpr *ov, const Expr *e) {
      if (shouldBindAsLValue(ov))
        return bind(CGF, ov, CGF.emitLValue(e));
      return bind(CGF, ov, CGF.emitAnyExpr(e));
    }

    static OpaqueValueMappingData
    bind(CIRGenFunction &CGF, const OpaqueValueExpr *ov, const LValue &lv) {
      assert(shouldBindAsLValue(ov));
      CGF.OpaqueLValues.insert(std::make_pair(ov, lv));
      return OpaqueValueMappingData(ov, true);
    }

    static OpaqueValueMappingData
    bind(CIRGenFunction &CGF, const OpaqueValueExpr *ov, const RValue &rv) {
      assert(!shouldBindAsLValue(ov));
      CGF.OpaqueRValues.insert(std::make_pair(ov, rv));

      OpaqueValueMappingData data(ov, false);

      // Work around an extremely aggressive peephole optimization in
      // EmitScalarConversion which assumes that all other uses of a
      // value are extant.
      assert(!cir::MissingFeatures::peepholeProtection() && "NYI");
      return data;
    }

    bool isValid() const { return OpaqueValue != nullptr; }
    void clear() { OpaqueValue = nullptr; }

    void unbind(CIRGenFunction &CGF) {
      assert(OpaqueValue && "no data to unbind!");

      if (BoundLValue) {
        CGF.OpaqueLValues.erase(OpaqueValue);
      } else {
        CGF.OpaqueRValues.erase(OpaqueValue);
        assert(!cir::MissingFeatures::peepholeProtection() && "NYI");
      }
    }
  };

  /// An RAII object to set (and then clear) a mapping for an OpaqueValueExpr.
  class OpaqueValueMapping {
    CIRGenFunction &CGF;
    OpaqueValueMappingData Data;

  public:
    static bool shouldBindAsLValue(const Expr *expr) {
      return OpaqueValueMappingData::shouldBindAsLValue(expr);
    }

    /// Build the opaque value mapping for the given conditional
    /// operator if it's the GNU ?: extension.  This is a common
    /// enough pattern that the convenience operator is really
    /// helpful.
    ///
    OpaqueValueMapping(CIRGenFunction &CGF,
                       const AbstractConditionalOperator *op)
        : CGF(CGF) {
      if (mlir::isa<ConditionalOperator>(op))
        // Leave Data empty.
        return;

      const BinaryConditionalOperator *e =
          mlir::cast<BinaryConditionalOperator>(op);
      Data = OpaqueValueMappingData::bind(CGF, e->getOpaqueValue(),
                                          e->getCommon());
    }

    /// Build the opaque value mapping for an OpaqueValueExpr whose source
    /// expression is set to the expression the OVE represents.
    OpaqueValueMapping(CIRGenFunction &CGF, const OpaqueValueExpr *OV)
        : CGF(CGF) {
      if (OV) {
        assert(OV->getSourceExpr() && "wrong form of OpaqueValueMapping used "
                                      "for OVE with no source expression");
        Data = OpaqueValueMappingData::bind(CGF, OV, OV->getSourceExpr());
      }
    }

    OpaqueValueMapping(CIRGenFunction &CGF, const OpaqueValueExpr *opaqueValue,
                       LValue lvalue)
        : CGF(CGF),
          Data(OpaqueValueMappingData::bind(CGF, opaqueValue, lvalue)) {}

    OpaqueValueMapping(CIRGenFunction &CGF, const OpaqueValueExpr *opaqueValue,
                       RValue rvalue)
        : CGF(CGF),
          Data(OpaqueValueMappingData::bind(CGF, opaqueValue, rvalue)) {}

    void pop() {
      Data.unbind(CGF);
      Data.clear();
    }

    ~OpaqueValueMapping() {
      if (Data.isValid())
        Data.unbind(CGF);
    }
  };

private:
  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(const clang::Decl *var, clang::QualType ty,
                              mlir::Location loc, clang::CharUnits alignment,
                              mlir::Value &addr, bool isParam = false);

  /// Declare a variable in the current scope but take an Address as input.
  mlir::LogicalResult declare(Address addr, const clang::Decl *var,
                              clang::QualType ty, mlir::Location loc,
                              clang::CharUnits alignment, mlir::Value &addrVal,
                              bool isParam = false);

public:
  // FIXME(cir): move this to CIRGenBuider.h
  mlir::Value emitAlloca(llvm::StringRef name, clang::QualType ty,
                         mlir::Location loc, clang::CharUnits alignment,
                         bool insertIntoFnEntryBlock = false,
                         mlir::Value arraySize = nullptr);
  mlir::Value emitAlloca(llvm::StringRef name, mlir::Type ty,
                         mlir::Location loc, clang::CharUnits alignment,
                         bool insertIntoFnEntryBlock = false,
                         mlir::Value arraySize = nullptr);
  mlir::Value emitAlloca(llvm::StringRef name, mlir::Type ty,
                         mlir::Location loc, clang::CharUnits alignment,
                         mlir::OpBuilder::InsertPoint ip,
                         mlir::Value arraySize = nullptr);

private:
  void emitAndUpdateRetAlloca(clang::QualType ty, mlir::Location loc,
                              clang::CharUnits alignment);

  // Track current variable initialization (if there's one)
  const clang::VarDecl *currVarDecl = nullptr;
  class VarDeclContext {
    CIRGenFunction &P;
    const clang::VarDecl *OldVal = nullptr;

  public:
    VarDeclContext(CIRGenFunction &p, const VarDecl *Value) : P(p) {
      if (P.currVarDecl)
        OldVal = P.currVarDecl;
      P.currVarDecl = Value;
    }

    /// Can be used to restore the state early, before the dtor
    /// is run.
    void restore() { P.currVarDecl = OldVal; }
    ~VarDeclContext() { restore(); }
  };

  /// -------
  /// Source Location tracking
  /// -------

public:
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

  using SymTableScopeTy =
      llvm::ScopedHashTableScope<const clang::Decl *, mlir::Value>;

  enum class EvaluationOrder {
    ///! No langauge constraints on evaluation order.
    Default,
    ///! Language semantics require left-to-right evaluation
    ForceLeftToRight,
    ///! Language semantics require right-to-left evaluation.
    ForceRightToLeft
  };

  /// Situations in which we might emit a check for the suitability of a pointer
  /// or glvalue. Needs to be kept in sync with ubsan_handlers.cpp in
  /// compiler-rt.
  enum TypeCheckKind {
    /// Checking the operand of a load. Must be suitably sized and aligned.
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
    /// Checking the operand of a dynamic_cast or a typeid expression.  Must be
    /// null or an object within its lifetime.
    TCK_DynamicOperation
  };

  // Holds coroutine data if the current function is a coroutine. We use a
  // wrapper to manage its lifetime, so that we don't have to define CGCoroData
  // in this header.
  struct CGCoroInfo {
    std::unique_ptr<CGCoroData> Data;
    CGCoroInfo();
    ~CGCoroInfo();
  };
  CGCoroInfo CurCoro;

  bool isCoroutine() const { return CurCoro.Data != nullptr; }

  /// The GlobalDecl for the current function being compiled.
  clang::GlobalDecl CurGD;

  /// Unified return block.
  /// In CIR this is a function because each scope might have
  /// it's associated return block.
  JumpDest returnBlock(mlir::Block *retBlock) {
    return getJumpDestInCurrentScope(retBlock);
  }

  unsigned nextCleanupDestIndex = 1;

  /// The temporary alloca to hold the return value. This is
  /// invalid iff the function has no return value.
  Address ReturnValue = Address::invalid();

  /// Tracks function scope overall cleanup handling.
  EHScopeStack EHStack;
  llvm::SmallVector<char, 256> LifetimeExtendedCleanupStack;

  // A stack of cleanups which were added to EHStack but have to be deactivated
  // later before being popped or emitted. These are usually deactivated on
  // exiting a `CleanupDeactivationScope` scope. For instance, after a
  // full-expr.
  //
  // These are specially useful for correctly emitting cleanups while
  // encountering branches out of expression (through stmt-expr or coroutine
  // suspensions).
  struct DeferredDeactivateCleanup {
    EHScopeStack::stable_iterator Cleanup;
    mlir::Operation *DominatingIP;
  };
  llvm::SmallVector<DeferredDeactivateCleanup> DeferredDeactivationCleanupStack;

  // Enters a new scope for capturing cleanups which are deferred to be
  // deactivated, all of which will be deactivated once the scope is exited.
  struct CleanupDeactivationScope {
    CIRGenFunction &CGF;
    size_t OldDeactivateCleanupStackSize;
    bool Deactivated;
    CleanupDeactivationScope(CIRGenFunction &CGF)
        : CGF(CGF), OldDeactivateCleanupStackSize(
                        CGF.DeferredDeactivationCleanupStack.size()),
          Deactivated(false) {}

    void ForceDeactivate() {
      assert(!Deactivated && "Deactivating already deactivated scope");
      auto &Stack = CGF.DeferredDeactivationCleanupStack;
      for (size_t I = Stack.size(); I > OldDeactivateCleanupStackSize; I--) {
        CGF.DeactivateCleanupBlock(Stack[I - 1].Cleanup,
                                   Stack[I - 1].DominatingIP);
        Stack[I - 1].DominatingIP->erase();
      }
      Stack.resize(OldDeactivateCleanupStackSize);
      Deactivated = true;
    }

    ~CleanupDeactivationScope() {
      if (Deactivated)
        return;
      ForceDeactivate();
    }
  };

  /// A mapping from NRVO variables to the flags used to indicate
  /// when the NRVO has been applied to this variable.
  llvm::DenseMap<const VarDecl *, mlir::Value> NRVOFlags;

  /// Counts of the number return expressions in the function.
  unsigned NumReturnExprs = 0;

  clang::QualType FnRetQualTy;
  std::optional<mlir::Type> FnRetCIRTy;
  std::optional<mlir::Value> FnRetAlloca;

  llvm::DenseMap<const clang::ValueDecl *, clang::FieldDecl *>
      LambdaCaptureFields;
  clang::FieldDecl *LambdaThisCaptureField = nullptr;

  void emitForwardingCallToLambda(const CXXMethodDecl *LambdaCallOperator,
                                  CallArgList &CallArgs);
  void emitLambdaDelegatingInvokeBody(const CXXMethodDecl *MD);
  void emitLambdaStaticInvokeBody(const CXXMethodDecl *MD);

  LValue emitPredefinedLValue(const PredefinedExpr *E);

  /// When generating code for a C++ member function, this will
  /// hold the implicit 'this' declaration.
  clang::ImplicitParamDecl *CXXABIThisDecl = nullptr;
  mlir::Value CXXABIThisValue = nullptr;
  mlir::Value CXXThisValue = nullptr;
  clang::CharUnits CXXABIThisAlignment;
  clang::CharUnits CXXThisAlignment;

  /// When generating code for a constructor or destructor, this will hold the
  /// implicit argument (e.g. VTT).
  ImplicitParamDecl *CXXStructorImplicitParamDecl{};
  mlir::Value CXXStructorImplicitParamValue{};

  /// The value of 'this' to sue when evaluating CXXDefaultInitExprs within this
  /// expression.
  Address CXXDefaultInitExprThis = Address::invalid();

  // Holds the Decl for the current outermost non-closure context
  const clang::Decl *CurFuncDecl = nullptr;
  /// This is the inner-most code context, which includes blocks.
  const clang::Decl *CurCodeDecl = nullptr;
  const CIRGenFunctionInfo *CurFnInfo = nullptr;
  clang::QualType FnRetTy;

  /// This is the current function or global initializer that is generated code
  /// for.
  mlir::Operation *CurFn = nullptr;

  /// Save Parameter Decl for coroutine.
  llvm::SmallVector<const ParmVarDecl *, 4> FnArgs;

  // The CallExpr within the current statement that the musttail attribute
  // applies to. nullptr if there is no 'musttail' on the current statement.
  const clang::CallExpr *MustTailCall = nullptr;

  /// The type of the condition for the emitting switch statement.
  llvm::SmallVector<mlir::Type, 2> condTypeStack;

  clang::ASTContext &getContext() const;

  CIRGenBuilderTy &getBuilder() { return builder; }

  CIRGenModule &getCIRGenModule() { return CGM; }
  const CIRGenModule &getCIRGenModule() const { return CGM; }

  mlir::Block *getCurFunctionEntryBlock() {
    auto Fn = mlir::dyn_cast<cir::FuncOp>(CurFn);
    assert(Fn && "other callables NYI");
    return &Fn.getRegion().front();
  }

  /// Sanitizers enabled for this function.
  clang::SanitizerSet SanOpts;

  class CIRGenFPOptionsRAII {
  public:
    CIRGenFPOptionsRAII(CIRGenFunction &CGF, FPOptions FPFeatures);
    CIRGenFPOptionsRAII(CIRGenFunction &CGF, const clang::Expr *E);
    ~CIRGenFPOptionsRAII();

  private:
    void ConstructorHelper(clang::FPOptions FPFeatures);
    CIRGenFunction &CGF;
    clang::FPOptions OldFPFeatures;
    cir::fp::ExceptionBehavior OldExcept;
    llvm::RoundingMode OldRounding;
  };
  clang::FPOptions CurFPFeatures;

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
  /// This keeps track of the CIR allocas or globals for local C
  /// delcs.
  DeclMapTy LocalDeclMap;

  CIRGenDebugInfo *debugInfo = nullptr;

  /// Whether llvm.stacksave has been called. Used to avoid
  /// calling llvm.stacksave for multiple VLAs in the same scope.
  /// TODO: Translate to MLIR
  bool DidCallStackSave = false;

  /// Whether we processed a Microsoft-style asm block during CIRGen. These can
  /// potentially set the return value.
  bool SawAsmBlock = false;

  /// True if CodeGen currently emits code inside preserved access index region.
  bool IsInPreservedAIRegion = false;

  /// In C++, whether we are code generating a thunk. This controls whether we
  /// should emit cleanups.
  bool CurFuncIsThunk = false;

  /// Hold counters for incrementally naming temporaries
  unsigned CounterRefTmp = 0;
  unsigned CounterAggTmp = 0;
  std::string getCounterRefTmpAsString();
  std::string getCounterAggTmpAsString();

  mlir::Type convertTypeForMem(QualType T);

  mlir::Type convertType(clang::QualType T);
  mlir::Type convertType(const TypeDecl *T) {
    return convertType(getContext().getTypeDeclType(T));
  }

  ///  Return the cir::TypeEvaluationKind of QualType \c T.
  static cir::TypeEvaluationKind getEvaluationKind(clang::QualType T);

  static bool hasScalarEvaluationKind(clang::QualType T) {
    return getEvaluationKind(T) == cir::TEK_Scalar;
  }

  static bool hasAggregateEvaluationKind(clang::QualType T) {
    return getEvaluationKind(T) == cir::TEK_Aggregate;
  }

  CIRGenFunction(CIRGenModule &CGM, CIRGenBuilderTy &builder,
                 bool suppressNewContext = false);
  ~CIRGenFunction();

  CIRGenTypes &getTypes() const { return CGM.getTypes(); }

  const TargetInfo &getTarget() const { return CGM.getTarget(); }
  mlir::MLIRContext &getMLIRContext() { return CGM.getMLIRContext(); }

  const TargetCIRGenInfo &getTargetHooks() const {
    return CGM.getTargetCIRGenInfo();
  }

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(clang::SourceLocation SLoc);

  mlir::Location getLoc(clang::SourceRange SLoc);

  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  const clang::LangOptions &getLangOpts() const { return CGM.getLangOpts(); }

  CIRGenDebugInfo *getDebugInfo() { return debugInfo; }

  void emitReturnOfRValue(mlir::Location loc, RValue RV, QualType Ty);

  /// Set the address of a local variable.
  void setAddrOfLocalVar(const clang::VarDecl *VD, Address Addr) {
    assert(!LocalDeclMap.count(VD) && "Decl already exists in LocalDeclMap!");
    LocalDeclMap.insert({VD, Addr});
    // Add to the symbol table if not there already.
    if (symbolTable.count(VD))
      return;
    symbolTable.insert(VD, Addr.getPointer());
  }

  /// True if an insertion point is defined. If not, this indicates that the
  /// current code being emitted is unreachable.
  /// FIXME(cir): we need to inspect this and perhaps use a cleaner mechanism
  /// since we don't yet force null insertion point to designate behavior (like
  /// LLVM's codegen does) and we probably shouldn't.
  bool HaveInsertPoint() const {
    return builder.getInsertionBlock() != nullptr;
  }

  /// Whether any type-checking sanitizers are enabled. If \c false, calls to
  /// emitTypeCheck can be skipped.
  bool sanitizePerformTypeCheck() const;

  void emitTypeCheck(TypeCheckKind TCK, clang::SourceLocation Loc,
                     mlir::Value V, clang::QualType Type,
                     clang::CharUnits Alignment = clang::CharUnits::Zero(),
                     clang::SanitizerSet SkippedChecks = clang::SanitizerSet(),
                     std::optional<mlir::Value> ArraySize = std::nullopt);

  void emitAggExpr(const clang::Expr *E, AggValueSlot Slot);

  /// Emit the computation of the specified expression of complex type,
  /// returning the result.
  mlir::Value emitComplexExpr(const Expr *E);

  void emitComplexExprIntoLValue(const Expr *E, LValue dest, bool isInit);

  void emitStoreOfComplex(mlir::Location Loc, mlir::Value V, LValue dest,
                          bool isInit);

  Address emitAddrOfRealComponent(mlir::Location loc, Address complex,
                                  QualType complexType);
  Address emitAddrOfImagComponent(mlir::Location loc, Address complex,
                                  QualType complexType);

  LValue emitComplexAssignmentLValue(const BinaryOperator *E);
  LValue emitComplexCompoundAssignmentLValue(const CompoundAssignOperator *E);

  /// Emits a reference binding to the passed in expression.
  RValue emitReferenceBindingToExpr(const Expr *E);

  LValue emitCastLValue(const CastExpr *E);

  void emitCXXConstructExpr(const clang::CXXConstructExpr *E,
                            AggValueSlot Dest);

  /// Emit a call to an inheriting constructor (that is, one that invokes a
  /// constructor inherited from a base class) by inlining its definition. This
  /// is necessary if the ABI does not support forwarding the arguments to the
  /// base class constructor (because they're variadic or similar).
  void emitInlinedInheritingCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                               CXXCtorType CtorType,
                                               bool ForVirtualBase,
                                               bool Delegating,
                                               CallArgList &Args);

  /// Emit a call to a constructor inherited from a base class, passing the
  /// current constructor's arguments along unmodified (without even making
  /// a copy).
  void emitInheritedCXXConstructorCall(const CXXConstructorDecl *D,
                                       bool ForVirtualBase, Address This,
                                       bool InheritedFromVBase,
                                       const CXXInheritedCtorInitExpr *E);

  void emitCXXConstructorCall(const clang::CXXConstructorDecl *D,
                              clang::CXXCtorType Type, bool ForVirtualBase,
                              bool Delegating, AggValueSlot ThisAVS,
                              const clang::CXXConstructExpr *E);

  void emitCXXConstructorCall(const clang::CXXConstructorDecl *D,
                              clang::CXXCtorType Type, bool ForVirtualBase,
                              bool Delegating, Address This, CallArgList &Args,
                              AggValueSlot::Overlap_t Overlap,
                              clang::SourceLocation Loc,
                              bool NewPointerIsChecked);

  RValue emitCXXMemberOrOperatorCall(
      const clang::CXXMethodDecl *Method, const CIRGenCallee &Callee,
      ReturnValueSlot ReturnValue, mlir::Value This, mlir::Value ImplicitParam,
      clang::QualType ImplicitParamTy, const clang::CallExpr *E,
      CallArgList *RtlArgs);

  RValue emitCXXMemberCallExpr(const clang::CXXMemberCallExpr *E,
                               ReturnValueSlot ReturnValue);
  RValue emitCXXMemberPointerCallExpr(const CXXMemberCallExpr *E,
                                      ReturnValueSlot ReturnValue);
  RValue emitCXXMemberOrOperatorMemberCallExpr(
      const clang::CallExpr *CE, const clang::CXXMethodDecl *MD,
      ReturnValueSlot ReturnValue, bool HasQualifier,
      clang::NestedNameSpecifier *Qualifier, bool IsArrow,
      const clang::Expr *Base);
  RValue emitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                       const CXXMethodDecl *MD,
                                       ReturnValueSlot ReturnValue);
  void emitNullInitialization(mlir::Location loc, Address DestPtr, QualType Ty);
  bool shouldNullCheckClassCastValue(const CastExpr *CE);

  void emitCXXTemporary(const CXXTemporary *Temporary, QualType TempType,
                        Address Ptr);
  mlir::Value emitCXXNewExpr(const CXXNewExpr *E);
  void emitCXXDeleteExpr(const CXXDeleteExpr *E);

  void emitNewArrayInitializer(const CXXNewExpr *E, QualType ElementType,
                               mlir::Type ElementTy, Address BeginPtr,
                               mlir::Value NumElements,
                               mlir::Value AllocSizeWithoutCookie);

  void emitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                  const clang::ArrayType *ArrayTy,
                                  Address ArrayPtr, const CXXConstructExpr *E,
                                  bool NewPointerIsChecked,
                                  bool ZeroInitialization = false);

  void emitCXXAggrConstructorCall(const CXXConstructorDecl *ctor,
                                  mlir::Value numElements, Address arrayBase,
                                  const CXXConstructExpr *E,
                                  bool NewPointerIsChecked,
                                  bool zeroInitialize);

  /// Compute the length of an array, even if it's a VLA, and drill down to the
  /// base element type.
  mlir::Value emitArrayLength(const clang::ArrayType *arrayType,
                              QualType &baseType, Address &addr);

  void emitDeleteCall(const FunctionDecl *DeleteFD, mlir::Value Ptr,
                      QualType DeleteTy, mlir::Value NumElements = nullptr,
                      CharUnits CookieSize = CharUnits());

  RValue emitBuiltinNewDeleteCall(const FunctionProtoType *type,
                                  const CallExpr *theCallExpr, bool isDelete);

  mlir::Value emitDynamicCast(Address ThisAddr, const CXXDynamicCastExpr *DCE);

  mlir::Value createLoad(const clang::VarDecl *VD, const char *Name);

  mlir::Value emitScalarPrePostIncDec(const UnaryOperator *E, LValue LV,
                                      bool isInc, bool isPre);
  mlir::Value emitComplexPrePostIncDec(const UnaryOperator *E, LValue LV,
                                       bool isInc, bool isPre);

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

  /// If a ParmVarDecl had the pass_object_size attribute, this
  /// will contain a mapping from said ParmVarDecl to its implicit "object_size"
  /// parameter.
  llvm::SmallDenseMap<const ParmVarDecl *, const ImplicitParamDecl *, 2>
      SizeArguments;

  // Build a "reference" to a va_list; this is either the address or the value
  // of the expression, depending on how va_list is defined.
  Address emitVAListRef(const Expr *E);

  /// Emits a CIR variable-argument operation, either
  /// \c cir.va.start or \c cir.va.end.
  ///
  /// \param ArgValue A reference to the \c va_list as emitted by either
  /// \c emitVAListRef or \c emitMSVAListRef.
  ///
  /// \param IsStart If \c true, emits \c cir.va.start, otherwise \c cir.va.end.
  void emitVAStartEnd(mlir::Value ArgValue, bool IsStart);

  /// Generate code to get an argument from the passed in pointer
  /// and update it accordingly.
  ///
  /// \param VE The \c VAArgExpr for which to generate code.
  ///
  /// \param VAListAddr Receives a reference to the \c va_list as emitted by
  /// either \c emitVAListRef or \c emitMSVAListRef.
  ///
  /// \returns SSA value with the argument.
  mlir::Value emitVAArg(VAArgExpr *VE, Address &VAListAddr);

  void emitVariablyModifiedType(QualType Ty);

  struct VlaSizePair {
    mlir::Value NumElts;
    QualType Type;

    VlaSizePair(mlir::Value NE, QualType T) : NumElts(NE), Type(T) {}
  };

  /// Returns an MLIR value that corresponds to the size,
  /// in non-variably-sized elements, of a variable length array type,
  /// plus that largest non-variably-sized element type.  Assumes that
  /// the type has already been emitted with emitVariablyModifiedType.
  VlaSizePair getVLASize(const VariableArrayType *vla);
  VlaSizePair getVLASize(QualType vla);

  mlir::Value emitBuiltinObjectSize(const Expr *E, unsigned Type,
                                    cir::IntType ResType, mlir::Value EmittedE,
                                    bool IsDynamic);
  mlir::Value evaluateOrEmitBuiltinObjectSize(const Expr *E, unsigned Type,
                                              cir::IntType ResType,
                                              mlir::Value EmittedE,
                                              bool IsDynamic);

  /// Given an expression that represents a value lvalue, this method emits
  /// the address of the lvalue, then loads the result as an rvalue,
  /// returning the rvalue.
  RValue emitLoadOfLValue(LValue LV, SourceLocation Loc);
  mlir::Value emitLoadOfScalar(Address addr, bool isVolatile,
                               clang::QualType ty, clang::SourceLocation loc,
                               LValueBaseInfo baseInfo, TBAAAccessInfo tbaaInfo,
                               bool isNontemporal = false);
  mlir::Value emitLoadOfScalar(Address addr, bool isVolatile,
                               clang::QualType ty, mlir::Location loc,
                               LValueBaseInfo baseInfo, TBAAAccessInfo tbaaInfo,
                               bool isNontemporal = false);

  int64_t getAccessedFieldNo(unsigned idx, const mlir::ArrayAttr elts);

  RValue emitLoadOfExtVectorElementLValue(LValue LV);

  void emitStoreThroughExtVectorComponentLValue(RValue Src, LValue Dst);

  RValue emitLoadOfBitfieldLValue(LValue LV, SourceLocation Loc);

  /// Load a scalar value from an address, taking care to appropriately convert
  /// from the memory representation to CIR value representation.
  mlir::Value emitLoadOfScalar(Address addr, bool isVolatile,
                               clang::QualType ty, clang::SourceLocation loc,
                               AlignmentSource source = AlignmentSource::Type,
                               bool isNontemporal = false) {
    return emitLoadOfScalar(addr, isVolatile, ty, loc, LValueBaseInfo(source),
                            CGM.getTBAAAccessInfo(ty), isNontemporal);
  }

  /// Load a scalar value from an address, taking care to appropriately convert
  /// form the memory representation to the CIR value representation. The
  /// l-value must be a simple l-value.
  mlir::Value emitLoadOfScalar(LValue lvalue, clang::SourceLocation Loc);
  mlir::Value emitLoadOfScalar(LValue lvalue, mlir::Location Loc);

  /// Load a complex number from the specified l-value.
  mlir::Value emitLoadOfComplex(LValue src, SourceLocation loc);

  Address emitLoadOfReference(LValue refLVal, mlir::Location loc,
                              LValueBaseInfo *pointeeBaseInfo = nullptr,
                              TBAAAccessInfo *pointeeTBAAInfo = nullptr);
  LValue emitLoadOfReferenceLValue(LValue RefLVal, mlir::Location Loc);
  LValue
  emitLoadOfReferenceLValue(Address RefAddr, mlir::Location Loc, QualType RefTy,
                            AlignmentSource Source = AlignmentSource::Type) {
    LValue RefLVal = makeAddrLValue(RefAddr, RefTy, LValueBaseInfo(Source),
                                    CGM.getTBAAAccessInfo(RefTy));
    return emitLoadOfReferenceLValue(RefLVal, Loc);
  }
  void emitImplicitAssignmentOperatorBody(FunctionArgList &Args);

  void emitAggregateStore(mlir::Value Val, Address Dest, bool DestIsVolatile);

  void emitCallArgs(
      CallArgList &Args, PrototypeWrapper Prototype,
      llvm::iterator_range<clang::CallExpr::const_arg_iterator> ArgRange,
      AbstractCallee AC = AbstractCallee(), unsigned ParamsToSkip = 0,
      EvaluationOrder Order = EvaluationOrder::Default);

  void checkTargetFeatures(const CallExpr *E, const FunctionDecl *TargetDecl);
  void checkTargetFeatures(SourceLocation Loc, const FunctionDecl *TargetDecl);

  LValue emitStmtExprLValue(const StmtExpr *E);

  LValue emitPointerToDataMemberBinaryExpr(const BinaryOperator *E);

  /// TODO: Add TBAAAccessInfo
  Address emitCXXMemberDataPointerAddress(
      const Expr *E, Address base, mlir::Value memberPtr,
      const MemberPointerType *memberPtrType, LValueBaseInfo *baseInfo,
      TBAAAccessInfo *tbaaInfo);

  /// Generate a call of the given function, expecting the given
  /// result type, and using the given argument list which specifies both the
  /// LLVM arguments and the types they were derived from.
  RValue emitCall(const CIRGenFunctionInfo &CallInfo,
                  const CIRGenCallee &Callee, ReturnValueSlot ReturnValue,
                  const CallArgList &Args,
                  cir::CIRCallOpInterface *callOrTryCall, bool IsMustTail,
                  mlir::Location loc,
                  std::optional<const clang::CallExpr *> E = std::nullopt);
  RValue emitCall(const CIRGenFunctionInfo &CallInfo,
                  const CIRGenCallee &Callee, ReturnValueSlot ReturnValue,
                  const CallArgList &Args,
                  cir::CIRCallOpInterface *callOrTryCall = nullptr,
                  bool IsMustTail = false) {
    assert(currSrcLoc && "source location must have been set");
    return emitCall(CallInfo, Callee, ReturnValue, Args, callOrTryCall,
                    IsMustTail, *currSrcLoc, std::nullopt);
  }
  RValue emitCall(clang::QualType FnType, const CIRGenCallee &Callee,
                  const clang::CallExpr *E, ReturnValueSlot returnValue,
                  mlir::Value Chain = nullptr);

  RValue emitCallExpr(const clang::CallExpr *E,
                      ReturnValueSlot ReturnValue = ReturnValueSlot());

  Address getAsNaturalAddressOf(Address Addr, QualType PointeeTy);

  mlir::Value getAsNaturalPointerTo(Address Addr, QualType PointeeType) {
    return getAsNaturalAddressOf(Addr, PointeeType).getBasePointer();
  }

  mlir::Value emitRuntimeCall(mlir::Location loc, cir::FuncOp callee,
                              llvm::ArrayRef<mlir::Value> args = {});

  void emitInvariantStart(CharUnits Size);

  /// Create a check for a function parameter that may potentially be
  /// declared as non-null.
  void emitNonNullArgCheck(RValue RV, QualType ArgType, SourceLocation ArgLoc,
                           AbstractCallee AC, unsigned ParmNum);

  void emitCallArg(CallArgList &args, const clang::Expr *E,
                   clang::QualType ArgType);

  LValue emitCallExprLValue(const CallExpr *E);

  /// Similarly to emitAnyExpr(), however, the result will always be accessible
  /// even if no aggregate location is provided.
  RValue emitAnyExprToTemp(const clang::Expr *E);

  CIRGenCallee emitCallee(const clang::Expr *E);

  void finishFunction(SourceLocation EndLoc);

  /// Emit code to compute the specified expression which can have any type. The
  /// result is returned as an RValue struct. If this is an aggregate
  /// expression, the aggloc/agglocvolatile arguments indicate where the result
  /// should be returned.
  RValue emitAnyExpr(const clang::Expr *E,
                     AggValueSlot aggSlot = AggValueSlot::ignored(),
                     bool ignoreResult = false);

  mlir::LogicalResult emitFunctionBody(const clang::Stmt *Body);
  mlir::LogicalResult emitCoroutineBody(const CoroutineBodyStmt &S);
  mlir::LogicalResult emitCoreturnStmt(const CoreturnStmt &S);

  cir::CallOp emitCoroIDBuiltinCall(mlir::Location loc, mlir::Value nullPtr);
  cir::CallOp emitCoroAllocBuiltinCall(mlir::Location loc);
  cir::CallOp emitCoroBeginBuiltinCall(mlir::Location loc,
                                       mlir::Value coroframeAddr);
  cir::CallOp emitCoroEndBuiltinCall(mlir::Location loc, mlir::Value nullPtr);

  RValue emitCoawaitExpr(const CoawaitExpr &E,
                         AggValueSlot aggSlot = AggValueSlot::ignored(),
                         bool ignoreResult = false);
  RValue emitCoyieldExpr(const CoyieldExpr &E,
                         AggValueSlot aggSlot = AggValueSlot::ignored(),
                         bool ignoreResult = false);
  RValue emitCoroutineIntrinsic(const CallExpr *E, unsigned int IID);
  RValue emitCoroutineFrame();

  // Many of MSVC builtins are on x64, ARM and AArch64; to avoid repeating code,
  // we handle them here.
  enum class MSVCIntrin {
    _BitScanForward,
    _BitScanReverse,
    _InterlockedAnd,
    _InterlockedDecrement,
    _InterlockedExchange,
    _InterlockedExchangeAdd,
    _InterlockedExchangeSub,
    _InterlockedIncrement,
    _InterlockedOr,
    _InterlockedXor,
    _InterlockedExchangeAdd_acq,
    _InterlockedExchangeAdd_rel,
    _InterlockedExchangeAdd_nf,
    _InterlockedExchange_acq,
    _InterlockedExchange_rel,
    _InterlockedExchange_nf,
    _InterlockedCompareExchange_acq,
    _InterlockedCompareExchange_rel,
    _InterlockedCompareExchange_nf,
    _InterlockedCompareExchange128,
    _InterlockedCompareExchange128_acq,
    _InterlockedCompareExchange128_rel,
    _InterlockedCompareExchange128_nf,
    _InterlockedOr_acq,
    _InterlockedOr_rel,
    _InterlockedOr_nf,
    _InterlockedXor_acq,
    _InterlockedXor_rel,
    _InterlockedXor_nf,
    _InterlockedAnd_acq,
    _InterlockedAnd_rel,
    _InterlockedAnd_nf,
    _InterlockedIncrement_acq,
    _InterlockedIncrement_rel,
    _InterlockedIncrement_nf,
    _InterlockedDecrement_acq,
    _InterlockedDecrement_rel,
    _InterlockedDecrement_nf,
    __fastfail,
  };

  mlir::Value emitARMMVEBuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                    ReturnValueSlot ReturnValue,
                                    llvm::Triple::ArchType Arch);
  mlir::Value emitARMCDEBuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                    ReturnValueSlot ReturnValue,
                                    llvm::Triple::ArchType Arch);
  mlir::Value emitCommonNeonBuiltinExpr(
      unsigned builtinID, unsigned llvmIntrinsic, unsigned altLLVMIntrinsic,
      const char *nameHint, unsigned modifier, const CallExpr *e,
      llvm::SmallVectorImpl<mlir::Value> &ops, Address ptrOp0, Address ptrOp1,
      llvm::Triple::ArchType arch);

  mlir::Value emitAlignmentAssumption(mlir::Value ptrValue, QualType ty,
                                      SourceLocation loc,
                                      SourceLocation assumptionLoc,
                                      mlir::IntegerAttr alignment,
                                      mlir::Value offsetValue = nullptr);

  mlir::Value emitAlignmentAssumption(mlir::Value ptrValue, const Expr *expr,
                                      SourceLocation assumptionLoc,
                                      mlir::IntegerAttr alignment,
                                      mlir::Value offsetValue = nullptr);

  /// Build a debug stoppoint if we are emitting debug info.
  void emitStopPoint(const Stmt *S);

  // Build CIR for a statement. useCurrentScope should be true if no
  // new scopes need be created when finding a compound statement.
  mlir::LogicalResult
  emitStmt(const clang::Stmt *S, bool useCurrentScope,
           llvm::ArrayRef<const Attr *> Attrs = std::nullopt);

  mlir::LogicalResult emitSimpleStmt(const clang::Stmt *S,
                                     bool useCurrentScope);

  mlir::LogicalResult emitForStmt(const clang::ForStmt &S);
  mlir::LogicalResult emitWhileStmt(const clang::WhileStmt &S);
  mlir::LogicalResult emitDoStmt(const clang::DoStmt &S);
  mlir::LogicalResult
  emitCXXForRangeStmt(const CXXForRangeStmt &S,
                      llvm::ArrayRef<const Attr *> Attrs = std::nullopt);
  mlir::LogicalResult emitSwitchStmt(const clang::SwitchStmt &S);

  mlir::LogicalResult emitCXXTryStmtUnderScope(const clang::CXXTryStmt &S);
  mlir::LogicalResult emitCXXTryStmt(const clang::CXXTryStmt &S);
  void enterCXXTryStmt(const CXXTryStmt &S, cir::TryOp catchOp,
                       bool IsFnTryBlock = false);
  void exitCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock = false);

  Address emitCompoundStmt(const clang::CompoundStmt &S, bool getLast = false,
                           AggValueSlot slot = AggValueSlot::ignored());

  Address
  emitCompoundStmtWithoutScope(const clang::CompoundStmt &S,
                               bool getLast = false,
                               AggValueSlot slot = AggValueSlot::ignored());
  GlobalDecl CurSEHParent;
  bool currentFunctionUsesSEHTry() const { return !!CurSEHParent; }

  /// Returns true inside SEH __try blocks.
  bool isSEHTryScope() const { return cir::MissingFeatures::isSEHTryScope(); }

  mlir::Operation *CurrentFuncletPad = nullptr;

  /// Returns true while emitting a cleanuppad.
  bool isCleanupPadScope() const {
    assert(!CurrentFuncletPad && "NYI");
    return false;
  }

  /// Return a landing pad that just calls terminate.
  mlir::Operation *getTerminateLandingPad();

  /// Emit code to compute the specified expression,
  /// ignoring the result.
  void emitIgnoredExpr(const clang::Expr *E);

  LValue emitArraySubscriptExpr(const clang::ArraySubscriptExpr *E,
                                bool Accessed = false);

  mlir::LogicalResult emitDeclStmt(const clang::DeclStmt &S);

  /// Determine whether a return value slot may overlap some other object.
  AggValueSlot::Overlap_t getOverlapForReturnValue() {
    // FIXME: Assuming no overlap here breaks guaranteed copy elision for base
    // class subobjects. These cases may need to be revisited depending on the
    // resolution of the relevant core issue.
    return AggValueSlot::DoesNotOverlap;
  }

  /// Determine whether a base class initialization may overlap some other
  /// object.
  AggValueSlot::Overlap_t getOverlapForBaseInit(const CXXRecordDecl *RD,
                                                const CXXRecordDecl *BaseRD,
                                                bool IsVirtual);

  /// Get an appropriate 'undef' rvalue for the given type.
  /// TODO: What's the equivalent for MLIR? Currently we're only using this for
  /// void types so it just returns RValue::get(nullptr) but it'll need
  /// addressed later.
  RValue GetUndefRValue(clang::QualType Ty);

  /// Given a value and its clang type, returns the value casted from its memory
  /// representation.
  /// Note: CIR defers most of the special casting to the final lowering passes
  /// to conserve the high level information.
  mlir::Value emitFromMemory(mlir::Value Value, clang::QualType Ty);

  mlir::LogicalResult emitAsmStmt(const clang::AsmStmt &S);

  std::pair<mlir::Value, mlir::Type>
  emitAsmInputLValue(const TargetInfo::ConstraintInfo &Info, LValue InputValue,
                     QualType InputType, std::string &ConstraintStr,
                     SourceLocation Loc);

  std::pair<mlir::Value, mlir::Type>
  emitAsmInput(const TargetInfo::ConstraintInfo &Info, const Expr *InputExpr,
               std::string &ConstraintStr);

  mlir::LogicalResult emitIfStmt(const clang::IfStmt &S);

  mlir::LogicalResult emitReturnStmt(const clang::ReturnStmt &S);

  mlir::LogicalResult emitGotoStmt(const clang::GotoStmt &S);

  mlir::LogicalResult emitLabel(const clang::LabelDecl *D);
  mlir::LogicalResult emitLabelStmt(const clang::LabelStmt &S);

  mlir::LogicalResult emitAttributedStmt(const AttributedStmt &S);

  mlir::LogicalResult emitBreakStmt(const clang::BreakStmt &S);
  mlir::LogicalResult emitContinueStmt(const clang::ContinueStmt &S);

  // OpenMP gen functions:
  mlir::LogicalResult emitOMPParallelDirective(const OMPParallelDirective &S);
  mlir::LogicalResult emitOMPTaskwaitDirective(const OMPTaskwaitDirective &S);
  mlir::LogicalResult emitOMPTaskyieldDirective(const OMPTaskyieldDirective &S);
  mlir::LogicalResult emitOMPBarrierDirective(const OMPBarrierDirective &S);

  LValue emitOpaqueValueLValue(const OpaqueValueExpr *e);

  /// Emit code to compute a designator that specifies the location
  /// of the expression.
  /// FIXME: document this function better.
  LValue emitLValue(const clang::Expr *E);

  void emitDecl(const clang::Decl &D);

  /// Emit local annotations for the local variable V, declared by D.
  void emitVarAnnotations(const VarDecl *decl, mlir::Value val);

  /// If the specified expression does not fold to a constant, or if it does but
  /// contains a label, return false.  If it constant folds return true and set
  /// the boolean result in Result.
  bool ConstantFoldsToSimpleInteger(const clang::Expr *Cond, bool &ResultBool,
                                    bool AllowLabels = false);
  bool ConstantFoldsToSimpleInteger(const clang::Expr *Cond,
                                    llvm::APSInt &ResultInt,
                                    bool AllowLabels = false);

  /// Return true if the statement contains a label in it.  If
  /// this statement is not executed normally, it not containing a label means
  /// that we can just remove the code.
  bool ContainsLabel(const clang::Stmt *S, bool IgnoreCaseStmts = false);

  /// Emit an if on a boolean condition to the specified blocks.
  /// FIXME: Based on the condition, this might try to simplify the codegen of
  /// the conditional based on the branch. TrueCount should be the number of
  /// times we expect the condition to evaluate to true based on PGO data. We
  /// might decide to leave this as a separate pass (see EmitBranchOnBoolExpr
  /// for extra ideas).
  mlir::LogicalResult emitIfOnBoolExpr(const clang::Expr *cond,
                                       const clang::Stmt *thenS,
                                       const clang::Stmt *elseS);
  cir::IfOp emitIfOnBoolExpr(
      const clang::Expr *cond,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> thenBuilder,
      mlir::Location thenLoc,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> elseBuilder,
      std::optional<mlir::Location> elseLoc = {});
  mlir::Value emitTernaryOnBoolExpr(const clang::Expr *cond, mlir::Location loc,
                                    const clang::Stmt *thenS,
                                    const clang::Stmt *elseS);
  mlir::Value emitOpOnBoolExpr(mlir::Location loc, const clang::Expr *cond);

  class ConstantEmission {
    // Cannot use mlir::TypedAttr directly here because of bit availability.
    llvm::PointerIntPair<mlir::Attribute, 1, bool> ValueAndIsReference;
    ConstantEmission(mlir::TypedAttr C, bool isReference)
        : ValueAndIsReference(C, isReference) {}

  public:
    ConstantEmission() {}
    static ConstantEmission forReference(mlir::TypedAttr C) {
      return ConstantEmission(C, true);
    }
    static ConstantEmission forValue(mlir::TypedAttr C) {
      return ConstantEmission(C, false);
    }

    explicit operator bool() const {
      return ValueAndIsReference.getOpaqueValue() != nullptr;
    }

    bool isReference() const { return ValueAndIsReference.getInt(); }
    LValue getReferenceLValue(CIRGenFunction &CGF, Expr *refExpr) const {
      assert(isReference());
      // create<cir::ConstantOp>(loc, ty, getZeroAttr(ty));
      // CGF.getBuilder().const
      // return CGF.MakeNaturalAlignAddrLValue(ValueAndIsReference.getPointer(),
      //                                       refExpr->getType());
      llvm_unreachable("NYI");
    }

    mlir::TypedAttr getValue() const {
      assert(!isReference());
      return mlir::cast<mlir::TypedAttr>(ValueAndIsReference.getPointer());
    }
  };

  ConstantEmission tryEmitAsConstant(DeclRefExpr *refExpr);
  ConstantEmission tryEmitAsConstant(const MemberExpr *ME);

  /// Emit the computation of the specified expression of scalar type,
  /// ignoring the result.
  mlir::Value emitScalarExpr(const clang::Expr *E);
  mlir::Value emitScalarConstant(const ConstantEmission &Constant, Expr *E);

  mlir::Value emitPromotedComplexExpr(const Expr *E, QualType PromotionType);
  mlir::Value emitPromotedScalarExpr(const clang::Expr *E,
                                     QualType PromotionType);
  mlir::Value emitPromotedValue(mlir::Value result, QualType PromotionType);
  mlir::Value emitUnPromotedValue(mlir::Value result, QualType PromotionType);

  const CaseStmt *foldCaseStmt(const clang::CaseStmt &S, mlir::Type condType,
                               mlir::ArrayAttr &value, cir::CaseOpKind &kind);

  template <typename T>
  mlir::LogicalResult emitCaseDefaultCascade(const T *stmt, mlir::Type condType,
                                             mlir::ArrayAttr value,
                                             cir::CaseOpKind kind,
                                             bool buildingTopLevelCase);

  mlir::LogicalResult emitCaseStmt(const clang::CaseStmt &S,
                                   mlir::Type condType,
                                   bool buildingTopLevelCase);

  mlir::LogicalResult emitDefaultStmt(const clang::DefaultStmt &S,
                                      mlir::Type condType,
                                      bool buildingTopLevelCase);

  mlir::LogicalResult emitSwitchCase(const clang::SwitchCase &S,
                                     bool buildingTopLevelCase);

  mlir::LogicalResult emitSwitchBody(const clang::Stmt *S);

  cir::FuncOp generateCode(clang::GlobalDecl GD, cir::FuncOp Fn,
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

      llvm_unreachable("NYI");
    }
  };

  LValue emitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E);

  /// Emit the alloca and debug information for a
  /// local variable.  Does not emit initialization or destruction.
  AutoVarEmission emitAutoVarAlloca(const clang::VarDecl &D,
                                    mlir::OpBuilder::InsertPoint = {});

  void emitAutoVarInit(const AutoVarEmission &emission);
  void emitAutoVarCleanups(const AutoVarEmission &emission);
  void emitAutoVarTypeCleanup(const AutoVarEmission &emission,
                              clang::QualType::DestructionKind dtorKind);

  void emitStoreOfScalar(mlir::Value value, LValue lvalue);
  void emitStoreOfScalar(mlir::Value value, Address addr, bool isVolatile,
                         clang::QualType ty, LValueBaseInfo baseInfo,
                         TBAAAccessInfo tbaaInfo, bool isInit = false,
                         bool isNontemporal = false);
  void emitStoreOfScalar(mlir::Value value, Address addr, bool isVolatile,
                         QualType ty,
                         AlignmentSource source = AlignmentSource::Type,
                         bool isInit = false, bool isNontemporal = false) {
    emitStoreOfScalar(value, addr, isVolatile, ty, LValueBaseInfo(source),
                      CGM.getTBAAAccessInfo(ty), isInit, isNontemporal);
  }
  void emitStoreOfScalar(mlir::Value value, LValue lvalue, bool isInit);

  /// Given a value and its clang type, returns the value casted to its memory
  /// representation.
  /// Note: CIR defers most of the special casting to the final lowering passes
  /// to conserve the high level information.
  mlir::Value emitToMemory(mlir::Value Value, clang::QualType Ty);

  void emitDeclRefExprDbgValue(const DeclRefExpr *E, const APValue &Init);

  /// Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void emitStoreThroughLValue(RValue Src, LValue Dst, bool isInit = false);

  void emitStoreThroughBitfieldLValue(RValue Src, LValue Dst,
                                      mlir::Value &Result);

  /// The given basic block lies in the current EH scope, but may be a
  /// target of a potentially scope-crossing jump; get a stable handle
  /// to which we can perform this jump later.
  /// CIRGen: this mostly tracks state for figuring out the proper scope
  /// information, no actual branches are emitted.
  JumpDest getJumpDestInCurrentScope(mlir::Block *target) {
    return JumpDest(target, EHStack.getInnermostNormalCleanup(),
                    nextCleanupDestIndex++);
  }

  cir::BrOp emitBranchThroughCleanup(mlir::Location Loc, JumpDest Dest);

  /// Given an assignment `*LHS = RHS`, emit a test that checks if \p RHS is
  /// nonnull, if 1\p LHS is marked _Nonnull.
  void emitNullabilityCheck(LValue LHS, mlir::Value RHS,
                            clang::SourceLocation Loc);

  /// Same as IRBuilder::CreateInBoundsGEP, but additionally emits a check to
  /// detect undefined behavior when the pointer overflow sanitizer is enabled.
  /// \p SignedIndices indicates whether any of the GEP indices are signed.
  /// \p IsSubtraction indicates whether the expression used to form the GEP
  /// is a subtraction.
  mlir::Value emitCheckedInBoundsGEP(mlir::Type ElemTy, mlir::Value Ptr,
                                     llvm::ArrayRef<mlir::Value> IdxList,
                                     bool SignedIndices, bool IsSubtraction,
                                     SourceLocation Loc);

  void emitScalarInit(const clang::Expr *init, mlir::Location loc,
                      LValue lvalue, bool capturedByInit = false);

  LValue emitDeclRefLValue(const clang::DeclRefExpr *E);
  LValue emitExtVectorElementExpr(const ExtVectorElementExpr *E);
  LValue emitBinaryOperatorLValue(const clang::BinaryOperator *E);
  LValue emitCompoundAssignmentLValue(const clang::CompoundAssignOperator *E);
  LValue emitUnaryOpLValue(const clang::UnaryOperator *E);
  LValue emitStringLiteralLValue(const StringLiteral *E);
  RValue emitBuiltinExpr(const clang::GlobalDecl GD, unsigned BuiltinID,
                         const clang::CallExpr *E, ReturnValueSlot ReturnValue);
  RValue emitRotate(const CallExpr *E, bool IsRotateRight);
  template <uint32_t N>
  RValue emitBuiltinWithOneOverloadedType(const CallExpr *E,
                                          llvm::StringRef Name) {
    static_assert(N, "expect non-empty argument");
    mlir::Type cirTy = convertType(E->getArg(0)->getType());
    SmallVector<mlir::Value, N> args;
    for (uint32_t i = 0; i < N; ++i) {
      args.push_back(emitScalarExpr(E->getArg(i)));
    }
    const auto call = builder.create<cir::LLVMIntrinsicCallOp>(
        getLoc(E->getExprLoc()), builder.getStringAttr(Name), cirTy, args);
    return RValue::get(call->getResult(0));
  }
  mlir::Value emitTargetBuiltinExpr(unsigned BuiltinID,
                                    const clang::CallExpr *E,
                                    ReturnValueSlot ReturnValue);

  // Target specific builtin emission
  mlir::Value emitScalarOrConstFoldImmArg(unsigned ICEArguments, unsigned Idx,
                                          const CallExpr *E);
  mlir::Value emitAArch64BuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                     ReturnValueSlot ReturnValue,
                                     llvm::Triple::ArchType Arch);
  mlir::Value emitAArch64SVEBuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  mlir::Value emitAArch64SMEBuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  mlir::Value emitX86BuiltinExpr(unsigned BuiltinID, const CallExpr *E);

  /// Given an expression with a pointer type, emit the value and compute our
  /// best estimate of the alignment of the pointee.
  ///
  /// \param BaseInfo - If non-null, this will be initialized with
  /// information about the source of the alignment and the may-alias
  /// attribute.  Note that this function will conservatively fall back on
  /// the type when it doesn't recognize the expression and may-alias will
  /// be set to false.
  ///
  /// One reasonable way to use this information is when there's a language
  /// guarantee that the pointer must be aligned to some stricter value, and
  /// we're simply trying to ensure that sufficiently obvious uses of under-
  /// aligned objects don't get miscompiled; for example, a placement new
  /// into the address of a local variable.  In such a case, it's quite
  /// reasonable to just ignore the returned alignment when it isn't from an
  /// explicit source.
  Address
  emitPointerWithAlignment(const clang::Expr *expr,
                           LValueBaseInfo *baseInfo = nullptr,
                           TBAAAccessInfo *tbaaInfo = nullptr,
                           KnownNonNull_t isKnownNonNull = NotKnownNonNull);

  LValue emitConditionalOperatorLValue(const AbstractConditionalOperator *expr);

  /// Emit an expression as an initializer for an object (variable, field, etc.)
  /// at the given location.  The expression is not necessarily the normal
  /// initializer for the object, and the address is not necessarily
  /// its normal location.
  ///
  /// \param init the initializing expression
  /// \param D the object to act as if we're initializing
  /// \param lvalue the lvalue to initialize
  /// \param capturedByInit true if \p D is a __block variable whose address is
  /// potentially changed by the initializer
  void emitExprAsInit(const clang::Expr *init, const clang::ValueDecl *D,
                      LValue lvalue, bool capturedByInit = false);

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void emitAutoVarDecl(const clang::VarDecl &D);

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void emitVarDecl(const clang::VarDecl &D);

  cir::GlobalOp addInitializerToStaticVarDecl(const VarDecl &D,
                                              cir::GlobalOp GV,
                                              cir::GetGlobalOp GVAddr);

  void emitStaticVarDecl(const VarDecl &D, cir::GlobalLinkageKind Linkage);

  /// Perform the usual unary conversions on the specified
  /// expression and compare the result against zero, returning an Int1Ty value.
  mlir::Value evaluateExprAsBool(const clang::Expr *E);

  void emitCtorPrologue(const clang::CXXConstructorDecl *CD,
                        clang::CXXCtorType Type, FunctionArgList &Args);
  void emitConstructorBody(FunctionArgList &Args);
  void emitDestructorBody(FunctionArgList &Args);
  void emitCXXDestructorCall(const CXXDestructorDecl *D, CXXDtorType Type,
                             bool ForVirtualBase, bool Delegating, Address This,
                             QualType ThisTy);
  RValue emitCXXDestructorCall(GlobalDecl Dtor, const CIRGenCallee &Callee,
                               mlir::Value This, QualType ThisTy,
                               mlir::Value ImplicitParam,
                               QualType ImplicitParamTy, const CallExpr *E);

  /// Enter the cleanups necessary to complete the given phase of destruction
  /// for a destructor. The end result should call destructors on members and
  /// base classes in reverse order of their construction.
  void EnterDtorCleanups(const CXXDestructorDecl *Dtor, CXXDtorType Type);

  /// Determines whether an EH cleanup is required to destroy a type
  /// with the given destruction kind.
  /// TODO(cir): could be shared with Clang LLVM codegen
  bool needsEHCleanup(QualType::DestructionKind kind) {
    switch (kind) {
    case QualType::DK_none:
      return false;
    case QualType::DK_cxx_destructor:
    case QualType::DK_objc_weak_lifetime:
    case QualType::DK_nontrivial_c_struct:
      return getLangOpts().Exceptions;
    case QualType::DK_objc_strong_lifetime:
      return getLangOpts().Exceptions &&
             CGM.getCodeGenOpts().ObjCAutoRefCountExceptions;
    }
    llvm_unreachable("bad destruction kind");
  }

  CleanupKind getCleanupKind(QualType::DestructionKind kind) {
    return (needsEHCleanup(kind) ? NormalAndEHCleanup : NormalCleanup);
  }

  void pushEHDestroy(QualType::DestructionKind dtorKind, Address addr,
                     QualType type);

  void pushStackRestore(CleanupKind kind, Address SPMem);

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
  /// Return the Value of the vtable pointer member pointed to by This.
  mlir::Value getVTablePtr(mlir::Location Loc, Address This,
                           mlir::Type VTableTy,
                           const CXXRecordDecl *VTableClass);

  /// Returns whether we should perform a type checked load when loading a
  /// virtual function for virtual calls to members of RD. This is generally
  /// true when both vcall CFI and whole-program-vtables are enabled.
  bool shouldEmitVTableTypeCheckedLoad(const CXXRecordDecl *RD);

  /// If whole-program virtual table optimization is enabled, emit an assumption
  /// that VTable is a member of RD's type identifier. Or, if vptr CFI is
  /// enabled, emit a check that VTable is a member of RD's type identifier.
  void emitTypeMetadataCodeForVCall(const CXXRecordDecl *RD, mlir::Value VTable,
                                    SourceLocation Loc);

  /// Return the VTT parameter that should be passed to a base
  /// constructor/destructor with virtual bases.
  /// FIXME: VTTs are Itanium ABI-specific, so the definition should move
  /// to CIRGenItaniumCXXABI.cpp together with all the references to VTT.
  mlir::Value GetVTTParameter(GlobalDecl GD, bool ForVirtualBase,
                              bool Delegating);

  /// Source location information about the default argument or member
  /// initializer expression we're evaluating, if any.
  clang::CurrentSourceLocExprScope CurSourceLocExprScope;
  using SourceLocExprScopeGuard =
      clang::CurrentSourceLocExprScope::SourceLocExprScopeGuard;

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

  /// The scope of a CXXDefaultInitExpr. Within this scope, the value of 'this'
  /// is overridden to be the object under construction.
  class CXXDefaultInitExprScope {
  public:
    CXXDefaultInitExprScope(CIRGenFunction &CGF,
                            const clang::CXXDefaultInitExpr *E)
        : CGF{CGF}, OldCXXThisValue(CGF.CXXThisValue),
          OldCXXThisAlignment(CGF.CXXThisAlignment),
          SourceLocScope(E, CGF.CurSourceLocExprScope) {
      CGF.CXXThisValue = CGF.CXXDefaultInitExprThis.getPointer();
      CGF.CXXThisAlignment = CGF.CXXDefaultInitExprThis.getAlignment();
    }
    ~CXXDefaultInitExprScope() {
      CGF.CXXThisValue = OldCXXThisValue;
      CGF.CXXThisAlignment = OldCXXThisAlignment;
    }

  public:
    CIRGenFunction &CGF;
    mlir::Value OldCXXThisValue;
    clang::CharUnits OldCXXThisAlignment;
    SourceLocExprScopeGuard SourceLocScope;
  };

  struct CXXDefaultArgExprScope : SourceLocExprScopeGuard {
    CXXDefaultArgExprScope(CIRGenFunction &CGF, const CXXDefaultArgExpr *E)
        : SourceLocExprScopeGuard(E, CGF.CurSourceLocExprScope) {}
  };

  LValue MakeNaturalAlignPointeeAddrLValue(mlir::Value V, clang::QualType T);
  LValue MakeNaturalAlignAddrLValue(mlir::Value val, QualType ty);

  /// Construct an address with the natural alignment of T. If a pointer to T
  /// is expected to be signed, the pointer passed to this function must have
  /// been signed, and the returned Address will have the pointer authentication
  /// information needed to authenticate the signed pointer.
  Address makeNaturalAddressForPointer(
      mlir::Value ptr, QualType t, CharUnits alignment = CharUnits::Zero(),
      bool forPointeeType = false, LValueBaseInfo *baseInfo = nullptr,
      TBAAAccessInfo *tbaaInfo = nullptr,
      KnownNonNull_t isKnownNonNull = NotKnownNonNull) {
    if (alignment.isZero())
      alignment =
          CGM.getNaturalTypeAlignment(t, baseInfo, tbaaInfo, forPointeeType);
    return Address(ptr, convertTypeForMem(t), alignment, isKnownNonNull);
  }

  /// Load the value for 'this'. This function is only valid while generating
  /// code for an C++ member function.
  /// FIXME(cir): this should return a mlir::Value!
  mlir::Value LoadCXXThis() {
    assert(CXXThisValue && "no 'this' value for this function");
    return CXXThisValue;
  }
  Address LoadCXXThisAddress();

  /// Load the VTT parameter to base constructors/destructors have virtual
  /// bases. FIXME: Every place that calls LoadCXXVTT is something that needs to
  /// be abstracted properly.
  mlir::Value LoadCXXVTT() {
    assert(CXXStructorImplicitParamValue && "no VTT value for this function");
    return CXXStructorImplicitParamValue;
  }

  /// Convert the given pointer to a complete class to the given direct base.
  Address getAddressOfDirectBaseInCompleteClass(mlir::Location loc,
                                                Address Value,
                                                const CXXRecordDecl *Derived,
                                                const CXXRecordDecl *Base,
                                                bool BaseIsVirtual);

  Address getAddressOfBaseClass(Address Value, const CXXRecordDecl *Derived,
                                CastExpr::path_const_iterator PathBegin,
                                CastExpr::path_const_iterator PathEnd,
                                bool NullCheckValue, SourceLocation Loc);

  Address getAddressOfDerivedClass(Address baseAddr,
                                   const CXXRecordDecl *derived,
                                   CastExpr::path_const_iterator pathBegin,
                                   CastExpr::path_const_iterator pathEnd,
                                   bool nullCheckValue);

  /// Emit code for the start of a function.
  /// \param Loc       The location to be associated with the function.
  /// \param StartLoc  The location of the function body.
  void StartFunction(clang::GlobalDecl GD, clang::QualType RetTy,
                     cir::FuncOp Fn, const CIRGenFunctionInfo &FnInfo,
                     const FunctionArgList &Args, clang::SourceLocation Loc,
                     clang::SourceLocation StartLoc);

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  mlir::Value emitScalarConversion(mlir::Value Src, clang::QualType SrcTy,
                                   clang::QualType DstTy,
                                   clang::SourceLocation Loc);

  /// Emit a conversion from the specified complex type to the specified
  /// destination type, where the destination type is an LLVM scalar type.
  mlir::Value emitComplexToScalarConversion(mlir::Value Src, QualType SrcTy,
                                            QualType DstTy, SourceLocation Loc);

  LValue makeAddrLValue(Address addr, clang::QualType ty,
                        LValueBaseInfo baseInfo, TBAAAccessInfo tbaaInfo) {
    return LValue::makeAddr(addr, ty, getContext(), baseInfo, tbaaInfo);
  }

  LValue makeAddrLValue(Address addr, clang::QualType ty,
                        AlignmentSource source = AlignmentSource::Type) {
    return LValue::makeAddr(addr, ty, getContext(), LValueBaseInfo(source),
                            CGM.getTBAAAccessInfo(ty));
  }

  void initializeVTablePointers(mlir::Location loc,
                                const clang::CXXRecordDecl *RD);
  void initializeVTablePointer(mlir::Location loc, const VPtr &Vptr);

  AggValueSlot::Overlap_t getOverlapForFieldInit(const FieldDecl *FD);
  LValue emitLValueForField(LValue base, const clang::FieldDecl *field);
  LValue emitLValueForBitField(LValue base, const FieldDecl *field);
  LValue emitLValueForLambdaField(const FieldDecl *field);
  LValue emitLValueForLambdaField(const FieldDecl *field,
                                  mlir::Value thisValue);

  /// Like emitLValueForField, excpet that if the Field is a reference, this
  /// will return the address of the reference and not the address of the value
  /// stored in the reference.
  LValue emitLValueForFieldInitialization(LValue Base,
                                          const clang::FieldDecl *Field,
                                          llvm::StringRef FieldName);

  void emitInitializerForField(clang::FieldDecl *Field, LValue LHS,
                               clang::Expr *Init);

  /// Determine whether the given initializer is trivial in the sense
  /// that it requires no code to be generated.
  bool isTrivialInitializer(const clang::Expr *Init);

  // TODO: this can also be abstrated into common AST helpers
  bool hasBooleanRepresentation(clang::QualType Ty);

  void emitCXXThrowExpr(const CXXThrowExpr *E);

  RValue emitAtomicExpr(AtomicExpr *E);
  void emitAtomicStore(RValue rvalue, LValue lvalue, bool isInit);
  void emitAtomicStore(RValue rvalue, LValue lvalue, cir::MemOrder MO,
                       bool IsVolatile, bool isInit);
  void emitAtomicInit(Expr *init, LValue dest);

  /// Return the address of a local variable.
  Address GetAddrOfLocalVar(const clang::VarDecl *VD) {
    auto it = LocalDeclMap.find(VD);
    assert(it != LocalDeclMap.end() &&
           "Invalid argument to GetAddrOfLocalVar(), no decl!");
    return it->second;
  }

  Address getAddrOfBitFieldStorage(LValue base, const clang::FieldDecl *field,
                                   mlir::Type fieldType, unsigned index);

  /// Given an opaque value expression, return its LValue mapping if it exists,
  /// otherwise create one.
  LValue getOrCreateOpaqueLValueMapping(const OpaqueValueExpr *e);

  /// Given an opaque value expression, return its RValue mapping if it exists,
  /// otherwise create one.
  RValue getOrCreateOpaqueRValueMapping(const OpaqueValueExpr *e);

  /// Check if \p E is a C++ "this" pointer wrapped in value-preserving casts.
  static bool isWrappedCXXThis(const clang::Expr *E);

  void emitDelegateCXXConstructorCall(const clang::CXXConstructorDecl *Ctor,
                                      clang::CXXCtorType CtorType,
                                      const FunctionArgList &Args,
                                      clang::SourceLocation Loc);

  // It's important not to confuse this and the previous function. Delegating
  // constructors are the C++11 feature. The constructor delegate optimization
  // is used to reduce duplication in the base and complete constructors where
  // they are substantially the same.
  void emitDelegatingCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                        const FunctionArgList &Args);

  /// We are performing a delegate call; that is, the current function is
  /// delegating to another one. Produce a r-value suitable for passing the
  /// given parameter.
  void emitDelegateCallArg(CallArgList &args, const clang::VarDecl *param,
                           clang::SourceLocation loc);

  /// Return true if the current function should not be instrumented with
  /// sanitizers.
  bool ShouldSkipSanitizerInstrumentation();
  bool ShouldXRayInstrumentFunction() const;

  /// Return true if the current function should be instrumented with
  /// __cyg_profile_func_* calls
  bool ShouldInstrumentFunction();

  /// TODO(cir): add TBAAAccessInfo
  Address emitArrayToPointerDecay(const Expr *Array,
                                  LValueBaseInfo *BaseInfo = nullptr,
                                  TBAAAccessInfo *TBAAInfo = nullptr);

  /// Emits the code necessary to evaluate an arbitrary expression into the
  /// given memory location.
  void emitAnyExprToMem(const Expr *E, Address Location, Qualifiers Quals,
                        bool IsInitializer);
  void emitAnyExprToExn(const Expr *E, Address Addr);

  LValue emitCheckedLValue(const Expr *E, TypeCheckKind TCK);
  LValue emitMemberExpr(const MemberExpr *E);
  LValue emitCompoundLiteralLValue(const CompoundLiteralExpr *E);

  /// Specifies which type of sanitizer check to apply when handling a
  /// particular builtin.
  enum BuiltinCheckKind {
    BCK_CTZPassedZero,
    BCK_CLZPassedZero,
  };

  /// Emits an argument for a call to a builtin. If the builtin sanitizer is
  /// enabled, a runtime check specified by \p Kind is also emitted.
  mlir::Value emitCheckedArgForBuiltin(const Expr *E, BuiltinCheckKind Kind);

  /// Emits an argument for a call to a `__builtin_assume`. If the builtin
  /// sanitizer is enabled, a runtime check is also emitted.
  mlir::Value emitCheckedArgForAssume(const Expr *E);

  /// returns true if aggregate type has a volatile member.
  /// TODO(cir): this could be a common AST helper between LLVM / CIR.
  bool hasVolatileMember(QualType T) {
    if (const RecordType *RT = T->getAs<RecordType>()) {
      const RecordDecl *RD = mlir::cast<RecordDecl>(RT->getDecl());
      return RD->hasVolatileMember();
    }
    return false;
  }

  /// Emit an aggregate assignment.
  void emitAggregateAssign(LValue Dest, LValue Src, QualType EltTy) {
    bool IsVolatile = hasVolatileMember(EltTy);
    emitAggregateCopy(Dest, Src, EltTy, AggValueSlot::MayOverlap, IsVolatile);
  }

  LValue emitAggExprToLValue(const Expr *E);

  /// Emit an aggregate copy.
  ///
  /// \param isVolatile \c true iff either the source or the destination is
  ///        volatile.
  /// \param MayOverlap Whether the tail padding of the destination might be
  ///        occupied by some other object. More efficient code can often be
  ///        generated if not.
  void emitAggregateCopy(LValue Dest, LValue Src, QualType EltTy,
                         AggValueSlot::Overlap_t MayOverlap,
                         bool isVolatile = false);

  /// Emit a reached-unreachable diagnostic if \p Loc is valid and runtime
  /// checking is enabled. Otherwise, just emit an unreachable instruction.
  void emitUnreachable(SourceLocation Loc);

  ///
  /// Cleanups
  /// --------

  /// Header for data within LifetimeExtendedCleanupStack.
  struct LifetimeExtendedCleanupHeader {
    /// The size of the following cleanup object.
    unsigned Size;
    /// The kind of cleanup to push: a value from the CleanupKind enumeration.
    unsigned Kind : 31;
    /// Whether this is a conditional cleanup.
    unsigned IsConditional : 1;

    size_t getSize() const { return Size; }
    CleanupKind getKind() const { return (CleanupKind)Kind; }
    bool isConditional() const { return IsConditional; }
  };

  /// Emits try/catch information for the current EH stack.
  cir::CallOp callWithExceptionCtx = nullptr;
  mlir::Operation *emitLandingPad(cir::TryOp tryOp);
  void emitEHResumeBlock(bool isCleanup, mlir::Block *ehResumeBlock,
                         mlir::Location loc);
  mlir::Block *getEHResumeBlock(bool isCleanup, cir::TryOp tryOp);
  mlir::Block *getEHDispatchBlock(EHScopeStack::stable_iterator scope,
                                  cir::TryOp tryOp);
  /// Unified block containing a call to cir.resume
  mlir::Block *ehResumeBlock = nullptr;
  llvm::DenseMap<mlir::Block *, mlir::Block *> cleanupsToPatch;

  /// The cleanup depth enclosing all the cleanups associated with the
  /// parameters.
  EHScopeStack::stable_iterator PrologueCleanupDepth;

  mlir::Operation *getInvokeDestImpl(cir::TryOp tryOp);
  mlir::Operation *getInvokeDest(cir::TryOp tryOp) {
    if (!EHStack.requiresLandingPad())
      return nullptr;
    // Return the respective cir.try, this can be used to compute
    // any other relevant information.
    return getInvokeDestImpl(tryOp);
  }
  bool isInvokeDest();

  /// Takes the old cleanup stack size and emits the cleanup blocks
  /// that have been added.
  void
  PopCleanupBlocks(EHScopeStack::stable_iterator OldCleanupStackSize,
                   std::initializer_list<mlir::Value *> ValuesToReload = {});

  /// Takes the old cleanup stack size and emits the cleanup blocks
  /// that have been added, then adds all lifetime-extended cleanups from
  /// the given position to the stack.
  void
  PopCleanupBlocks(EHScopeStack::stable_iterator OldCleanupStackSize,
                   size_t OldLifetimeExtendedStackSize,
                   std::initializer_list<mlir::Value *> ValuesToReload = {});

  /// Will pop the cleanup entry on the stack and process all branch fixups.
  void PopCleanupBlock(bool FallThroughIsBranchThrough = false);

  /// Deactivates the given cleanup block. The block cannot be reactivated. Pops
  /// it if it's the top of the stack.
  ///
  /// \param DominatingIP - An instruction which is known to
  ///   dominate the current IP (if set) and which lies along
  ///   all paths of execution between the current IP and the
  ///   the point at which the cleanup comes into scope.
  void DeactivateCleanupBlock(EHScopeStack::stable_iterator Cleanup,
                              mlir::Operation *DominatingIP);

  typedef void Destroyer(CIRGenFunction &CGF, Address addr, QualType ty);

  static Destroyer destroyCXXObject;

  void pushDestroy(QualType::DestructionKind dtorKind, Address addr,
                   QualType type);

  void pushDestroy(CleanupKind kind, Address addr, QualType type,
                   Destroyer *destroyer, bool useEHCleanupForArray);

  Destroyer *getDestroyer(QualType::DestructionKind kind);

  void emitDestroy(Address addr, QualType type, Destroyer *destroyer,
                   bool useEHCleanupForArray);

  /// An object to manage conditionally-evaluated expressions.
  class ConditionalEvaluation {
    mlir::OpBuilder::InsertPoint insertPt;

  public:
    ConditionalEvaluation(CIRGenFunction &CGF)
        : insertPt(CGF.builder.saveInsertionPoint()) {}
    ConditionalEvaluation(mlir::OpBuilder::InsertPoint ip) : insertPt(ip) {}

    void begin(CIRGenFunction &CGF) {
      assert(CGF.OutermostConditional != this);
      if (!CGF.OutermostConditional)
        CGF.OutermostConditional = this;
    }

    void end(CIRGenFunction &CGF) {
      assert(CGF.OutermostConditional != nullptr);
      if (CGF.OutermostConditional == this)
        CGF.OutermostConditional = nullptr;
    }

    /// Returns the insertion point which will be executed prior to each
    /// evaluation of the conditional code. In LLVM OG, this method
    /// is called getStartingBlock.
    mlir::OpBuilder::InsertPoint getInsertPoint() const { return insertPt; }
  };

  struct ConditionalInfo {
    std::optional<LValue> LHS{}, RHS{};
    mlir::Value Result{};
  };

  template <typename FuncTy>
  ConditionalInfo emitConditionalBlocks(const AbstractConditionalOperator *E,
                                        const FuncTy &BranchGenFunc);

  // Return true if we're currently emitting one branch or the other of a
  // conditional expression.
  bool isInConditionalBranch() const { return OutermostConditional != nullptr; }

  void setBeforeOutermostConditional(mlir::Value value, Address addr) {
    assert(isInConditionalBranch());
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.restoreInsertionPoint(OutermostConditional->getInsertPoint());
      builder.createStore(
          value.getLoc(), value, addr,
          /*volatile*/ false,
          mlir::IntegerAttr::get(
              mlir::IntegerType::get(value.getContext(), 64),
              (uint64_t)addr.getAlignment().getAsAlign().value()));
    }
  }

  void pushIrregularPartialArrayCleanup(mlir::Value arrayBegin,
                                        Address arrayEndPointer,
                                        QualType elementType,
                                        CharUnits elementAlign,
                                        Destroyer *destroyer);
  void pushRegularPartialArrayCleanup(mlir::Value arrayBegin,
                                      mlir::Value arrayEnd,
                                      QualType elementType,
                                      CharUnits elementAlign,
                                      Destroyer *destroyer);
  void pushDestroyAndDeferDeactivation(QualType::DestructionKind dtorKind,
                                       Address addr, QualType type);
  void pushDestroyAndDeferDeactivation(CleanupKind cleanupKind, Address addr,
                                       QualType type, Destroyer *destroyer,
                                       bool useEHCleanupForArray);
  void emitArrayDestroy(mlir::Value begin, mlir::Value end,
                        QualType elementType, CharUnits elementAlign,
                        Destroyer *destroyer, bool checkZeroLength,
                        bool useEHCleanup);

  /// The values of function arguments to use when evaluating
  /// CXXInheritedCtorInitExprs within this context.
  CallArgList CXXInheritedCtorInitExprArgs;

  // Points to the outermost active conditional control. This is used so that
  // we know if a temporary should be destroyed conditionally.
  ConditionalEvaluation *OutermostConditional = nullptr;

  template <class T>
  typename DominatingValue<T>::saved_type saveValueInCond(T value) {
    return DominatingValue<T>::save(*this, value);
  }

  /// Push a cleanup to be run at the end of the current full-expression.  Safe
  /// against the possibility that we're currently inside a
  /// conditionally-evaluated expression.
  template <class T, class... As>
  void pushFullExprCleanup(CleanupKind kind, As... A) {
    // If we're not in a conditional branch, or if none of the
    // arguments requires saving, then use the unconditional cleanup.
    if (!isInConditionalBranch())
      return EHStack.pushCleanup<T>(kind, A...);

    // Stash values in a tuple so we can guarantee the order of saves.
    typedef std::tuple<typename DominatingValue<As>::saved_type...> SavedTuple;
    SavedTuple Saved{saveValueInCond(A)...};

    typedef EHScopeStack::ConditionalCleanup<T, As...> CleanupType;
    EHStack.pushCleanupTuple<CleanupType>(kind, Saved);
    initFullExprCleanup();
  }

  /// Set up the last cleanup that was pushed as a conditional
  /// full-expression cleanup.
  void initFullExprCleanup() {
    initFullExprCleanupWithFlag(createCleanupActiveFlag());
  }

  void initFullExprCleanupWithFlag(Address ActiveFlag);
  Address createCleanupActiveFlag();

  /// Enters a new scope for capturing cleanups, all of which
  /// will be executed once the scope is exited.
  class RunCleanupsScope {
    EHScopeStack::stable_iterator CleanupStackDepth, OldCleanupScopeDepth;
    size_t LifetimeExtendedCleanupStackSize;
    bool OldDidCallStackSave;

  protected:
    bool PerformCleanup;

  private:
    RunCleanupsScope(const RunCleanupsScope &) = delete;
    void operator=(const RunCleanupsScope &) = delete;

  protected:
    CIRGenFunction &CGF;

  public:
    /// Enter a new cleanup scope.
    explicit RunCleanupsScope(CIRGenFunction &CGF)
        : PerformCleanup(true), CGF(CGF) {
      CleanupStackDepth = CGF.EHStack.stable_begin();
      LifetimeExtendedCleanupStackSize =
          CGF.LifetimeExtendedCleanupStack.size();
      OldDidCallStackSave = CGF.DidCallStackSave;
      CGF.DidCallStackSave = false;
      OldCleanupScopeDepth = CGF.CurrentCleanupScopeDepth;
      CGF.CurrentCleanupScopeDepth = CleanupStackDepth;
    }

    /// Exit this cleanup scope, emitting any accumulated cleanups.
    ~RunCleanupsScope() {
      if (PerformCleanup)
        ForceCleanup();
    }

    /// Determine whether this scope requires any cleanups.
    bool requiresCleanups() const {
      return CGF.EHStack.stable_begin() != CleanupStackDepth;
    }

    /// Force the emission of cleanups now, instead of waiting
    /// until this object is destroyed.
    /// \param ValuesToReload - A list of values that need to be available at
    /// the insertion point after cleanup emission. If cleanup emission created
    /// a shared cleanup block, these value pointers will be rewritten.
    /// Otherwise, they not will be modified.
    void
    ForceCleanup(std::initializer_list<mlir::Value *> ValuesToReload = {}) {
      assert(PerformCleanup && "Already forced cleanup");
      {
        mlir::OpBuilder::InsertionGuard guard(CGF.getBuilder());
        CGF.DidCallStackSave = OldDidCallStackSave;
        CGF.PopCleanupBlocks(CleanupStackDepth,
                             LifetimeExtendedCleanupStackSize, ValuesToReload);
        PerformCleanup = false;
        CGF.CurrentCleanupScopeDepth = OldCleanupScopeDepth;
      }
    }
  };

  // Cleanup stack depth of the RunCleanupsScope that was pushed most recently.
  EHScopeStack::stable_iterator CurrentCleanupScopeDepth =
      EHScopeStack::stable_end();

  /// -------
  /// Lexical Scope: to be read as in the meaning in CIR, a scope is always
  /// related with initialization and destruction of objects.
  /// -------

public:
  // Represents a cir.scope, cir.if, and then/else regions. I.e. lexical
  // scopes that require cleanups.
  struct LexicalScope : public RunCleanupsScope {
  private:
    // Block containing cleanup code for things initialized in this
    // lexical context (scope).
    mlir::Block *CleanupBlock = nullptr;

    // Points to scope entry block. This is useful, for instance, for
    // helping to insert allocas before finalizing any recursive codegen
    // from switches.
    mlir::Block *EntryBlock;

    // On a coroutine body, the OnFallthrough sub stmt holds the handler
    // (CoreturnStmt) for control flow falling off the body. Keep track
    // of emitted co_return in this scope and allow OnFallthrough to be
    // skipeed.
    bool HasCoreturn = false;

    LexicalScope *ParentScope = nullptr;

    // Holds actual value for ScopeKind::Try
    cir::TryOp tryOp = nullptr;

    // FIXME: perhaps we can use some info encoded in operations.
    enum Kind {
      Regular,   // cir.if, cir.scope, if_regions
      Ternary,   // cir.ternary
      Switch,    // cir.switch
      Try,       // cir.try
      GlobalInit // cir.global initalization code
    } ScopeKind = Regular;

    // Track scope return value.
    mlir::Value retVal = nullptr;

  public:
    unsigned Depth = 0;
    bool HasReturn = false;

    LexicalScope(CIRGenFunction &CGF, mlir::Location loc, mlir::Block *eb)
        : RunCleanupsScope(CGF), EntryBlock(eb), ParentScope(CGF.currLexScope),
          BeginLoc(loc), EndLoc(loc) {

      CGF.currLexScope = this;
      if (ParentScope)
        Depth++;

      // Has multiple locations: overwrite with separate start and end locs.
      if (const auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
        assert(fusedLoc.getLocations().size() == 2 && "too many locations");
        BeginLoc = fusedLoc.getLocations()[0];
        EndLoc = fusedLoc.getLocations()[1];
      }

      assert(EntryBlock && "expected valid block");
    }

    void setRetVal(mlir::Value v) { retVal = v; }

    void cleanup();
    void restore() { CGF.currLexScope = ParentScope; }

    ~LexicalScope() {
      // EmitLexicalBlockEnd
      assert(!cir::MissingFeatures::generateDebugInfo());
      // If we should perform a cleanup, force them now.  Note that
      // this ends the cleanup scope before rescoping any labels.
      cleanup();
      restore();
    }

    /// Force the emission of cleanups now, instead of waiting
    /// until this object is destroyed.
    void ForceCleanup() {
      RunCleanupsScope::ForceCleanup();
      // TODO(cir): something akin to rescopeLabels if it makes sense to CIR.
    }

    // ---
    // Coroutine tracking
    // ---
    bool hasCoreturn() const { return HasCoreturn; }
    void setCoreturn() { HasCoreturn = true; }

    // ---
    // Kind
    // ---
    bool isGlobalInit() { return ScopeKind == Kind::GlobalInit; }
    bool isRegular() { return ScopeKind == Kind::Regular; }
    bool isSwitch() { return ScopeKind == Kind::Switch; }
    bool isTernary() { return ScopeKind == Kind::Ternary; }
    bool isTry() { return ScopeKind == Kind::Try; }
    cir::TryOp getTry() {
      assert(isTry());
      return tryOp;
    }
    cir::TryOp getClosestTryParent();

    void setAsGlobalInit() { ScopeKind = Kind::GlobalInit; }
    void setAsSwitch() { ScopeKind = Kind::Switch; }
    void setAsTernary() { ScopeKind = Kind::Ternary; }
    void setAsTry(cir::TryOp op) {
      ScopeKind = Kind::Try;
      tryOp = op;
    }

    // ---
    // Goto handling
    // ---

    // Lazy create cleanup block or return what's available.
    mlir::Block *getOrCreateCleanupBlock(mlir::OpBuilder &builder) {
      if (CleanupBlock)
        return getCleanupBlock(builder);
      CleanupBlock = createCleanupBlock(builder);
      return CleanupBlock;
    }

    mlir::Block *getCleanupBlock(mlir::OpBuilder &builder) {
      return CleanupBlock;
    }
    mlir::Block *createCleanupBlock(mlir::OpBuilder &builder) {
      {
        // Create the cleanup block but dont hook it up around just yet.
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Region *r = builder.getBlock() ? builder.getBlock()->getParent()
                                             : &CGF.CurFn->getRegion(0);
        CleanupBlock = builder.createBlock(r);
      }
      return CleanupBlock;
    }

    // ---
    // Return handling
    // ---

  private:
    // On switches we need one return block per region, since cases don't
    // have their own scopes but are distinct regions nonetheless.
    llvm::SmallVector<mlir::Block *> RetBlocks;
    llvm::DenseMap<mlir::Block *, mlir::Location> RetLocs;
    llvm::DenseMap<cir::CaseOp, unsigned> RetBlockInCaseIndex;
    std::optional<unsigned> NormalRetBlockIndex;
    llvm::SmallVector<std::unique_ptr<mlir::Region>> SwitchRegions;

    // There's usually only one ret block per scope, but this needs to be
    // get or create because of potential unreachable return statements, note
    // that for those, all source location maps to the first one found.
    mlir::Block *createRetBlock(CIRGenFunction &CGF, mlir::Location loc) {
      assert((isa_and_nonnull<cir::CaseOp>(
                  CGF.builder.getBlock()->getParentOp()) ||
              RetBlocks.size() == 0) &&
             "only switches can hold more than one ret block");

      // Create the cleanup block but dont hook it up around just yet.
      mlir::OpBuilder::InsertionGuard guard(CGF.builder);
      auto *b = CGF.builder.createBlock(CGF.builder.getBlock()->getParent());
      RetBlocks.push_back(b);
      updateRetLoc(b, loc);
      return b;
    }

    cir::ReturnOp emitReturn(mlir::Location loc);
    void emitImplicitReturn();

  public:
    llvm::ArrayRef<mlir::Block *> getRetBlocks() { return RetBlocks; }
    mlir::Location getRetLoc(mlir::Block *b) { return RetLocs.at(b); }
    void updateRetLoc(mlir::Block *b, mlir::Location loc) {
      RetLocs.insert_or_assign(b, loc);
    }
    llvm::MutableArrayRef<std::unique_ptr<mlir::Region>> getSwitchRegions() {
      assert(isSwitch() && "expected switch scope");
      return SwitchRegions;
    }

    mlir::Region *createSwitchRegion() {
      assert(isSwitch() && "expected switch scope");
      SwitchRegions.push_back(std::make_unique<mlir::Region>());
      return SwitchRegions.back().get();
    }

    mlir::Block *getOrCreateRetBlock(CIRGenFunction &CGF, mlir::Location loc) {
      mlir::Block *ret = nullptr;
      if (auto caseOp = mlir::dyn_cast_if_present<cir::CaseOp>(
              CGF.builder.getBlock()->getParentOp())) {
        auto iter = RetBlockInCaseIndex.find(caseOp);
        if (iter != RetBlockInCaseIndex.end())
          ret = RetBlocks[iter->second];
        else {
          ret = createRetBlock(CGF, loc);
          RetBlockInCaseIndex[caseOp] = RetBlocks.size() - 1;
          return ret;
        }
      } else if (!NormalRetBlockIndex) {
        ret = createRetBlock(CGF, loc);
        NormalRetBlockIndex = RetBlocks.size() - 1;
        return ret;
      } else {
        ret = &*RetBlocks[*NormalRetBlockIndex];
      }
      updateRetLoc(ret, loc);
      return ret;
    }

    // Scope entry block tracking
    mlir::Block *getEntryBlock() { return EntryBlock; }

    mlir::Location BeginLoc, EndLoc;
  };

  LexicalScope *currLexScope = nullptr;

  class InlinedInheritingConstructorScope {
  public:
    InlinedInheritingConstructorScope(CIRGenFunction &CGF, GlobalDecl GD)
        : CGF(CGF), OldCurGD(CGF.CurGD), OldCurFuncDecl(CGF.CurFuncDecl),
          OldCurCodeDecl(CGF.CurCodeDecl),
          OldCXXABIThisDecl(CGF.CXXABIThisDecl),
          OldCXXABIThisValue(CGF.CXXABIThisValue),
          OldCXXThisValue(CGF.CXXThisValue),
          OldCXXABIThisAlignment(CGF.CXXABIThisAlignment),
          OldCXXThisAlignment(CGF.CXXThisAlignment),
          OldReturnValue(CGF.ReturnValue), OldFnRetTy(CGF.FnRetTy),
          OldCXXInheritedCtorInitExprArgs(
              std::move(CGF.CXXInheritedCtorInitExprArgs)) {
      CGF.CurGD = GD;
      CGF.CurFuncDecl = CGF.CurCodeDecl =
          mlir::cast<CXXConstructorDecl>(GD.getDecl());
      CGF.CXXABIThisDecl = nullptr;
      CGF.CXXABIThisValue = nullptr;
      CGF.CXXThisValue = nullptr;
      CGF.CXXABIThisAlignment = CharUnits();
      CGF.CXXThisAlignment = CharUnits();
      CGF.ReturnValue = Address::invalid();
      CGF.FnRetTy = QualType();
      CGF.CXXInheritedCtorInitExprArgs.clear();
    }
    ~InlinedInheritingConstructorScope() {
      CGF.CurGD = OldCurGD;
      CGF.CurFuncDecl = OldCurFuncDecl;
      CGF.CurCodeDecl = OldCurCodeDecl;
      CGF.CXXABIThisDecl = OldCXXABIThisDecl;
      CGF.CXXABIThisValue = OldCXXABIThisValue;
      CGF.CXXThisValue = OldCXXThisValue;
      CGF.CXXABIThisAlignment = OldCXXABIThisAlignment;
      CGF.CXXThisAlignment = OldCXXThisAlignment;
      CGF.ReturnValue = OldReturnValue;
      CGF.FnRetTy = OldFnRetTy;
      CGF.CXXInheritedCtorInitExprArgs =
          std::move(OldCXXInheritedCtorInitExprArgs);
    }

  private:
    CIRGenFunction &CGF;
    GlobalDecl OldCurGD;
    const Decl *OldCurFuncDecl;
    const Decl *OldCurCodeDecl;
    ImplicitParamDecl *OldCXXABIThisDecl;
    mlir::Value OldCXXABIThisValue;
    mlir::Value OldCXXThisValue;
    CharUnits OldCXXABIThisAlignment;
    CharUnits OldCXXThisAlignment;
    Address OldReturnValue;
    QualType OldFnRetTy;
    CallArgList OldCXXInheritedCtorInitExprArgs;
  };

  /// CIR build helpers
  /// -----------------

  /// This creates an alloca and inserts it into the entry block if \p ArraySize
  /// is nullptr,
  ///
  /// TODO(cir): ... otherwise inserts it at the current insertion point of
  ///            the builder.
  /// The caller is responsible for setting an appropriate alignment on
  /// the alloca.
  ///
  /// \p ArraySize is the number of array elements to be allocated if it
  ///    is not nullptr.
  ///
  /// LangAS::Default is the address space of pointers to local variables and
  /// temporaries, as exposed in the source language. In certain
  /// configurations, this is not the same as the alloca address space, and a
  /// cast is needed to lift the pointer from the alloca AS into
  /// LangAS::Default. This can happen when the target uses a restricted
  /// address space for the stack but the source language requires
  /// LangAS::Default to be a generic address space. The latter condition is
  /// common for most programming languages; OpenCL is an exception in that
  /// LangAS::Default is the private address space, which naturally maps
  /// to the stack.
  ///
  /// Because the address of a temporary is often exposed to the program in
  /// various ways, this function will perform the cast. The original alloca
  /// instruction is returned through \p Alloca if it is not nullptr.
  ///
  /// The cast is not performaed in CreateTempAllocaWithoutCast. This is
  /// more efficient if the caller knows that the address will not be exposed.
  cir::AllocaOp CreateTempAlloca(mlir::Type Ty, mlir::Location Loc,
                                 const Twine &Name = "tmp",
                                 mlir::Value ArraySize = nullptr,
                                 bool insertIntoFnEntryBlock = false);
  cir::AllocaOp CreateTempAllocaInFnEntryBlock(mlir::Type Ty,
                                               mlir::Location Loc,
                                               const Twine &Name = "tmp",
                                               mlir::Value ArraySize = nullptr);
  cir::AllocaOp CreateTempAlloca(mlir::Type Ty, mlir::Location Loc,
                                 const Twine &Name = "tmp",
                                 mlir::OpBuilder::InsertPoint ip = {},
                                 mlir::Value ArraySize = nullptr);
  Address CreateTempAlloca(mlir::Type Ty, CharUnits align, mlir::Location Loc,
                           const Twine &Name = "tmp",
                           mlir::Value ArraySize = nullptr,
                           Address *Alloca = nullptr,
                           mlir::OpBuilder::InsertPoint ip = {});
  Address CreateTempAllocaWithoutCast(mlir::Type Ty, CharUnits align,
                                      mlir::Location Loc,
                                      const Twine &Name = "tmp",
                                      mlir::Value ArraySize = nullptr,
                                      mlir::OpBuilder::InsertPoint ip = {});

  /// Create a temporary memory object of the given type, with
  /// appropriate alignmen and cast it to the default address space. Returns
  /// the original alloca instruction by \p Alloca if it is not nullptr.
  Address CreateMemTemp(QualType T, mlir::Location Loc,
                        const Twine &Name = "tmp", Address *Alloca = nullptr,
                        mlir::OpBuilder::InsertPoint ip = {});
  Address CreateMemTemp(QualType T, CharUnits Align, mlir::Location Loc,
                        const Twine &Name = "tmp", Address *Alloca = nullptr,
                        mlir::OpBuilder::InsertPoint ip = {});

  /// Create a temporary memory object of the given type, with
  /// appropriate alignment without casting it to the default address space.
  Address CreateMemTempWithoutCast(QualType T, mlir::Location Loc,
                                   const Twine &Name = "tmp");
  Address CreateMemTempWithoutCast(QualType T, CharUnits Align,
                                   mlir::Location Loc,
                                   const Twine &Name = "tmp");

  /// Create a temporary memory object for the given
  /// aggregate type.
  AggValueSlot CreateAggTemp(QualType T, mlir::Location Loc,
                             const Twine &Name = "tmp",
                             Address *Alloca = nullptr) {
    return AggValueSlot::forAddr(
        CreateMemTemp(T, Loc, Name, Alloca), T.getQualifiers(),
        AggValueSlot::IsNotDestructed, AggValueSlot::DoesNotNeedGCBarriers,
        AggValueSlot::IsNotAliased, AggValueSlot::DoesNotOverlap);
  }

private:
  QualType getVarArgType(const Expr *Arg);
};

/// Helper class with most of the code for saving a value for a
/// conditional expression cleanup.
struct DominatingCIRValue {
  typedef llvm::PointerIntPair<mlir::Value, 1, bool> saved_type;

  /// Answer whether the given value needs extra work to be saved.
  static bool needsSaving(mlir::Value value) {
    if (!value)
      return false;

    // If it's a block argument, we don't need to save.
    mlir::Operation *definingOp = value.getDefiningOp();
    if (!definingOp)
      return false;

    // If value is defined the function or a global init entry block, we don't
    // need to save.
    mlir::Block *currBlock = definingOp->getBlock();
    if (!currBlock->isEntryBlock() || !definingOp->getParentOp())
      return false;

    if (auto fnOp = definingOp->getParentOfType<cir::FuncOp>()) {
      if (&fnOp.getBody().front() == currBlock)
        return true;
      return false;
    }

    if (auto globalOp = definingOp->getParentOfType<cir::GlobalOp>()) {
      assert(globalOp.getNumRegions() == 2 && "other regions NYI");
      if (&globalOp.getCtorRegion().front() == currBlock)
        return true;
      if (&globalOp.getDtorRegion().front() == currBlock)
        return true;
      return false;
    }

    return false;
  }

  static saved_type save(CIRGenFunction &CGF, mlir::Value value);
  static mlir::Value restore(CIRGenFunction &CGF, saved_type value);
};

inline DominatingCIRValue::saved_type
DominatingCIRValue::save(CIRGenFunction &CGF, mlir::Value value) {
  if (!needsSaving(value))
    return saved_type(value, false);

  // Otherwise, we need an alloca.
  auto align = CharUnits::fromQuantity(
      CGF.CGM.getDataLayout().getPrefTypeAlign(value.getType()));
  mlir::Location loc = value.getLoc();
  Address alloca =
      CGF.CreateTempAlloca(value.getType(), align, loc, "cond-cleanup.save");
  CGF.getBuilder().createStore(loc, value, alloca);

  return saved_type(alloca.emitRawPointer(), true);
}

inline mlir::Value DominatingCIRValue::restore(CIRGenFunction &CGF,
                                               saved_type value) {
  // If the value says it wasn't saved, trust that it's still dominating.
  if (!value.getInt())
    return value.getPointer();

  // Otherwise, it should be an alloca instruction, as set up in save().
  auto alloca = mlir::cast<cir::AllocaOp>(value.getPointer().getDefiningOp());
  mlir::Value val = CGF.getBuilder().createAlignedLoad(
      alloca.getLoc(), alloca.getType(), alloca);
  cir::LoadOp loadOp = mlir::cast<cir::LoadOp>(val.getDefiningOp());
  loadOp.setAlignment(alloca.getAlignment());
  return val;
}

/// A specialization of DominatingValue for RValue.
template <> struct DominatingValue<RValue> {
  typedef RValue type;
  class saved_type {
    enum Kind {
      ScalarLiteral,
      ScalarAddress,
      AggregateLiteral,
      AggregateAddress,
      ComplexAddress
    };
    union {
      struct {
        DominatingCIRValue::saved_type first, second;
      } Vals;
      DominatingValue<Address>::saved_type AggregateAddr;
    };
    LLVM_PREFERRED_TYPE(Kind)
    unsigned K : 3;

    saved_type(DominatingCIRValue::saved_type Val1, unsigned K)
        : Vals{Val1, DominatingCIRValue::saved_type()}, K(K) {}

    saved_type(DominatingCIRValue::saved_type Val1,
               DominatingCIRValue::saved_type Val2)
        : Vals{Val1, Val2}, K(ComplexAddress) {}

    saved_type(DominatingValue<Address>::saved_type AggregateAddr, unsigned K)
        : AggregateAddr(AggregateAddr), K(K) {}

  public:
    static bool needsSaving(RValue value);
    static saved_type save(CIRGenFunction &CGF, RValue value);
    RValue restore(CIRGenFunction &CGF);
  };

  static bool needsSaving(type value) { return saved_type::needsSaving(value); }
  static saved_type save(CIRGenFunction &CGF, type value) {
    return saved_type::save(CGF, value);
  }
  static type restore(CIRGenFunction &CGF, saved_type value) {
    return value.restore(CGF);
  }
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H
