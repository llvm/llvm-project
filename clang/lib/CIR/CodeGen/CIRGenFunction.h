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
#include "EHScopeStack.h"

#include "Address.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/BaseSubobject.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/TypeEvaluationKind.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace {
class ScalarExprEmitter;
} // namespace

namespace mlir {
namespace acc {
class LoopOp;
} // namespace acc
} // namespace mlir

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

  /// Tracks function scope overall cleanup handling.
  EHScopeStack ehStack;

  /// CXXThisDecl - When generating code for a C++ member function,
  /// this will hold the implicit 'this' declaration.
  ImplicitParamDecl *cxxabiThisDecl = nullptr;
  mlir::Value cxxabiThisValue = nullptr;
  mlir::Value cxxThisValue = nullptr;
  clang::CharUnits cxxThisAlignment;

  /// The value of 'this' to sue when evaluating CXXDefaultInitExprs within this
  /// expression.
  Address cxxDefaultInitExprThis = Address::invalid();

  // Holds the Decl for the current outermost non-closure context
  const clang::Decl *curFuncDecl = nullptr;

  /// The function for which code is currently being generated.
  cir::FuncOp curFn;

  using DeclMapTy = llvm::DenseMap<const clang::Decl *, Address>;
  /// This keeps track of the CIR allocas or globals for local C
  /// declarations.
  DeclMapTy localDeclMap;

  /// The type of the condition for the emitting switch statement.
  llvm::SmallVector<mlir::Type, 2> condTypeStack;

  clang::ASTContext &getContext() const { return cgm.getASTContext(); }

  CIRGenBuilderTy &getBuilder() { return builder; }

  CIRGenModule &getCIRGenModule() { return cgm; }
  const CIRGenModule &getCIRGenModule() const { return cgm; }

  mlir::Block *getCurFunctionEntryBlock() { return &curFn.getRegion().front(); }

  /// Sanitizers enabled for this function.
  clang::SanitizerSet sanOpts;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated,
  /// the scope is destroyed and the mappings created in this scope are
  /// dropped.
  using SymTableTy = llvm::ScopedHashTable<const clang::Decl *, mlir::Value>;
  SymTableTy symbolTable;

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

  static bool hasAggregateEvaluationKind(clang::QualType type) {
    return getEvaluationKind(type) == cir::TEK_Aggregate;
  }

  CIRGenFunction(CIRGenModule &cgm, CIRGenBuilderTy &builder,
                 bool suppressNewContext = false);
  ~CIRGenFunction();

  CIRGenTypes &getTypes() const { return cgm.getTypes(); }

  const TargetInfo &getTarget() const { return cgm.getTarget(); }
  mlir::MLIRContext &getMLIRContext() { return cgm.getMLIRContext(); }

  // ---------------------
  // Opaque value handling
  // ---------------------

  /// Keeps track of the current set of opaque value expressions.
  llvm::DenseMap<const OpaqueValueExpr *, LValue> opaqueLValues;
  llvm::DenseMap<const OpaqueValueExpr *, RValue> opaqueRValues;

public:
  /// A non-RAII class containing all the information about a bound
  /// opaque value.  OpaqueValueMapping, below, is a RAII wrapper for
  /// this which makes individual mappings very simple; using this
  /// class directly is useful when you have a variable number of
  /// opaque values or don't want the RAII functionality for some
  /// reason.
  class OpaqueValueMappingData {
    const OpaqueValueExpr *opaqueValue;
    bool boundLValue;

    OpaqueValueMappingData(const OpaqueValueExpr *ov, bool boundLValue)
        : opaqueValue(ov), boundLValue(boundLValue) {}

  public:
    OpaqueValueMappingData() : opaqueValue(nullptr) {}

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
    bind(CIRGenFunction &cgf, const OpaqueValueExpr *ov, const Expr *e) {
      if (shouldBindAsLValue(ov))
        return bind(cgf, ov, cgf.emitLValue(e));
      return bind(cgf, ov, cgf.emitAnyExpr(e));
    }

    static OpaqueValueMappingData
    bind(CIRGenFunction &cgf, const OpaqueValueExpr *ov, const LValue &lv) {
      assert(shouldBindAsLValue(ov));
      cgf.opaqueLValues.insert(std::make_pair(ov, lv));
      return OpaqueValueMappingData(ov, true);
    }

    static OpaqueValueMappingData
    bind(CIRGenFunction &cgf, const OpaqueValueExpr *ov, const RValue &rv) {
      assert(!shouldBindAsLValue(ov));
      cgf.opaqueRValues.insert(std::make_pair(ov, rv));

      OpaqueValueMappingData data(ov, false);

      // Work around an extremely aggressive peephole optimization in
      // EmitScalarConversion which assumes that all other uses of a
      // value are extant.
      assert(!cir::MissingFeatures::peepholeProtection() && "NYI");
      return data;
    }

    bool isValid() const { return opaqueValue != nullptr; }
    void clear() { opaqueValue = nullptr; }

    void unbind(CIRGenFunction &cgf) {
      assert(opaqueValue && "no data to unbind!");

      if (boundLValue) {
        cgf.opaqueLValues.erase(opaqueValue);
      } else {
        cgf.opaqueRValues.erase(opaqueValue);
        assert(!cir::MissingFeatures::peepholeProtection() && "NYI");
      }
    }
  };

  /// An RAII object to set (and then clear) a mapping for an OpaqueValueExpr.
  class OpaqueValueMapping {
    CIRGenFunction &cgf;
    OpaqueValueMappingData data;

  public:
    static bool shouldBindAsLValue(const Expr *expr) {
      return OpaqueValueMappingData::shouldBindAsLValue(expr);
    }

    /// Build the opaque value mapping for the given conditional
    /// operator if it's the GNU ?: extension.  This is a common
    /// enough pattern that the convenience operator is really
    /// helpful.
    ///
    OpaqueValueMapping(CIRGenFunction &cgf,
                       const AbstractConditionalOperator *op)
        : cgf(cgf) {
      if (mlir::isa<ConditionalOperator>(op))
        // Leave Data empty.
        return;

      const BinaryConditionalOperator *e =
          mlir::cast<BinaryConditionalOperator>(op);
      data = OpaqueValueMappingData::bind(cgf, e->getOpaqueValue(),
                                          e->getCommon());
    }

    /// Build the opaque value mapping for an OpaqueValueExpr whose source
    /// expression is set to the expression the OVE represents.
    OpaqueValueMapping(CIRGenFunction &cgf, const OpaqueValueExpr *ov)
        : cgf(cgf) {
      if (ov) {
        assert(ov->getSourceExpr() && "wrong form of OpaqueValueMapping used "
                                      "for OVE with no source expression");
        data = OpaqueValueMappingData::bind(cgf, ov, ov->getSourceExpr());
      }
    }

    OpaqueValueMapping(CIRGenFunction &cgf, const OpaqueValueExpr *opaqueValue,
                       LValue lvalue)
        : cgf(cgf),
          data(OpaqueValueMappingData::bind(cgf, opaqueValue, lvalue)) {}

    OpaqueValueMapping(CIRGenFunction &cgf, const OpaqueValueExpr *opaqueValue,
                       RValue rvalue)
        : cgf(cgf),
          data(OpaqueValueMappingData::bind(cgf, opaqueValue, rvalue)) {}

    void pop() {
      data.unbind(cgf);
      data.clear();
    }

    ~OpaqueValueMapping() {
      if (data.isValid())
        data.unbind(cgf);
    }
  };

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

  using SymTableScopeTy =
      llvm::ScopedHashTableScope<const clang::Decl *, mlir::Value>;

  /// Hold counters for incrementally naming temporaries
  unsigned counterRefTmp = 0;
  unsigned counterAggTmp = 0;
  std::string getCounterRefTmpAsString();
  std::string getCounterAggTmpAsString();

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(clang::SourceLocation srcLoc);
  mlir::Location getLoc(clang::SourceRange srcLoc);
  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  const clang::LangOptions &getLangOpts() const { return cgm.getLangOpts(); }

  /// True if an insertion point is defined. If not, this indicates that the
  /// current code being emitted is unreachable.
  /// FIXME(cir): we need to inspect this and perhaps use a cleaner mechanism
  /// since we don't yet force null insertion point to designate behavior (like
  /// LLVM's codegen does) and we probably shouldn't.
  bool haveInsertPoint() const {
    return builder.getInsertionBlock() != nullptr;
  }

  // Wrapper for function prototype sources. Wraps either a FunctionProtoType or
  // an ObjCMethodDecl.
  struct PrototypeWrapper {
    llvm::PointerUnion<const clang::FunctionProtoType *,
                       const clang::ObjCMethodDecl *>
        p;

    PrototypeWrapper(const clang::FunctionProtoType *ft) : p(ft) {}
    PrototypeWrapper(const clang::ObjCMethodDecl *md) : p(md) {}
  };

  bool isLValueSuitableForInlineAtomic(LValue lv);

  /// An abstract representation of regular/ObjC call/message targets.
  class AbstractCallee {
    /// The function declaration of the callee.
    [[maybe_unused]] const clang::Decl *calleeDecl;

  public:
    AbstractCallee() : calleeDecl(nullptr) {}
    AbstractCallee(const clang::FunctionDecl *fd) : calleeDecl(fd) {}

    bool hasFunctionDecl() const {
      return llvm::isa_and_nonnull<clang::FunctionDecl>(calleeDecl);
    }

    unsigned getNumParams() const {
      if (const auto *fd = llvm::dyn_cast<clang::FunctionDecl>(calleeDecl))
        return fd->getNumParams();
      return llvm::cast<clang::ObjCMethodDecl>(calleeDecl)->param_size();
    }

    const clang::ParmVarDecl *getParamDecl(unsigned I) const {
      if (const auto *fd = llvm::dyn_cast<clang::FunctionDecl>(calleeDecl))
        return fd->getParamDecl(I);
      return *(llvm::cast<clang::ObjCMethodDecl>(calleeDecl)->param_begin() +
               I);
    }
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

  class ConstantEmission {
    // Cannot use mlir::TypedAttr directly here because of bit availability.
    llvm::PointerIntPair<mlir::Attribute, 1, bool> valueAndIsReference;
    ConstantEmission(mlir::TypedAttr c, bool isReference)
        : valueAndIsReference(c, isReference) {}

  public:
    ConstantEmission() {}
    static ConstantEmission forReference(mlir::TypedAttr c) {
      return ConstantEmission(c, true);
    }
    static ConstantEmission forValue(mlir::TypedAttr c) {
      return ConstantEmission(c, false);
    }

    explicit operator bool() const {
      return valueAndIsReference.getOpaqueValue() != nullptr;
    }

    bool isReference() const { return valueAndIsReference.getInt(); }
    LValue getReferenceLValue(CIRGenFunction &cgf, Expr *refExpr) const {
      assert(isReference());
      cgf.cgm.errorNYI(refExpr->getSourceRange(),
                       "ConstantEmission::getReferenceLValue");
      return {};
    }

    mlir::TypedAttr getValue() const {
      assert(!isReference());
      return mlir::cast<mlir::TypedAttr>(valueAndIsReference.getPointer());
    }
  };

  ConstantEmission tryEmitAsConstant(DeclRefExpr *refExpr);

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
    Address getObjectAddress(CIRGenFunction &cgf) const {
      if (!IsEscapingByRef)
        return Addr;

      assert(!cir::MissingFeatures::opAllocaEscapeByReference());
      return Address::invalid();
    }
  };

  /// Perform the usual unary conversions on the specified expression and
  /// compare the result against zero, returning an Int1Ty value.
  mlir::Value evaluateExprAsBool(const clang::Expr *e);

  cir::GlobalOp addInitializerToStaticVarDecl(const VarDecl &d,
                                              cir::GlobalOp gv,
                                              cir::GetGlobalOp gvAddr);

  /// Set the address of a local variable.
  void setAddrOfLocalVar(const clang::VarDecl *vd, Address addr) {
    assert(!localDeclMap.count(vd) && "Decl already exists in LocalDeclMap!");
    localDeclMap.insert({vd, addr});

    // Add to the symbol table if not there already.
    if (symbolTable.count(vd))
      return;
    symbolTable.insert(vd, addr.getPointer());
  }

  /// Removes a declaration from the address-relationship.  This is a function
  /// that shouldn't need to be used except in cases where we're adding/removing
  /// things that aren't part of the language-semantics AST.
  void removeAddrOfLocalVar(const clang::VarDecl *vd) {
    localDeclMap.erase(vd);
  }

  bool shouldNullCheckClassCastValue(const CastExpr *ce);

  RValue convertTempToRValue(Address addr, clang::QualType type,
                             clang::SourceLocation loc);

  static bool
  isConstructorDelegationValid(const clang::CXXConstructorDecl *ctor);

  struct VPtr {
    clang::BaseSubobject base;
    const clang::CXXRecordDecl *nearestVBase;
    clang::CharUnits offsetFromNearestVBase;
    const clang::CXXRecordDecl *vtableClass;
  };

  using VPtrsVector = llvm::SmallVector<VPtr, 4>;
  VPtrsVector getVTablePointers(const clang::CXXRecordDecl *vtableClass);
  void getVTablePointers(clang::BaseSubobject base,
                         const clang::CXXRecordDecl *nearestVBase,
                         clang::CharUnits offsetFromNearestVBase,
                         bool baseIsNonVirtualPrimaryBase,
                         const clang::CXXRecordDecl *vtableClass,
                         VPtrsVector &vptrs);
  /// Return the Value of the vtable pointer member pointed to by thisAddr.
  mlir::Value getVTablePtr(mlir::Location loc, Address thisAddr,
                           const clang::CXXRecordDecl *vtableClass);

  /// A scope within which we are constructing the fields of an object which
  /// might use a CXXDefaultInitExpr. This stashes away a 'this' value to use if
  /// we need to evaluate the CXXDefaultInitExpr within the evaluation.
  class FieldConstructionScope {
  public:
    FieldConstructionScope(CIRGenFunction &cgf, Address thisAddr)
        : cgf(cgf), oldCXXDefaultInitExprThis(cgf.cxxDefaultInitExprThis) {
      cgf.cxxDefaultInitExprThis = thisAddr;
    }
    ~FieldConstructionScope() {
      cgf.cxxDefaultInitExprThis = oldCXXDefaultInitExprThis;
    }

  private:
    CIRGenFunction &cgf;
    Address oldCXXDefaultInitExprThis;
  };

  LValue makeNaturalAlignPointeeAddrLValue(mlir::Value v, clang::QualType t);
  LValue makeNaturalAlignAddrLValue(mlir::Value val, QualType ty);

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

  Address getAddressOfBaseClass(
      Address value, const CXXRecordDecl *derived,
      llvm::iterator_range<CastExpr::path_const_iterator> path,
      bool nullCheckValue, SourceLocation loc);

  LValue makeAddrLValue(Address addr, QualType ty,
                        AlignmentSource source = AlignmentSource::Type) {
    return makeAddrLValue(addr, ty, LValueBaseInfo(source));
  }

  LValue makeAddrLValue(Address addr, QualType ty, LValueBaseInfo baseInfo) {
    return LValue::makeAddr(addr, ty, baseInfo);
  }

  void initializeVTablePointers(mlir::Location loc,
                                const clang::CXXRecordDecl *rd);
  void initializeVTablePointer(mlir::Location loc, const VPtr &vptr);

  /// Return the address of a local variable.
  Address getAddrOfLocalVar(const clang::VarDecl *vd) {
    auto it = localDeclMap.find(vd);
    assert(it != localDeclMap.end() &&
           "Invalid argument to getAddrOfLocalVar(), no decl!");
    return it->second;
  }

  Address getAddrOfBitFieldStorage(LValue base, const clang::FieldDecl *field,
                                   mlir::Type fieldType, unsigned index);

  /// Load the value for 'this'. This function is only valid while generating
  /// code for an C++ member function.
  /// FIXME(cir): this should return a mlir::Value!
  mlir::Value loadCXXThis() {
    assert(cxxThisValue && "no 'this' value for this function");
    return cxxThisValue;
  }
  Address loadCXXThisAddress();

  /// Convert the given pointer to a complete class to the given direct base.
  Address getAddressOfDirectBaseInCompleteClass(mlir::Location loc,
                                                Address value,
                                                const CXXRecordDecl *derived,
                                                const CXXRecordDecl *base,
                                                bool baseIsVirtual);

  /// Determine whether a base class initialization may overlap some other
  /// object.
  AggValueSlot::Overlap_t getOverlapForBaseInit(const CXXRecordDecl *rd,
                                                const CXXRecordDecl *baseRD,
                                                bool isVirtual);

  /// Get an appropriate 'undef' rvalue for the given type.
  /// TODO: What's the equivalent for MLIR? Currently we're only using this for
  /// void types so it just returns RValue::get(nullptr) but it'll need
  /// addressed later.
  RValue getUndefRValue(clang::QualType ty);

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

  /// The cleanup depth enclosing all the cleanups associated with the
  /// parameters.
  EHScopeStack::stable_iterator prologueCleanupDepth;

  /// Takes the old cleanup stack size and emits the cleanup blocks
  /// that have been added.
  void popCleanupBlocks(EHScopeStack::stable_iterator oldCleanupStackDepth);
  void popCleanupBlock();

  /// Push a cleanup to be run at the end of the current full-expression.  Safe
  /// against the possibility that we're currently inside a
  /// conditionally-evaluated expression.
  template <class T, class... As>
  void pushFullExprCleanup(CleanupKind kind, As... a) {
    // If we're not in a conditional branch, or if none of the
    // arguments requires saving, then use the unconditional cleanup.
    if (!isInConditionalBranch())
      return ehStack.pushCleanup<T>(kind, a...);

    cgm.errorNYI("pushFullExprCleanup in conditional branch");
  }

  /// Enters a new scope for capturing cleanups, all of which
  /// will be executed once the scope is exited.
  class RunCleanupsScope {
    EHScopeStack::stable_iterator cleanupStackDepth, oldCleanupStackDepth;

  protected:
    bool performCleanup;

  private:
    RunCleanupsScope(const RunCleanupsScope &) = delete;
    void operator=(const RunCleanupsScope &) = delete;

  protected:
    CIRGenFunction &cgf;

  public:
    /// Enter a new cleanup scope.
    explicit RunCleanupsScope(CIRGenFunction &cgf)
        : performCleanup(true), cgf(cgf) {
      cleanupStackDepth = cgf.ehStack.stable_begin();
      oldCleanupStackDepth = cgf.currentCleanupStackDepth;
      cgf.currentCleanupStackDepth = cleanupStackDepth;
    }

    /// Exit this cleanup scope, emitting any accumulated cleanups.
    ~RunCleanupsScope() {
      if (performCleanup)
        forceCleanup();
    }

    /// Force the emission of cleanups now, instead of waiting
    /// until this object is destroyed.
    void forceCleanup() {
      assert(performCleanup && "Already forced cleanup");
      {
        mlir::OpBuilder::InsertionGuard guard(cgf.getBuilder());
        cgf.popCleanupBlocks(cleanupStackDepth);
        performCleanup = false;
        cgf.currentCleanupStackDepth = oldCleanupStackDepth;
      }
    }
  };

  // Cleanup stack depth of the RunCleanupsScope that was pushed most recently.
  EHScopeStack::stable_iterator currentCleanupStackDepth = ehStack.stable_end();

public:
  /// Represents a scope, including function bodies, compound statements, and
  /// the substatements of if/while/do/for/switch/try statements.  This class
  /// handles any automatic cleanup, along with the return value.
  struct LexicalScope : public RunCleanupsScope {
  private:
    // Block containing cleanup code for things initialized in this
    // lexical context (scope).
    mlir::Block *cleanupBlock = nullptr;

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
        : RunCleanupsScope(cgf), entryBlock(eb), parentScope(cgf.curLexScope),
          beginLoc(loc), endLoc(loc) {

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

    // Lazy create cleanup block or return what's available.
    mlir::Block *getOrCreateCleanupBlock(mlir::OpBuilder &builder) {
      if (cleanupBlock)
        return cleanupBlock;
      cleanupBlock = createCleanupBlock(builder);
      return cleanupBlock;
    }

    mlir::Block *getCleanupBlock(mlir::OpBuilder &builder) {
      return cleanupBlock;
    }

    mlir::Block *createCleanupBlock(mlir::OpBuilder &builder) {
      // Create the cleanup block but dont hook it up around just yet.
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Region *r = builder.getBlock() ? builder.getBlock()->getParent()
                                           : &cgf.curFn->getRegion(0);
      cleanupBlock = builder.createBlock(r);
      return cleanupBlock;
    }

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

  typedef void Destroyer(CIRGenFunction &cgf, Address addr, QualType ty);

  static Destroyer destroyCXXObject;

  void pushDestroy(CleanupKind kind, Address addr, QualType type,
                   Destroyer *destroyer);

  Destroyer *getDestroyer(clang::QualType::DestructionKind kind);

  /// ----------------------
  /// CIR emit functions
  /// ----------------------
public:
  mlir::Value emitAlignmentAssumption(mlir::Value ptrValue, QualType ty,
                                      SourceLocation loc,
                                      SourceLocation assumptionLoc,
                                      int64_t alignment,
                                      mlir::Value offsetValue = nullptr);

  mlir::Value emitAlignmentAssumption(mlir::Value ptrValue, const Expr *expr,
                                      SourceLocation assumptionLoc,
                                      int64_t alignment,
                                      mlir::Value offsetValue = nullptr);

private:
  void emitAndUpdateRetAlloca(clang::QualType type, mlir::Location loc,
                              clang::CharUnits alignment);

  CIRGenCallee emitDirectCallee(const GlobalDecl &gd);

public:
  Address emitAddrOfFieldStorage(Address base, const FieldDecl *field,
                                 llvm::StringRef fieldName,
                                 unsigned fieldIndex);

  mlir::Value emitAlloca(llvm::StringRef name, mlir::Type ty,
                         mlir::Location loc, clang::CharUnits alignment,
                         bool insertIntoFnEntryBlock,
                         mlir::Value arraySize = nullptr);
  mlir::Value emitAlloca(llvm::StringRef name, mlir::Type ty,
                         mlir::Location loc, clang::CharUnits alignment,
                         mlir::OpBuilder::InsertPoint ip,
                         mlir::Value arraySize = nullptr);

  void emitAggregateStore(mlir::Value value, Address dest);

  void emitAggExpr(const clang::Expr *e, AggValueSlot slot);

  LValue emitAggExprToLValue(const Expr *e);

  /// Emit code to compute the specified expression which can have any type. The
  /// result is returned as an RValue struct. If this is an aggregate
  /// expression, the aggloc/agglocvolatile arguments indicate where the result
  /// should be returned.
  RValue emitAnyExpr(const clang::Expr *e,
                     AggValueSlot aggSlot = AggValueSlot::ignored());

  /// Emits the code necessary to evaluate an arbitrary expression into the
  /// given memory location.
  void emitAnyExprToMem(const Expr *e, Address location, Qualifiers quals,
                        bool isInitializer);

  /// Similarly to emitAnyExpr(), however, the result will always be accessible
  /// even if no aggregate location is provided.
  RValue emitAnyExprToTemp(const clang::Expr *e);

  void emitArrayDestroy(mlir::Value begin, mlir::Value end,
                        QualType elementType, CharUnits elementAlign,
                        Destroyer *destroyer);

  mlir::Value emitArrayLength(const clang::ArrayType *arrayType,
                              QualType &baseType, Address &addr);
  LValue emitArraySubscriptExpr(const clang::ArraySubscriptExpr *e);

  Address emitArrayToPointerDecay(const Expr *array);

  RValue emitAtomicExpr(AtomicExpr *e);
  void emitAtomicInit(Expr *init, LValue dest);

  AutoVarEmission emitAutoVarAlloca(const clang::VarDecl &d,
                                    mlir::OpBuilder::InsertPoint ip = {});

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void emitAutoVarDecl(const clang::VarDecl &d);

  void emitAutoVarCleanups(const AutoVarEmission &emission);
  void emitAutoVarInit(const AutoVarEmission &emission);
  void emitAutoVarTypeCleanup(const AutoVarEmission &emission,
                              clang::QualType::DestructionKind dtorKind);

  void maybeEmitDeferredVarDeclInit(const VarDecl *vd);

  void emitBaseInitializer(mlir::Location loc, const CXXRecordDecl *classDecl,
                           CXXCtorInitializer *baseInit);

  LValue emitBinaryOperatorLValue(const BinaryOperator *e);

  mlir::LogicalResult emitBreakStmt(const clang::BreakStmt &s);

  RValue emitBuiltinExpr(const clang::GlobalDecl &gd, unsigned builtinID,
                         const clang::CallExpr *e, ReturnValueSlot returnValue);

  RValue emitCall(const CIRGenFunctionInfo &funcInfo,
                  const CIRGenCallee &callee, ReturnValueSlot returnValue,
                  const CallArgList &args, cir::CIRCallOpInterface *callOp,
                  mlir::Location loc);
  RValue emitCall(const CIRGenFunctionInfo &funcInfo,
                  const CIRGenCallee &callee, ReturnValueSlot returnValue,
                  const CallArgList &args,
                  cir::CIRCallOpInterface *callOrTryCall = nullptr) {
    assert(currSrcLoc && "source location must have been set");
    return emitCall(funcInfo, callee, returnValue, args, callOrTryCall,
                    *currSrcLoc);
  }

  RValue emitCall(clang::QualType calleeTy, const CIRGenCallee &callee,
                  const clang::CallExpr *e, ReturnValueSlot returnValue);
  void emitCallArg(CallArgList &args, const clang::Expr *e,
                   clang::QualType argType);
  void emitCallArgs(
      CallArgList &args, PrototypeWrapper prototype,
      llvm::iterator_range<clang::CallExpr::const_arg_iterator> argRange,
      AbstractCallee callee = AbstractCallee(), unsigned paramsToSkip = 0);
  RValue emitCallExpr(const clang::CallExpr *e,
                      ReturnValueSlot returnValue = ReturnValueSlot());
  LValue emitCallExprLValue(const clang::CallExpr *e);
  CIRGenCallee emitCallee(const clang::Expr *e);

  template <typename T>
  mlir::LogicalResult emitCaseDefaultCascade(const T *stmt, mlir::Type condType,
                                             mlir::ArrayAttr value,
                                             cir::CaseOpKind kind,
                                             bool buildingTopLevelCase);

  mlir::LogicalResult emitCaseStmt(const clang::CaseStmt &s,
                                   mlir::Type condType,
                                   bool buildingTopLevelCase);

  LValue emitCastLValue(const CastExpr *e);

  /// Emits an argument for a call to a `__builtin_assume`. If the builtin
  /// sanitizer is enabled, a runtime check is also emitted.
  mlir::Value emitCheckedArgForAssume(const Expr *e);

  /// Emit a conversion from the specified complex type to the specified
  /// destination type, where the destination type is an LLVM scalar type.
  mlir::Value emitComplexToScalarConversion(mlir::Value src, QualType srcTy,
                                            QualType dstTy, SourceLocation loc);

  LValue emitCompoundAssignmentLValue(const clang::CompoundAssignOperator *e);
  LValue emitCompoundLiteralLValue(const CompoundLiteralExpr *e);

  void emitConstructorBody(FunctionArgList &args);

  void emitDestroy(Address addr, QualType type, Destroyer *destroyer);

  void emitDestructorBody(FunctionArgList &args);

  mlir::LogicalResult emitContinueStmt(const clang::ContinueStmt &s);

  void emitCXXConstructExpr(const clang::CXXConstructExpr *e,
                            AggValueSlot dest);

  void emitCXXAggrConstructorCall(const CXXConstructorDecl *ctor,
                                  const clang::ArrayType *arrayType,
                                  Address arrayBegin, const CXXConstructExpr *e,
                                  bool newPointerIsChecked,
                                  bool zeroInitialize = false);
  void emitCXXAggrConstructorCall(const CXXConstructorDecl *ctor,
                                  mlir::Value numElements, Address arrayBase,
                                  const CXXConstructExpr *e,
                                  bool newPointerIsChecked,
                                  bool zeroInitialize);
  void emitCXXConstructorCall(const clang::CXXConstructorDecl *d,
                              clang::CXXCtorType type, bool forVirtualBase,
                              bool delegating, AggValueSlot thisAVS,
                              const clang::CXXConstructExpr *e);

  void emitCXXConstructorCall(const clang::CXXConstructorDecl *d,
                              clang::CXXCtorType type, bool forVirtualBase,
                              bool delegating, Address thisAddr,
                              CallArgList &args, clang::SourceLocation loc);

  void emitCXXDestructorCall(const CXXDestructorDecl *dd, CXXDtorType type,
                             bool forVirtualBase, bool delegating,
                             Address thisAddr, QualType thisTy);

  RValue emitCXXDestructorCall(GlobalDecl dtor, const CIRGenCallee &callee,
                               mlir::Value thisVal, QualType thisTy,
                               mlir::Value implicitParam,
                               QualType implicitParamTy, const CallExpr *e);

  mlir::LogicalResult emitCXXForRangeStmt(const CXXForRangeStmt &s,
                                          llvm::ArrayRef<const Attr *> attrs);

  RValue emitCXXMemberCallExpr(const clang::CXXMemberCallExpr *e,
                               ReturnValueSlot returnValue);

  RValue emitCXXMemberOrOperatorCall(
      const clang::CXXMethodDecl *md, const CIRGenCallee &callee,
      ReturnValueSlot returnValue, mlir::Value thisPtr,
      mlir::Value implicitParam, clang::QualType implicitParamTy,
      const clang::CallExpr *ce, CallArgList *rtlArgs);

  RValue emitCXXMemberOrOperatorMemberCallExpr(
      const clang::CallExpr *ce, const clang::CXXMethodDecl *md,
      ReturnValueSlot returnValue, bool hasQualifier,
      clang::NestedNameSpecifier qualifier, bool isArrow,
      const clang::Expr *base);

  mlir::Value emitCXXNewExpr(const CXXNewExpr *e);

  RValue emitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *e,
                                       const CXXMethodDecl *md,
                                       ReturnValueSlot returnValue);

  RValue emitCXXPseudoDestructorExpr(const CXXPseudoDestructorExpr *expr);

  void emitCtorPrologue(const clang::CXXConstructorDecl *ctor,
                        clang::CXXCtorType ctorType, FunctionArgList &args);

  // It's important not to confuse this and emitDelegateCXXConstructorCall.
  // Delegating constructors are the C++11 feature. The constructor delegate
  // optimization is used to reduce duplication in the base and complete
  // constructors where they are substantially the same.
  void emitDelegatingCXXConstructorCall(const CXXConstructorDecl *ctor,
                                        const FunctionArgList &args);

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

  void emitImplicitAssignmentOperatorBody(FunctionArgList &args);

  void emitInitializerForField(clang::FieldDecl *field, LValue lhs,
                               clang::Expr *init);

  mlir::Value emitPromotedComplexExpr(const Expr *e, QualType promotionType);

  mlir::Value emitPromotedScalarExpr(const Expr *e, QualType promotionType);

  mlir::Value emitPromotedValue(mlir::Value result, QualType promotionType);

  /// Emit the computation of the specified expression of scalar type.
  mlir::Value emitScalarExpr(const clang::Expr *e);

  mlir::Value emitScalarPrePostIncDec(const UnaryOperator *e, LValue lv,
                                      cir::UnaryOpKind kind, bool isPre);

  /// Build a debug stoppoint if we are emitting debug info.
  void emitStopPoint(const Stmt *s);

  // Build CIR for a statement. useCurrentScope should be true if no
  // new scopes need be created when finding a compound statement.
  mlir::LogicalResult emitStmt(const clang::Stmt *s, bool useCurrentScope,
                               llvm::ArrayRef<const Attr *> attrs = {});

  mlir::LogicalResult emitSimpleStmt(const clang::Stmt *s,
                                     bool useCurrentScope);

  mlir::LogicalResult emitForStmt(const clang::ForStmt &s);

  /// Emit the computation of the specified expression of complex type,
  /// returning the result.
  mlir::Value emitComplexExpr(const Expr *e);

  void emitComplexExprIntoLValue(const Expr *e, LValue dest, bool isInit);

  mlir::Value emitComplexPrePostIncDec(const UnaryOperator *e, LValue lv,
                                       cir::UnaryOpKind op, bool isPre);

  LValue emitComplexAssignmentLValue(const BinaryOperator *e);
  LValue emitComplexCompoundAssignmentLValue(const CompoundAssignOperator *e);
  LValue emitScalarCompoundAssignWithComplex(const CompoundAssignOperator *e,
                                             mlir::Value &result);

  void emitCompoundStmt(const clang::CompoundStmt &s);

  void emitCompoundStmtWithoutScope(const clang::CompoundStmt &s);

  void emitDecl(const clang::Decl &d, bool evaluateConditionDecl = false);
  mlir::LogicalResult emitDeclStmt(const clang::DeclStmt &s);
  LValue emitDeclRefLValue(const clang::DeclRefExpr *e);

  mlir::LogicalResult emitDefaultStmt(const clang::DefaultStmt &s,
                                      mlir::Type condType,
                                      bool buildingTopLevelCase);

  void emitDelegateCXXConstructorCall(const clang::CXXConstructorDecl *ctor,
                                      clang::CXXCtorType ctorType,
                                      const FunctionArgList &args,
                                      clang::SourceLocation loc);

  /// We are performing a delegate call; that is, the current function is
  /// delegating to another one. Produce a r-value suitable for passing the
  /// given parameter.
  void emitDelegateCallArg(CallArgList &args, const clang::VarDecl *param,
                           clang::SourceLocation loc);

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

  mlir::LogicalResult emitLabel(const clang::LabelDecl &d);
  mlir::LogicalResult emitLabelStmt(const clang::LabelStmt &s);

  mlir::LogicalResult emitIfStmt(const clang::IfStmt &s);

  /// Emit code to compute the specified expression,
  /// ignoring the result.
  void emitIgnoredExpr(const clang::Expr *e);

  RValue emitLoadOfBitfieldLValue(LValue lv, SourceLocation loc);

  /// Load a complex number from the specified l-value.
  mlir::Value emitLoadOfComplex(LValue src, SourceLocation loc);

  /// Given an expression that represents a value lvalue, this method emits
  /// the address of the lvalue, then loads the result as an rvalue,
  /// returning the rvalue.
  RValue emitLoadOfLValue(LValue lv, SourceLocation loc);

  Address emitLoadOfReference(LValue refLVal, mlir::Location loc,
                              LValueBaseInfo *pointeeBaseInfo);
  LValue emitLoadOfReferenceLValue(Address refAddr, mlir::Location loc,
                                   QualType refTy, AlignmentSource source);

  /// EmitLoadOfScalar - Load a scalar value from an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.  The l-value must be a simple
  /// l-value.
  mlir::Value emitLoadOfScalar(LValue lvalue, SourceLocation loc);

  /// Emit code to compute a designator that specifies the location
  /// of the expression.
  /// FIXME: document this function better.
  LValue emitLValue(const clang::Expr *e);
  LValue emitLValueForBitField(LValue base, const FieldDecl *field);
  LValue emitLValueForField(LValue base, const clang::FieldDecl *field);

  /// Like emitLValueForField, excpet that if the Field is a reference, this
  /// will return the address of the reference and not the address of the value
  /// stored in the reference.
  LValue emitLValueForFieldInitialization(LValue base,
                                          const clang::FieldDecl *field,
                                          llvm::StringRef fieldName);

  LValue emitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *e);

  LValue emitMemberExpr(const MemberExpr *e);

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
                                   LValueBaseInfo *baseInfo = nullptr);

  /// Emits a reference binding to the passed in expression.
  RValue emitReferenceBindingToExpr(const Expr *e);

  mlir::LogicalResult emitReturnStmt(const clang::ReturnStmt &s);

  RValue emitRotate(const CallExpr *e, bool isRotateLeft);

  mlir::Value emitScalarConstant(const ConstantEmission &constant, Expr *e);

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  mlir::Value emitScalarConversion(mlir::Value src, clang::QualType srcType,
                                   clang::QualType dstType,
                                   clang::SourceLocation loc);

  void emitScalarInit(const clang::Expr *init, mlir::Location loc,
                      LValue lvalue, bool capturedByInit = false);

  void emitStaticVarDecl(const VarDecl &d, cir::GlobalLinkageKind linkage);

  void emitStoreOfComplex(mlir::Location loc, mlir::Value v, LValue dest,
                          bool isInit);

  void emitStoreOfScalar(mlir::Value value, Address addr, bool isVolatile,
                         clang::QualType ty, bool isInit = false,
                         bool isNontemporal = false);
  void emitStoreOfScalar(mlir::Value value, LValue lvalue, bool isInit);

  /// Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void emitStoreThroughLValue(RValue src, LValue dst, bool isInit = false);

  mlir::Value emitStoreThroughBitfieldLValue(RValue src, LValue dstresult);

  LValue emitStringLiteralLValue(const StringLiteral *e);

  mlir::LogicalResult emitSwitchBody(const clang::Stmt *s);
  mlir::LogicalResult emitSwitchCase(const clang::SwitchCase &s,
                                     bool buildingTopLevelCase);
  mlir::LogicalResult emitSwitchStmt(const clang::SwitchStmt &s);

  /// Given a value and its clang type, returns the value casted to its memory
  /// representation.
  /// Note: CIR defers most of the special casting to the final lowering passes
  /// to conserve the high level information.
  mlir::Value emitToMemory(mlir::Value value, clang::QualType ty);

  /// Emit a trap instruction, which is used to abort the program in an abnormal
  /// way, usually for debugging purposes.
  /// \p createNewBlock indicates whether to create a new block for the IR
  /// builder. Since the `cir.trap` operation is a terminator, operations that
  /// follow a trap cannot be emitted after `cir.trap` in the same block. To
  /// ensure these operations get emitted successfully, you need to create a new
  /// dummy block and set the insertion point there before continuing from the
  /// trap operation.
  void emitTrap(mlir::Location loc, bool createNewBlock);

  LValue emitUnaryOpLValue(const clang::UnaryOperator *e);

  /// Emit a reached-unreachable diagnostic if \p loc is valid and runtime
  /// checking is enabled. Otherwise, just emit an unreachable instruction.
  /// \p createNewBlock indicates whether to create a new block for the IR
  /// builder. Since the `cir.unreachable` operation is a terminator, operations
  /// that follow an unreachable point cannot be emitted after `cir.unreachable`
  /// in the same block. To ensure these operations get emitted successfully,
  /// you need to create a dummy block and set the insertion point there before
  /// continuing from the unreachable point.
  void emitUnreachable(clang::SourceLocation loc, bool createNewBlock);

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void emitVarDecl(const clang::VarDecl &d);

  void emitVariablyModifiedType(QualType ty);

  mlir::LogicalResult emitWhileStmt(const clang::WhileStmt &s);

  /// Given an assignment `*lhs = rhs`, emit a test that checks if \p rhs is
  /// nonnull, if 1\p LHS is marked _Nonnull.
  void emitNullabilityCheck(LValue lhs, mlir::Value rhs,
                            clang::SourceLocation loc);

  /// An object to manage conditionally-evaluated expressions.
  class ConditionalEvaluation {
    CIRGenFunction &cgf;
    mlir::OpBuilder::InsertPoint insertPt;

  public:
    ConditionalEvaluation(CIRGenFunction &cgf)
        : cgf(cgf), insertPt(cgf.builder.saveInsertionPoint()) {}
    ConditionalEvaluation(CIRGenFunction &cgf, mlir::OpBuilder::InsertPoint ip)
        : cgf(cgf), insertPt(ip) {}

    void beginEvaluation() {
      assert(cgf.outermostConditional != this);
      if (!cgf.outermostConditional)
        cgf.outermostConditional = this;
    }

    void endEvaluation() {
      assert(cgf.outermostConditional != nullptr);
      if (cgf.outermostConditional == this)
        cgf.outermostConditional = nullptr;
    }

    /// Returns the insertion point which will be executed prior to each
    /// evaluation of the conditional code. In LLVM OG, this method
    /// is called getStartingBlock.
    mlir::OpBuilder::InsertPoint getInsertPoint() const { return insertPt; }
  };

  struct ConditionalInfo {
    std::optional<LValue> lhs{}, rhs{};
    mlir::Value result{};
  };

  // Return true if we're currently emitting one branch or the other of a
  // conditional expression.
  bool isInConditionalBranch() const { return outermostConditional != nullptr; }

  void setBeforeOutermostConditional(mlir::Value value, Address addr) {
    assert(isInConditionalBranch());
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.restoreInsertionPoint(outermostConditional->getInsertPoint());
      builder.createStore(
          value.getLoc(), value, addr,
          mlir::IntegerAttr::get(
              mlir::IntegerType::get(value.getContext(), 64),
              (uint64_t)addr.getAlignment().getAsAlign().value()));
    }
  }

  // Points to the outermost active conditional control. This is used so that
  // we know if a temporary should be destroyed conditionally.
  ConditionalEvaluation *outermostConditional = nullptr;

  template <typename FuncTy>
  ConditionalInfo emitConditionalBlocks(const AbstractConditionalOperator *e,
                                        const FuncTy &branchGenFunc);

  mlir::Value emitTernaryOnBoolExpr(const clang::Expr *cond, mlir::Location loc,
                                    const clang::Stmt *thenS,
                                    const clang::Stmt *elseS);

  /// ----------------------
  /// CIR build helpers
  /// -----------------
public:
  cir::AllocaOp createTempAlloca(mlir::Type ty, mlir::Location loc,
                                 const Twine &name = "tmp",
                                 mlir::Value arraySize = nullptr,
                                 bool insertIntoFnEntryBlock = false);
  cir::AllocaOp createTempAlloca(mlir::Type ty, mlir::Location loc,
                                 const Twine &name = "tmp",
                                 mlir::OpBuilder::InsertPoint ip = {},
                                 mlir::Value arraySize = nullptr);
  Address createTempAlloca(mlir::Type ty, CharUnits align, mlir::Location loc,
                           const Twine &name = "tmp",
                           mlir::Value arraySize = nullptr,
                           Address *alloca = nullptr,
                           mlir::OpBuilder::InsertPoint ip = {});
  Address createTempAllocaWithoutCast(mlir::Type ty, CharUnits align,
                                      mlir::Location loc,
                                      const Twine &name = "tmp",
                                      mlir::Value arraySize = nullptr,
                                      mlir::OpBuilder::InsertPoint ip = {});

  /// Create a temporary memory object of the given type, with
  /// appropriate alignmen and cast it to the default address space. Returns
  /// the original alloca instruction by \p Alloca if it is not nullptr.
  Address createMemTemp(QualType t, mlir::Location loc,
                        const Twine &name = "tmp", Address *alloca = nullptr,
                        mlir::OpBuilder::InsertPoint ip = {});
  Address createMemTemp(QualType t, CharUnits align, mlir::Location loc,
                        const Twine &name = "tmp", Address *alloca = nullptr,
                        mlir::OpBuilder::InsertPoint ip = {});

  //===--------------------------------------------------------------------===//
  //                         OpenACC Emission
  //===--------------------------------------------------------------------===//
private:
  template <typename Op>
  Op emitOpenACCOp(mlir::Location start, OpenACCDirectiveKind dirKind,
                   SourceLocation dirLoc,
                   llvm::ArrayRef<const OpenACCClause *> clauses);
  // Function to do the basic implementation of an operation with an Associated
  // Statement.  Models AssociatedStmtConstruct.
  template <typename Op, typename TermOp>
  mlir::LogicalResult emitOpenACCOpAssociatedStmt(
      mlir::Location start, mlir::Location end, OpenACCDirectiveKind dirKind,
      SourceLocation dirLoc, llvm::ArrayRef<const OpenACCClause *> clauses,
      const Stmt *associatedStmt);

  template <typename Op, typename TermOp>
  mlir::LogicalResult emitOpenACCOpCombinedConstruct(
      mlir::Location start, mlir::Location end, OpenACCDirectiveKind dirKind,
      SourceLocation dirLoc, llvm::ArrayRef<const OpenACCClause *> clauses,
      const Stmt *loopStmt);

  template <typename Op>
  void emitOpenACCClauses(Op &op, OpenACCDirectiveKind dirKind,
                          SourceLocation dirLoc,
                          ArrayRef<const OpenACCClause *> clauses);
  // The second template argument doesn't need to be a template, since it should
  // always be an mlir::acc::LoopOp, but as this is a template anyway, we make
  // it a template argument as this way we can avoid including the OpenACC MLIR
  // headers here. We will count on linker failures/explicit instantiation to
  // ensure we don't mess this up, but it is only called from 1 place, and
  // instantiated 3x.
  template <typename ComputeOp, typename LoopOp>
  void emitOpenACCClauses(ComputeOp &op, LoopOp &loopOp,
                          OpenACCDirectiveKind dirKind, SourceLocation dirLoc,
                          ArrayRef<const OpenACCClause *> clauses);

  // The OpenACC LoopOp requires that we have auto, seq, or independent on all
  // LoopOp operations for the 'none' device type case. This function checks if
  // the LoopOp has one, else it updates it to have one.
  void updateLoopOpParallelism(mlir::acc::LoopOp &op, bool isOrphan,
                               OpenACCDirectiveKind dk);

  // The OpenACC 'cache' construct actually applies to the 'loop' if present. So
  // keep track of the 'loop' so that we can add the cache vars to it correctly.
  mlir::acc::LoopOp *activeLoopOp = nullptr;

  struct ActiveOpenACCLoopRAII {
    CIRGenFunction &cgf;
    mlir::acc::LoopOp *oldLoopOp;

    ActiveOpenACCLoopRAII(CIRGenFunction &cgf, mlir::acc::LoopOp *newOp)
        : cgf(cgf), oldLoopOp(cgf.activeLoopOp) {
      cgf.activeLoopOp = newOp;
    }
    ~ActiveOpenACCLoopRAII() { cgf.activeLoopOp = oldLoopOp; }
  };

public:
  // Helper type used to store the list of important information for a 'data'
  // clause variable, or a 'cache' variable reference.
  struct OpenACCDataOperandInfo {
    mlir::Location beginLoc;
    mlir::Value varValue;
    std::string name;
    QualType baseType;
    llvm::SmallVector<mlir::Value> bounds;
  };
  // Gets the collection of info required to lower and OpenACC clause or cache
  // construct variable reference.
  OpenACCDataOperandInfo getOpenACCDataOperandInfo(const Expr *e);
  // Helper function to emit the integer expressions as required by an OpenACC
  // clause/construct.
  mlir::Value emitOpenACCIntExpr(const Expr *intExpr);
  // Helper function to emit an integer constant as an mlir int type, used for
  // constants in OpenACC constructs/clauses.
  mlir::Value createOpenACCConstantInt(mlir::Location loc, unsigned width,
                                       int64_t value);

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

  /// Create a temporary memory object for the given aggregate type.
  AggValueSlot createAggTemp(QualType ty, mlir::Location loc,
                             const Twine &name = "tmp",
                             Address *alloca = nullptr) {
    assert(!cir::MissingFeatures::aggValueSlot());
    return AggValueSlot::forAddr(
        createMemTemp(ty, loc, name, alloca), ty.getQualifiers(),
        AggValueSlot::IsNotDestructed, AggValueSlot::IsNotAliased,
        AggValueSlot::DoesNotOverlap);
  }

private:
  QualType getVarArgType(const Expr *arg);
};

} // namespace clang::CIRGen

#endif
