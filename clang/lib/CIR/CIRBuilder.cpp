//===- CIRBuilder.cpp - MLIR Generation from a Toy AST --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "CIRGenTypes.h"

#include "clang/CIR/CIRBuilder.h"
#include "clang/CIR/CIRCodeGenFunction.h"

#include "mlir/Dialect/CIR/IR/CIRAttrs.h"
#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/CIR/IR/CIRTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace mlir::cir;
using namespace cir;
using namespace clang;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

CIRCodeGenFunction::CIRCodeGenFunction() = default;
TypeEvaluationKind CIRCodeGenFunction::getEvaluationKind(QualType type) {
  type = type.getCanonicalType();
  while (true) {
    switch (type->getTypeClass()) {
#define TYPE(name, parent)
#define ABSTRACT_TYPE(name, parent)
#define NON_CANONICAL_TYPE(name, parent) case Type::name:
#define DEPENDENT_TYPE(name, parent) case Type::name:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(name, parent) case Type::name:
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable("non-canonical or dependent type in IR-generation");

    case Type::Auto:
    case Type::DeducedTemplateSpecialization:
      llvm_unreachable("undeduced type in IR-generation");

    case Type::ArrayParameter:
      llvm_unreachable("NYI");

    // Various scalar types.
    case Type::Builtin:
    case Type::Pointer:
    case Type::BlockPointer:
    case Type::LValueReference:
    case Type::RValueReference:
    case Type::MemberPointer:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::FunctionProto:
    case Type::FunctionNoProto:
    case Type::Enum:
    case Type::ObjCObjectPointer:
    case Type::Pipe:
    case Type::BitInt:
      return TEK_Scalar;

    // Complexes.
    case Type::Complex:
      return TEK_Complex;

    // Arrays, records, and Objective-C objects.
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::Record:
    case Type::ObjCObject:
    case Type::ObjCInterface:
      return TEK_Aggregate;

    // We operate on atomic values according to their underlying type.
    case Type::Atomic:
      type = cast<AtomicType>(type)->getValueType();
      continue;
    }
    llvm_unreachable("unknown type kind!");
  }
}

namespace cir {

/// Implementation of a CIR/MLIR emission from Clang AST.
///
/// This will emit operations that are specific to C(++)/ObjC(++) language,
/// preserving the semantics of the language and (hopefully) allow to perform
/// accurate analysis and transformation based on these high level semantics.
class CIRBuildImpl {
public:
  CIRBuildImpl(mlir::MLIRContext &context, clang::ASTContext &astctx)
      : builder(&context), astCtx(astctx) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    genTypes = std::make_unique<CIRGenTypes>(astCtx, this->getBuilder());
  }
  CIRBuildImpl(CIRBuildImpl &) = delete;
  CIRBuildImpl &operator=(CIRBuildImpl &) = delete;
  ~CIRBuildImpl() = default;

  using SymTableTy = llvm::ScopedHashTable<const Decl *, mlir::Value>;
  using SymTableScopeTy = ScopedHashTableScope<const Decl *, mlir::Value>;

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
  CIRCodeGenFunction *CurCCGF = nullptr;

  /// Per-module type mapping from clang AST to CIR.
  std::unique_ptr<CIRGenTypes> genTypes;

  /// Helper conversion from Clang source location to an MLIR location.
  mlir::Location getLoc(SourceLocation SLoc) {
    const SourceManager &SM = astCtx.getSourceManager();
    PresumedLoc PLoc = SM.getPresumedLoc(SLoc);
    StringRef Filename = PLoc.getFilename();
    return mlir::FileLineColLoc::get(builder.getStringAttr(Filename),
                                     PLoc.getLine(), PLoc.getColumn());
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(const Decl *var, QualType T, mlir::Location loc,
                              mlir::Value &addr, bool IsParam = false) {
    const auto *namedVar = dyn_cast_or_null<NamedDecl>(var);
    assert(namedVar && "Needs a named decl");

    if (symbolTable.count(var))
      return mlir::failure();

    auto localVarTy = getCIRType(T);
    auto localVarPtrTy =
        mlir::cir::PointerType::get(builder.getContext(), localVarTy);

    auto localVarAddr = builder.create<mlir::cir::AllocaOp>(
        loc, /*addr type*/ localVarPtrTy, /*var type*/ localVarTy,
        IsParam ? InitStyle::paraminit : InitStyle::uninitialized);

    auto *parentBlock = localVarAddr->getBlock();
    localVarAddr->moveBefore(&parentBlock->front());

    // Insert into the symbol table, allocate some stack space in the
    // function entry block.
    symbolTable.insert(var, localVarAddr);
    addr = localVarAddr;

    return mlir::success();
  }

public:
  mlir::ModuleOp getModule() { return theModule; }
  mlir::OpBuilder &getBuilder() { return builder; }

  class RawAddress {
    mlir::Value Pointer;
    CharUnits Alignment;

  public:
    RawAddress(mlir::Value pointer, CharUnits alignment)
        : Pointer(pointer), Alignment(alignment) {
      assert((!alignment.isZero() || pointer == nullptr) &&
             "creating valid address with invalid alignment");
    }

    static RawAddress invalid() { return RawAddress(nullptr, CharUnits()); }
    bool isValid() const { return Pointer != nullptr; }

    mlir::Value getPointer() const {
      // assert(isValid());
      return Pointer;
    }

    /// Return the alignment of this pointer.
    CharUnits getAlignment() const {
      // assert(isValid());
      return Alignment;
    }
  };

  class LValue {
    enum {
      Simple,       // This is a normal l-value, use getAddress().
      VectorElt,    // This is a vector element l-value (V[i]), use getVector*
      BitField,     // This is a bitfield l-value, use getBitfield*.
      ExtVectorElt, // This is an extended vector subset, use getExtVectorComp
      GlobalReg,    // This is a register l-value, use getGlobalReg()
      MatrixElt     // This is a matrix element, use getVector*
    } LVType;
    QualType Type;

  private:
    void Initialize(CharUnits Alignment, QualType Type,
                    LValueBaseInfo BaseInfo) {
      // assert((!Alignment.isZero()) && // || Type->isIncompleteType()) &&
      //       "initializing l-value with zero alignment!");
      this->Type = Type;
      // This flag shows if a nontemporal load/stores should be used when
      // accessing this lvalue.
      const unsigned MaxAlign = 1U << 31;
      this->Alignment = Alignment.getQuantity() <= MaxAlign
                            ? Alignment.getQuantity()
                            : MaxAlign;
      assert(this->Alignment == Alignment.getQuantity() &&
             "Alignment exceeds allowed max!");
      this->BaseInfo = BaseInfo;
    }

    // The alignment to use when accessing this lvalue. (For vector elements,
    // this is the alignment of the whole vector)
    unsigned Alignment;
    mlir::Value V;
    LValueBaseInfo BaseInfo;

  public:
    bool isSimple() const { return LVType == Simple; }
    bool isVectorElt() const { return LVType == VectorElt; }
    bool isBitField() const { return LVType == BitField; }
    bool isExtVectorElt() const { return LVType == ExtVectorElt; }
    bool isGlobalReg() const { return LVType == GlobalReg; }
    bool isMatrixElt() const { return LVType == MatrixElt; }

    QualType getType() const { return Type; }

    mlir::Value getPointer() const { return V; }

    CharUnits getAlignment() const {
      return CharUnits::fromQuantity(Alignment);
    }

    RawAddress getAddress() const {
      return RawAddress(getPointer(), getAlignment());
    }

    LValueBaseInfo getBaseInfo() const { return BaseInfo; }
    void setBaseInfo(LValueBaseInfo Info) { BaseInfo = Info; }

    static LValue makeAddr(RawAddress address, QualType T,
                           AlignmentSource Source = AlignmentSource::Type) {
      LValue R;
      R.V = address.getPointer();
      R.Initialize(address.getAlignment(), T, LValueBaseInfo(Source));
      R.LVType = Simple;
      return R;
    }
  };

  /// This trivial value class is used to represent the result of an
  /// expression that is evaluated.  It can be one of three things: either a
  /// simple MLIR SSA value, a pair of SSA values for complex numbers, or the
  /// address of an aggregate value in memory.
  class RValue {
    enum Flavor { Scalar, Complex, Aggregate };

    // The shift to make to an aggregate's alignment to make it look
    // like a pointer.
    enum { AggAlignShift = 4 };

    // Stores first value and flavor.
    llvm::PointerIntPair<mlir::Value, 2, Flavor> V1;
    // Stores second value and volatility.
    llvm::PointerIntPair<mlir::Value, 1, bool> V2;

  public:
    bool isScalar() const { return V1.getInt() == Scalar; }
    bool isComplex() const { return V1.getInt() == Complex; }
    bool isAggregate() const { return V1.getInt() == Aggregate; }

    bool isVolatileQualified() const { return V2.getInt(); }

    /// getScalarVal() - Return the Value* of this scalar value.
    mlir::Value getScalarVal() const {
      assert(isScalar() && "Not a scalar!");
      return V1.getPointer();
    }

    /// getComplexVal - Return the real/imag components of this complex value.
    ///
    std::pair<mlir::Value, mlir::Value> getComplexVal() const {
      assert(0 && "not implemented");
      return {};
    }

    /// getAggregateAddr() - Return the Value* of the address of the
    /// aggregate.
    RawAddress getAggregateAddress() const {
      assert(0 && "not implemented");
      return RawAddress::invalid();
    }

    static RValue getIgnored() {
      // FIXME: should we make this a more explicit state?
      return get(nullptr);
    }

    static RValue get(mlir::Value V) {
      RValue ER;
      ER.V1.setPointer(V);
      ER.V1.setInt(Scalar);
      ER.V2.setInt(false);
      return ER;
    }
    static RValue getComplex(mlir::Value V1, mlir::Value V2) {
      assert(0 && "not implemented");
      return RValue{};
    }
    static RValue getComplex(const std::pair<mlir::Value, mlir::Value> &C) {
      assert(0 && "not implemented");
      return RValue{};
    }
    // FIXME: Aggregate rvalues need to retain information about whether they
    // are volatile or not.  Remove default to find all places that probably
    // get this wrong.
    static RValue getAggregate(RawAddress addr, bool isVolatile = false) {
      assert(0 && "not implemented");
      return RValue{};
    }
  };
  class ScalarExprEmitter : public StmtVisitor<ScalarExprEmitter, mlir::Value> {
    LLVM_ATTRIBUTE_UNUSED CIRCodeGenFunction &CGF;
    CIRBuildImpl &Builder;

  public:
    ScalarExprEmitter(CIRCodeGenFunction &cgf, CIRBuildImpl &builder)
        : CGF(cgf), Builder(builder) {
      (void)CGF;
    }

    mlir::Value Visit(Expr *E) {
      return StmtVisitor<ScalarExprEmitter, mlir::Value>::Visit(E);
    }

    LValue EmitDeclRefLValue(const DeclRefExpr *E) {
      const NamedDecl *ND = E->getDecl();

      assert(E->isNonOdrUse() != NOUR_Unevaluated &&
             "should not emit an unevaluated operand");

      if (const auto *VD = dyn_cast<VarDecl>(ND)) {
        // Global Named registers access via intrinsics only
        assert(VD->getStorageClass() != SC_Register && "not implemented");
        assert(E->isNonOdrUse() != NOUR_Constant && "not implemented");
        assert(!E->refersToEnclosingVariableOrCapture() && "not implemented");
        assert(!(VD->hasLinkage() || VD->isStaticDataMember()) &&
               "not implemented");
        assert(!VD->isEscapingByref() && "not implemented");
        assert(!VD->getType()->isReferenceType() && "not implemented");
        assert(Builder.symbolTable.count(VD) && "should be already mapped");

        mlir::Value V = Builder.symbolTable.lookup(VD);
        assert(V && "Name lookup must succeed");

        LValue LV = LValue::makeAddr(RawAddress(V, CharUnits::fromQuantity(4)),
                                     VD->getType(), AlignmentSource::Decl);
        return LV;
      }

      llvm_unreachable("Unhandled DeclRefExpr?");
    }

    LValue EmitLValue(const Expr *E) {
      switch (E->getStmtClass()) {
      case Expr::DeclRefExprClass:
        return EmitDeclRefLValue(cast<DeclRefExpr>(E));
      default:
        emitError(Builder.getLoc(E->getExprLoc()),
                  "l-value not implemented for '")
            << E->getStmtClassName() << "'";
        break;
      }
      return LValue::makeAddr(RawAddress::invalid(), E->getType());
    }

    /// Emits the address of the l-value, then loads and returns the result.
    mlir::Value buildLoadOfLValue(const Expr *E) {
      LValue LV = EmitLValue(E);
      auto load = Builder.builder.create<mlir::cir::LoadOp>(
          Builder.getLoc(E->getExprLoc()), Builder.getCIRType(E->getType()),
          LV.getPointer(), mlir::UnitAttr::get(Builder.builder.getContext()));
      // FIXME: add some akin to EmitLValueAlignmentAssumption(E, V);
      return load;
    }

    // Handle l-values.
    mlir::Value VisitDeclRefExpr(DeclRefExpr *E) {
      // FIXME: we could try to emit this as constant first, see
      // CGF.tryEmitAsConstant(E)
      return buildLoadOfLValue(E);
    }

    // Emit code for an explicit or implicit cast.  Implicit
    // casts have to handle a more broad range of conversions than explicit
    // casts, as they handle things like function to ptr-to-function decay
    // etc.
    mlir::Value VisitCastExpr(CastExpr *CE) {
      Expr *E = CE->getSubExpr();
      QualType DestTy = CE->getType();
      CastKind Kind = CE->getCastKind();
      switch (Kind) {
      case CK_LValueToRValue:
        assert(Builder.astCtx.hasSameUnqualifiedType(E->getType(), DestTy));
        assert(E->isGLValue() && "lvalue-to-rvalue applied to r-value!");
        return Visit(const_cast<Expr *>(E));
      case CK_NullToPointer: {
        // FIXME: use MustVisitNullValue(E) and evaluate expr.
        // Note that DestTy is used as the MLIR type instead of a custom
        // nullptr type.
        mlir::Type Ty = Builder.getCIRType(DestTy);
        return Builder.builder.create<mlir::cir::ConstantOp>(
            Builder.getLoc(E->getExprLoc()), Ty,
            mlir::cir::NullAttr::get(Builder.builder.getContext(), Ty));
      }
      default:
        emitError(Builder.getLoc(CE->getExprLoc()),
                  "cast kind not implemented: '")
            << CE->getCastKindName() << "'";
        return nullptr;
      }
    }

    mlir::Value VisitExpr(Expr *E) {
      emitError(Builder.getLoc(E->getExprLoc()), "scalar exp no implemented: '")
          << E->getStmtClassName() << "'";
      if (E->getType()->isVoidType())
        return nullptr;
      // FIXME: find a way to return "undef"...
      // return llvm::UndefValue::get(CGF.ConvertType(E->getType()));
      return nullptr;
    }

    // Leaves.
    mlir::Value VisitIntegerLiteral(const IntegerLiteral *E) {
      mlir::Type Ty = Builder.getCIRType(E->getType());
      return Builder.builder.create<mlir::cir::ConstantOp>(
          Builder.getLoc(E->getExprLoc()), Ty,
          Builder.builder.getIntegerAttr(Ty, E->getValue()));
    }
  };

  struct AutoVarEmission {
    const VarDecl *Variable;
    /// The address of the alloca for languages with explicit address space
    /// (e.g. OpenCL) or alloca casted to generic pointer for address space
    /// agnostic languages (e.g. C++). Invalid if the variable was emitted
    /// as a global constant.
    RawAddress Addr;

    /// True if the variable is of aggregate type and has a constant
    /// initializer.
    bool IsConstantAggregate;

    struct Invalid {};
    AutoVarEmission(Invalid) : Variable(nullptr), Addr(RawAddress::invalid()) {}

    AutoVarEmission(const VarDecl &variable)
        : Variable(&variable), Addr(RawAddress::invalid()),
          IsConstantAggregate(false) {}

    static AutoVarEmission invalid() { return AutoVarEmission(Invalid()); }
    /// Returns the raw, allocated address, which is not necessarily
    /// the address of the object itself. It is casted to default
    /// address space for address space agnostic languages.
    RawAddress getAllocatedAddress() const { return Addr; }
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
  bool isTypeConstant(QualType Ty, bool ExcludeCtor) {
    if (!Ty.isConstant(astCtx) && !Ty->isReferenceType())
      return false;

    if (astCtx.getLangOpts().CPlusPlus) {
      if (const CXXRecordDecl *Record =
              astCtx.getBaseElementType(Ty)->getAsCXXRecordDecl())
        return ExcludeCtor && !Record->hasMutableFields() &&
               Record->hasTrivialDestructor();
    }

    return true;
  }

  /// Emit the alloca and debug information for a
  /// local variable.  Does not emit initialization or destruction.
  AutoVarEmission buildAutoVarAlloca(const VarDecl &D) {
    QualType Ty = D.getType();
    // TODO: (|| Ty.getAddressSpace() == LangAS::opencl_private &&
    //        getLangOpts().OpenCL))
    assert(Ty.getAddressSpace() == LangAS::Default);

    assert(!D.isEscapingByref() && "not implemented");
    assert(!Ty->isVariablyModifiedType() && "not implemented");
    assert(!astCtx.getLangOpts().OpenMP && // !CGM.getLangOpts().OpenMPIRBuilder
           "not implemented");
    bool NRVO = astCtx.getLangOpts().ElideConstructors && D.isNRVOVariable();
    assert(!NRVO && "not implemented");
    assert(Ty->isConstantSizeType() && "not implemented");
    assert(!D.hasAttr<AnnotateAttr>() && "not implemented");

    AutoVarEmission emission(D);
    CharUnits alignment = astCtx.getDeclAlign(&D);
    // TODO: debug info
    // TODO: use CXXABI

    // If this value is an array or struct with a statically determinable
    // constant initializer, there are optimizations we can do.
    //
    // TODO: We should constant-evaluate the initializer of any variable,
    // as long as it is initialized by a constant expression. Currently,
    // isConstantInitializer produces wrong answers for structs with
    // reference or bitfield members, and a few other cases, and checking
    // for POD-ness protects us from some of these.
    if (D.getInit() && (Ty->isArrayType() || Ty->isRecordType()) &&
        (D.isConstexpr() ||
         ((Ty.isPODType(astCtx) ||
           astCtx.getBaseElementType(Ty)->isObjCObjectPointerType()) &&
          D.getInit()->isConstantInitializer(astCtx, false)))) {

      // If the variable's a const type, and it's neither an NRVO
      // candidate nor a __block variable and has no mutable members,
      // emit it as a global instead.
      // Exception is if a variable is located in non-constant address space
      // in OpenCL.
      // TODO: deal with CGM.getCodeGenOpts().MergeAllConstants
      // TODO: perhaps we don't need this at all at CIR since this can
      // be done as part of lowering down to LLVM.
      if ((!astCtx.getLangOpts().OpenCL ||
           Ty.getAddressSpace() == LangAS::opencl_constant) &&
          (!NRVO && !D.isEscapingByref() && isTypeConstant(Ty, true)))
        assert(0 && "not implemented");

      // Otherwise, tell the initialization code that we're in this case.
      emission.IsConstantAggregate = true;
    }

    // TODO: track source location range...
    mlir::Value addr;
    if (failed(declare(&D, Ty, getLoc(D.getSourceRange().getBegin()), addr))) {
      theModule.emitError("Cannot declare variable");
      return emission;
    }

    // TODO: what about emitting lifetime markers for MSVC catch parameters?
    // TODO: something like @llvm.lifetime.start/end here? revisit this later.
    emission.Addr = RawAddress{addr, alignment};
    return emission;
  }

  /// Determine whether the given initializer is trivial in the sense
  /// that it requires no code to be generated.
  bool isTrivialInitializer(const Expr *Init) {
    if (!Init)
      return true;

    if (const CXXConstructExpr *Construct = dyn_cast<CXXConstructExpr>(Init))
      if (CXXConstructorDecl *Constructor = Construct->getConstructor())
        if (Constructor->isTrivial() && Constructor->isDefaultConstructor() &&
            !Construct->requiresZeroInitialization())
          return true;

    return false;
  }

  // TODO: this can also be abstrated into common AST helpers
  bool hasBooleanRepresentation(QualType Ty) {
    if (Ty->isBooleanType())
      return true;

    if (const EnumType *ET = Ty->getAs<EnumType>())
      return ET->getDecl()->getIntegerType()->isBooleanType();

    if (const AtomicType *AT = Ty->getAs<AtomicType>())
      return hasBooleanRepresentation(AT->getValueType());

    return false;
  }

  mlir::Value buildToMemory(mlir::Value Value, QualType Ty) {
    // Bool has a different representation in memory than in registers.
    if (hasBooleanRepresentation(Ty))
      assert(0 && "not implemented");
    return Value;
  }

  void buildStoreOfScalar(mlir::Value value, LValue lvalue, const Decl *D,
                          bool isInit) {
    // TODO: constant matrix type, volatile, non temporal, TBAA
    buildStoreOfScalar(value, lvalue.getAddress(), false, lvalue.getType(),
                       lvalue.getBaseInfo(), D, isInit, false);
  }

  void buildStoreOfScalar(mlir::Value Value, RawAddress Addr, bool Volatile,
                          QualType Ty, LValueBaseInfo BaseInfo, const Decl *D,
                          bool isInit, bool isNontemporal) {
    // TODO: PreserveVec3Type
    // TODO: LValueIsSuitableForInlineAtomic ?
    // TODO: TBAA
    Value = buildToMemory(Value, Ty);
    if (Ty->isAtomicType() || isNontemporal) {
      assert(0 && "not implemented");
    }

    // Update the alloca with more info on initialization.
    auto SrcAlloca = dyn_cast_or_null<mlir::cir::AllocaOp>(
        Addr.getPointer().getDefiningOp());
    if (isInit) {
      InitStyle IS;
      const VarDecl *VD = dyn_cast_or_null<VarDecl>(D);
      assert(VD && "VarDecl expected");
      if (VD->hasInit()) {
        switch (VD->getInitStyle()) {
        case VarDecl::ParenListInit:
          llvm_unreachable("NYI");
        case VarDecl::CInit:
          IS = InitStyle::cinit;
          break;
        case VarDecl::CallInit:
          IS = InitStyle::callinit;
          break;
        case VarDecl::ListInit:
          IS = InitStyle::listinit;
          break;
        }
        SrcAlloca.setInitAttr(InitStyleAttr::get(builder.getContext(), IS));
      }
    }
    assert(SrcAlloca && "find a better way to retrieve source location");
    builder.create<mlir::cir::StoreOp>(SrcAlloca.getLoc(), Value,
                                       Addr.getPointer());
  }

  /// Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void buldStoreThroughLValue(RValue Src, LValue Dst, const Decl *D,
                              bool isInit) {
    assert(Dst.isSimple() && "only implemented simple");
    // TODO: ObjC lifetime.
    assert(Src.isScalar() && "Can't emit an agg store with this method");
    buildStoreOfScalar(Src.getScalarVal(), Dst, D, isInit);
  }

  void buildScalarInit(const Expr *init, const ValueDecl *D, LValue lvalue) {
    // TODO: this is where a lot of ObjC lifetime stuff would be done.
    mlir::Value value = buildScalarExpr(init);
    buldStoreThroughLValue(RValue::get(value), lvalue, D, true);
    return;
  }

  /// Emit an expression as an initializer for an object (variable, field, etc.)
  /// at the given location.  The expression is not necessarily the normal
  /// initializer for the object, and the address is not necessarily
  /// its normal location.
  ///
  /// \param init the initializing expression
  /// \param D the object to act as if we're initializing
  /// \param lvalue the lvalue to initialize
  void buildExprAsInit(const Expr *init, const ValueDecl *D, LValue lvalue) {
    QualType type = D->getType();

    if (type->isReferenceType()) {
      assert(0 && "not implemented");
      return;
    }
    switch (CIRCodeGenFunction::getEvaluationKind(type)) {
    case TEK_Scalar:
      buildScalarInit(init, D, lvalue);
      return;
    case TEK_Complex: {
      assert(0 && "not implemented");
      return;
    }
    case TEK_Aggregate:
      assert(0 && "not implemented");
      return;
    }
    llvm_unreachable("bad evaluation kind");
  }

  void buildAutoVarInit(const AutoVarEmission &emission) {
    assert(emission.Variable && "emission was not valid!");

    const VarDecl &D = *emission.Variable;
    QualType type = D.getType();

    // If this local has an initializer, emit it now.
    const Expr *Init = D.getInit();

    // TODO: in LLVM codegen if we are at an unreachable point, the initializer
    // isn't emitted unless it contains a label. What we want for CIR?
    assert(builder.getInsertionBlock());

    // Initialize the variable here if it doesn't have a initializer and it is a
    // C struct that is non-trivial to initialize or an array containing such a
    // struct.
    if (!Init && type.isNonTrivialToPrimitiveDefaultInitialize() ==
                     QualType::PDIK_Struct) {
      assert(0 && "not implemented");
      return;
    }

    const RawAddress Loc = emission.Addr;

    // Note: constexpr already initializes everything correctly.
    LangOptions::TrivialAutoVarInitKind trivialAutoVarInit =
        (D.isConstexpr()
             ? LangOptions::TrivialAutoVarInitKind::Uninitialized
             : (D.getAttr<UninitializedAttr>()
                    ? LangOptions::TrivialAutoVarInitKind::Uninitialized
                    : astCtx.getLangOpts().getTrivialAutoVarInit()));

    auto initializeWhatIsTechnicallyUninitialized = [&](RawAddress Loc) {
      if (trivialAutoVarInit ==
          LangOptions::TrivialAutoVarInitKind::Uninitialized)
        return;

      assert(0 && "unimplemented");
    };

    if (isTrivialInitializer(Init))
      return initializeWhatIsTechnicallyUninitialized(Loc);

    if (emission.IsConstantAggregate ||
        D.mightBeUsableInConstantExpressions(astCtx)) {
      assert(0 && "not implemented");
    }

    initializeWhatIsTechnicallyUninitialized(Loc);
    LValue lv = LValue::makeAddr(Loc, type, AlignmentSource::Decl);
    return buildExprAsInit(Init, &D, lv);
  }

  void buildAutoVarCleanups(const AutoVarEmission &emission) {
    assert(emission.Variable && "emission was not valid!");

    // TODO: in LLVM codegen if we are at an unreachable point codgen
    // is ignored. What we want for CIR?
    assert(builder.getInsertionBlock());
    const VarDecl &D = *emission.Variable;

    // Check the type for a cleanup.
    // TODO: something like emitAutoVarTypeCleanup
    if (QualType::DestructionKind dtorKind = D.needsDestruction(astCtx))
      assert(0 && "not implemented");

    // In GC mode, honor objc_precise_lifetime.
    if (astCtx.getLangOpts().getGC() != LangOptions::NonGC &&
        D.hasAttr<ObjCPreciseLifetimeAttr>())
      assert(0 && "not implemented");

    // Handle the cleanup attribute.
    if (const CleanupAttr *CA = D.getAttr<CleanupAttr>())
      assert(0 && "not implemented");

    // TODO: handle block variable
  }

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void buildAutoVarDecl(const VarDecl &D) {
    AutoVarEmission emission = buildAutoVarAlloca(D);
    buildAutoVarInit(emission);
    buildAutoVarCleanups(emission);
  }

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void buildVarDecl(const VarDecl &D) {
    if (D.hasExternalStorage()) {
      assert(0 && "should we just returns is there something to track?");
      // Don't emit it now, allow it to be emitted lazily on its first use.
      return;
    }

    // Some function-scope variable does not have static storage but still
    // needs to be emitted like a static variable, e.g. a function-scope
    // variable in constant address space in OpenCL.
    if (D.getStorageDuration() != SD_Automatic)
      assert(0 && "not implemented");

    if (D.getType().getAddressSpace() == LangAS::opencl_local)
      assert(0 && "not implemented");

    assert(D.hasLocalStorage());
    return buildAutoVarDecl(D);
  }

  void buildDecl(const Decl &D) {
    switch (D.getKind()) {
    case Decl::TopLevelStmt:
    case Decl::ImplicitConceptSpecialization:
    case Decl::HLSLBuffer:
    case Decl::UnnamedGlobalConstant:
      llvm_unreachable("NYI");
    case Decl::BuiltinTemplate:
    case Decl::TranslationUnit:
    case Decl::ExternCContext:
    case Decl::Namespace:
    case Decl::UnresolvedUsingTypename:
    case Decl::ClassTemplateSpecialization:
    case Decl::ClassTemplatePartialSpecialization:
    case Decl::VarTemplateSpecialization:
    case Decl::VarTemplatePartialSpecialization:
    case Decl::TemplateTypeParm:
    case Decl::UnresolvedUsingValue:
    case Decl::NonTypeTemplateParm:
    case Decl::CXXDeductionGuide:
    case Decl::CXXMethod:
    case Decl::CXXConstructor:
    case Decl::CXXDestructor:
    case Decl::CXXConversion:
    case Decl::Field:
    case Decl::MSProperty:
    case Decl::IndirectField:
    case Decl::ObjCIvar:
    case Decl::ObjCAtDefsField:
    case Decl::ParmVar:
    case Decl::ImplicitParam:
    case Decl::ClassTemplate:
    case Decl::VarTemplate:
    case Decl::FunctionTemplate:
    case Decl::TypeAliasTemplate:
    case Decl::TemplateTemplateParm:
    case Decl::ObjCMethod:
    case Decl::ObjCCategory:
    case Decl::ObjCProtocol:
    case Decl::ObjCInterface:
    case Decl::ObjCCategoryImpl:
    case Decl::ObjCImplementation:
    case Decl::ObjCProperty:
    case Decl::ObjCCompatibleAlias:
    case Decl::PragmaComment:
    case Decl::PragmaDetectMismatch:
    case Decl::AccessSpec:
    case Decl::LinkageSpec:
    case Decl::Export:
    case Decl::ObjCPropertyImpl:
    case Decl::FileScopeAsm:
    case Decl::Friend:
    case Decl::FriendTemplate:
    case Decl::Block:
    case Decl::Captured:
    case Decl::UsingShadow:
    case Decl::ConstructorUsingShadow:
    case Decl::ObjCTypeParam:
    case Decl::Binding:
    case Decl::UnresolvedUsingIfExists:
      llvm_unreachable("Declaration should not be in declstmts!");
    case Decl::Record:    // struct/union/class X;
    case Decl::CXXRecord: // struct/union/class X; [C++]
      assert(0 && "Not implemented");
      return;
    case Decl::Enum: // enum X;
      assert(0 && "Not implemented");
      return;
    case Decl::Function:     // void X();
    case Decl::EnumConstant: // enum ? { X = ? }
    case Decl::StaticAssert: // static_assert(X, ""); [C++0x]
    case Decl::Label:        // __label__ x;
    case Decl::Import:
    case Decl::MSGuid: // __declspec(uuid("..."))
    case Decl::TemplateParamObject:
    case Decl::OMPThreadPrivate:
    case Decl::OMPAllocate:
    case Decl::OMPCapturedExpr:
    case Decl::OMPRequires:
    case Decl::Empty:
    case Decl::Concept:
    case Decl::LifetimeExtendedTemporary:
    case Decl::RequiresExprBody:
      // None of these decls require codegen support.
      return;

    case Decl::NamespaceAlias:
      assert(0 && "Not implemented");
      return;
    case Decl::Using: // using X; [C++]
      assert(0 && "Not implemented");
      return;
    case Decl::UsingEnum: // using enum X; [C++]
      assert(0 && "Not implemented");
      return;
    case Decl::UsingPack:
      assert(0 && "Not implemented");
      return;
    case Decl::UsingDirective: // using namespace X; [C++]
      assert(0 && "Not implemented");
      return;
    case Decl::Var:
    case Decl::Decomposition: {
      const VarDecl &VD = cast<VarDecl>(D);
      assert(VD.isLocalVarDecl() &&
             "Should not see file-scope variables inside a function!");
      buildVarDecl(VD);
      if (auto *DD = dyn_cast<DecompositionDecl>(&VD))
        assert(0 && "Not implemented");

      // FIXME: add this
      // if (auto *DD = dyn_cast<DecompositionDecl>(&VD))
      //   for (auto *B : DD->bindings())
      //     if (auto *HD = B->getHoldingVar())
      //       EmitVarDecl(*HD);
      return;
    }

    case Decl::OMPDeclareReduction:
    case Decl::OMPDeclareMapper:
      assert(0 && "Not implemented");

    case Decl::Typedef:     // typedef int X;
    case Decl::TypeAlias: { // using X = int; [C++0x]
      assert(0 && "Not implemented");
    }
    }
  }

  /// Emit the computation of the specified expression of scalar type,
  /// ignoring the result.
  mlir::Value buildScalarExpr(const Expr *E) {
    assert(E && CIRCodeGenFunction::hasScalarEvaluationKind(E->getType()) &&
           "Invalid scalar expression to emit");

    return ScalarExprEmitter(*CurCCGF, *this).Visit(const_cast<Expr *>(E));
  }

  mlir::LogicalResult buildReturnStmt(const ReturnStmt &S) {
    assert(!(astCtx.getLangOpts().ElideConstructors && S.getNRVOCandidate() &&
             S.getNRVOCandidate()->isNRVOVariable()) &&
           "unimplemented");
    assert(!CurCCGF->FnRetQualTy->isReferenceType() && "unimplemented");

    // Emit the result value, even if unused, to evaluate the side effects.
    const Expr *RV = S.getRetValue();
    if (!RV) // Do nothing (return value is left uninitialized)
      return mlir::success();
    assert(!isa<ExprWithCleanups>(RV) && "unimplemented");

    mlir::Value V = nullptr;
    switch (CIRCodeGenFunction::getEvaluationKind(RV->getType())) {
    case TEK_Scalar:
      V = buildScalarExpr(RV);
      // Builder.CreateStore(EmitScalarExpr(RV), ReturnValue);
      break;
    case TEK_Complex:
    case TEK_Aggregate:
      llvm::errs() << "ReturnStmt EvaluationKind not implemented\n";
      return mlir::failure();
    }

    CurCCGF->RetValue = V;
    // Otherwise, this return operation has zero operands.
    if (!V || (RV && RV->getType()->isVoidType())) {
      // FIXME: evaluate for side effects.
    }

    builder.create<ReturnOp>(getLoc(RV->getExprLoc()),
                             V ? ArrayRef(V) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  mlir::LogicalResult buildDeclStmt(const DeclStmt &S) {
    if (!builder.getInsertionBlock())
      theModule.emitError(
          "Seems like this is unreachable code, what should we do?");

    for (const auto *I : S.decls()) {
      buildDecl(*I);
    }

    return mlir::success();
  }

  mlir::LogicalResult buildSimpleStmt(const Stmt *S) {
    switch (S->getStmtClass()) {
    default:
      return mlir::failure();
    case Stmt::DeclStmtClass:
      return buildDeclStmt(cast<DeclStmt>(*S));
    case Stmt::CompoundStmtClass:
      return buildCompoundStmt(cast<CompoundStmt>(*S));
    case Stmt::ReturnStmtClass:
      return buildReturnStmt(cast<ReturnStmt>(*S));
    case Stmt::NullStmtClass:
      break;

    case Stmt::LabelStmtClass:
    case Stmt::AttributedStmtClass:
    case Stmt::GotoStmtClass:
    case Stmt::BreakStmtClass:
    case Stmt::ContinueStmtClass:
    case Stmt::DefaultStmtClass:
    case Stmt::CaseStmtClass:
    case Stmt::SEHLeaveStmtClass:
      llvm::errs() << "CIR codegen for '" << S->getStmtClassName()
                   << "' not implemented\n";
      assert(0 && "not implemented");
    }

    return mlir::success();
  }

  mlir::LogicalResult buildStmt(const Stmt *S) {
    if (mlir::succeeded(buildSimpleStmt(S)))
      return mlir::success();
    assert(0 && "not implemented");
    return mlir::failure();
  }

  mlir::LogicalResult buildFunctionBody(const Stmt *Body) {
    const CompoundStmt *S = dyn_cast<CompoundStmt>(Body);
    assert(S && "expected compound stmt");
    return buildCompoundStmt(*S);
  }

  mlir::LogicalResult buildCompoundStmt(const CompoundStmt &S) {
    // Create a scope in the symbol table to hold variable declarations local
    // to this compound statement.
    SymTableScopeTy varScope(symbolTable);
    for (auto *CurStmt : S.body())
      if (buildStmt(CurStmt).failed())
        return mlir::failure();

    return mlir::success();
  }

  // Emit a new function and add it to the MLIR module.
  mlir::FuncOp buildCIR(CIRCodeGenFunction *CCGF, const FunctionDecl *FD) {
    CurCCGF = CCGF;

    // Create a scope in the symbol table to hold variable declarations.
    SymTableScopeTy varScope(symbolTable);

    const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
    assert(!MD && "methods not implemented");
    auto loc = getLoc(FD->getLocation());

    // Create an MLIR function for the given prototype.
    llvm::SmallVector<mlir::Type, 4> argTypes;

    for (auto *Param : FD->parameters())
      argTypes.push_back(getCIRType(Param->getType()));

    CurCCGF->FnRetQualTy = FD->getReturnType();
    auto funcType = builder.getFunctionType(
        argTypes, CurCCGF->FnRetQualTy->isVoidType()
                      ? mlir::TypeRange()
                      : getCIRType(CurCCGF->FnRetQualTy));
    mlir::FuncOp function = mlir::FuncOp::create(loc, FD->getName(), funcType);
    if (!function)
      return nullptr;

    // In MLIR the entry block of the function is special: it must have the
    // same argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();

    // Set the insertion point in the builder to the beginning of the
    // function body, it will be used throughout the codegen to create
    // operations in this function.
    builder.setInsertionPointToStart(&entryBlock);

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(FD->parameters(), entryBlock.getArguments())) {
      auto *paramVar = std::get<0>(nameValue);
      auto paramVal = std::get<1>(nameValue);
      mlir::Value addr;
      if (failed(declare(paramVar, paramVar->getType(),
                         getLoc(paramVar->getSourceRange().getBegin()), addr,
                         true /*param*/)))
        return nullptr;
      // Store params in local storage. FIXME: is this really needed
      // at this level of representation?
      builder.create<mlir::cir::StoreOp>(loc, paramVal, addr);
    }

    // Emit the body of the function.
    if (mlir::failed(buildFunctionBody(FD->getBody()))) {
      function.erase();
      return nullptr;
    }

    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp)
      builder.create<ReturnOp>(loc);

    if (mlir::failed(function.verifyBody()))
      return nullptr;
    theModule.push_back(function);
    return function;
  }

  mlir::Type getCIRType(const QualType &type) {
    return genTypes->ConvertType(type);
  }

  void verifyModule() {
    // Verify the module after we have finished constructing it, this will
    // check the structural properties of the IR and invoke any specific
    // verifiers we have on the CIR operations.
    if (failed(mlir::verify(theModule)))
      theModule.emitError("module verification error");
  }
};
} // namespace cir

CIRContext::CIRContext() {}

CIRContext::CIRContext(std::unique_ptr<raw_pwrite_stream> os)
    : outStream(std::move(os)) {}

CIRContext::~CIRContext() {
  // Run module verifier before shutdown.
  builder->verifyModule();
}

void CIRContext::Initialize(clang::ASTContext &astCtx) {
  using namespace llvm;

  this->astCtx = &astCtx;

  mlirCtx = std::make_unique<mlir::MLIRContext>();
  mlirCtx->getOrLoadDialect<mlir::func::FuncDialect>();
  mlirCtx->getOrLoadDialect<mlir::cir::CIRDialect>();
  mlirCtx->getOrLoadDialect<mlir::memref::MemRefDialect>();
  builder = std::make_unique<CIRBuildImpl>(*mlirCtx.get(), astCtx);
}

bool CIRContext::EmitFunction(const FunctionDecl *FD) {
  CIRCodeGenFunction CCGF{};
  auto func = builder->buildCIR(&CCGF, FD);
  assert(func && "should emit function");
  return true;
}

bool CIRContext::HandleTopLevelDecl(clang::DeclGroupRef D) {
  for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I) {
    auto *FD = cast<clang::FunctionDecl>(*I);
    assert(FD && "We can't handle anything else yet");
    EmitFunction(FD);
  }

  return true;
}

void CIRContext::HandleTranslationUnit(ASTContext &C) {
  if (outStream)
    builder->getModule()->print(*outStream);
}
