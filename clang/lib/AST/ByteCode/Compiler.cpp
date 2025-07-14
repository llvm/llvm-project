//===--- Compiler.cpp - Code generator for expressions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Compiler.h"
#include "ByteCodeEmitter.h"
#include "Context.h"
#include "FixedPoint.h"
#include "Floating.h"
#include "Function.h"
#include "InterpShared.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/Attr.h"

using namespace clang;
using namespace clang::interp;

using APSInt = llvm::APSInt;

namespace clang {
namespace interp {

static std::optional<bool> getBoolValue(const Expr *E) {
  if (const auto *CE = dyn_cast_if_present<ConstantExpr>(E);
      CE && CE->hasAPValueResult() &&
      CE->getResultAPValueKind() == APValue::ValueKind::Int) {
    return CE->getResultAsAPSInt().getBoolValue();
  }

  return std::nullopt;
}

/// Scope used to handle temporaries in toplevel variable declarations.
template <class Emitter> class DeclScope final : public LocalScope<Emitter> {
public:
  DeclScope(Compiler<Emitter> *Ctx, const ValueDecl *VD)
      : LocalScope<Emitter>(Ctx, VD), Scope(Ctx->P),
        OldInitializingDecl(Ctx->InitializingDecl) {
    Ctx->InitializingDecl = VD;
    Ctx->InitStack.push_back(InitLink::Decl(VD));
  }

  ~DeclScope() {
    this->Ctx->InitializingDecl = OldInitializingDecl;
    this->Ctx->InitStack.pop_back();
  }

private:
  Program::DeclScope Scope;
  const ValueDecl *OldInitializingDecl;
};

/// Scope used to handle initialization methods.
template <class Emitter> class OptionScope final {
public:
  /// Root constructor, compiling or discarding primitives.
  OptionScope(Compiler<Emitter> *Ctx, bool NewDiscardResult,
              bool NewInitializing)
      : Ctx(Ctx), OldDiscardResult(Ctx->DiscardResult),
        OldInitializing(Ctx->Initializing) {
    Ctx->DiscardResult = NewDiscardResult;
    Ctx->Initializing = NewInitializing;
  }

  ~OptionScope() {
    Ctx->DiscardResult = OldDiscardResult;
    Ctx->Initializing = OldInitializing;
  }

private:
  /// Parent context.
  Compiler<Emitter> *Ctx;
  /// Old discard flag to restore.
  bool OldDiscardResult;
  bool OldInitializing;
};

template <class Emitter>
bool InitLink::emit(Compiler<Emitter> *Ctx, const Expr *E) const {
  switch (Kind) {
  case K_This:
    return Ctx->emitThis(E);
  case K_Field:
    // We're assuming there's a base pointer on the stack already.
    return Ctx->emitGetPtrFieldPop(Offset, E);
  case K_Temp:
    return Ctx->emitGetPtrLocal(Offset, E);
  case K_Decl:
    return Ctx->visitDeclRef(D, E);
  case K_Elem:
    if (!Ctx->emitConstUint32(Offset, E))
      return false;
    return Ctx->emitArrayElemPtrPopUint32(E);
  case K_RVO:
    return Ctx->emitRVOPtr(E);
  case K_InitList:
    return true;
  default:
    llvm_unreachable("Unhandled InitLink kind");
  }
  return true;
}

/// Scope managing label targets.
template <class Emitter> class LabelScope {
public:
  virtual ~LabelScope() {}

protected:
  LabelScope(Compiler<Emitter> *Ctx) : Ctx(Ctx) {}
  /// Compiler instance.
  Compiler<Emitter> *Ctx;
};

/// Sets the context for break/continue statements.
template <class Emitter> class LoopScope final : public LabelScope<Emitter> {
public:
  using LabelTy = typename Compiler<Emitter>::LabelTy;
  using OptLabelTy = typename Compiler<Emitter>::OptLabelTy;

  LoopScope(Compiler<Emitter> *Ctx, LabelTy BreakLabel, LabelTy ContinueLabel)
      : LabelScope<Emitter>(Ctx), OldBreakLabel(Ctx->BreakLabel),
        OldContinueLabel(Ctx->ContinueLabel),
        OldBreakVarScope(Ctx->BreakVarScope),
        OldContinueVarScope(Ctx->ContinueVarScope) {
    this->Ctx->BreakLabel = BreakLabel;
    this->Ctx->ContinueLabel = ContinueLabel;
    this->Ctx->BreakVarScope = this->Ctx->VarScope;
    this->Ctx->ContinueVarScope = this->Ctx->VarScope;
  }

  ~LoopScope() {
    this->Ctx->BreakLabel = OldBreakLabel;
    this->Ctx->ContinueLabel = OldContinueLabel;
    this->Ctx->ContinueVarScope = OldContinueVarScope;
    this->Ctx->BreakVarScope = OldBreakVarScope;
  }

private:
  OptLabelTy OldBreakLabel;
  OptLabelTy OldContinueLabel;
  VariableScope<Emitter> *OldBreakVarScope;
  VariableScope<Emitter> *OldContinueVarScope;
};

// Sets the context for a switch scope, mapping labels.
template <class Emitter> class SwitchScope final : public LabelScope<Emitter> {
public:
  using LabelTy = typename Compiler<Emitter>::LabelTy;
  using OptLabelTy = typename Compiler<Emitter>::OptLabelTy;
  using CaseMap = typename Compiler<Emitter>::CaseMap;

  SwitchScope(Compiler<Emitter> *Ctx, CaseMap &&CaseLabels, LabelTy BreakLabel,
              OptLabelTy DefaultLabel)
      : LabelScope<Emitter>(Ctx), OldBreakLabel(Ctx->BreakLabel),
        OldDefaultLabel(this->Ctx->DefaultLabel),
        OldCaseLabels(std::move(this->Ctx->CaseLabels)),
        OldLabelVarScope(Ctx->BreakVarScope) {
    this->Ctx->BreakLabel = BreakLabel;
    this->Ctx->DefaultLabel = DefaultLabel;
    this->Ctx->CaseLabels = std::move(CaseLabels);
    this->Ctx->BreakVarScope = this->Ctx->VarScope;
  }

  ~SwitchScope() {
    this->Ctx->BreakLabel = OldBreakLabel;
    this->Ctx->DefaultLabel = OldDefaultLabel;
    this->Ctx->CaseLabels = std::move(OldCaseLabels);
    this->Ctx->BreakVarScope = OldLabelVarScope;
  }

private:
  OptLabelTy OldBreakLabel;
  OptLabelTy OldDefaultLabel;
  CaseMap OldCaseLabels;
  VariableScope<Emitter> *OldLabelVarScope;
};

template <class Emitter> class StmtExprScope final {
public:
  StmtExprScope(Compiler<Emitter> *Ctx) : Ctx(Ctx), OldFlag(Ctx->InStmtExpr) {
    Ctx->InStmtExpr = true;
  }

  ~StmtExprScope() { Ctx->InStmtExpr = OldFlag; }

private:
  Compiler<Emitter> *Ctx;
  bool OldFlag;
};

} // namespace interp
} // namespace clang

template <class Emitter>
bool Compiler<Emitter>::VisitCastExpr(const CastExpr *CE) {
  const Expr *SubExpr = CE->getSubExpr();

  if (DiscardResult)
    return this->delegate(SubExpr);

  switch (CE->getCastKind()) {
  case CK_LValueToRValue: {
    if (SubExpr->getType().isVolatileQualified())
      return this->emitInvalidCast(CastKind::Volatile, /*Fatal=*/true, CE);

    std::optional<PrimType> SubExprT = classify(SubExpr->getType());
    // Prepare storage for the result.
    if (!Initializing && !SubExprT) {
      std::optional<unsigned> LocalIndex = allocateLocal(SubExpr);
      if (!LocalIndex)
        return false;
      if (!this->emitGetPtrLocal(*LocalIndex, CE))
        return false;
    }

    if (!this->visit(SubExpr))
      return false;

    if (SubExprT)
      return this->emitLoadPop(*SubExprT, CE);

    // If the subexpr type is not primitive, we need to perform a copy here.
    // This happens for example in C when dereferencing a pointer of struct
    // type.
    return this->emitMemcpy(CE);
  }

  case CK_DerivedToBaseMemberPointer: {
    assert(classifyPrim(CE->getType()) == PT_MemberPtr);
    assert(classifyPrim(SubExpr->getType()) == PT_MemberPtr);
    const auto *FromMP = SubExpr->getType()->castAs<MemberPointerType>();
    const auto *ToMP = CE->getType()->castAs<MemberPointerType>();

    unsigned DerivedOffset =
        Ctx.collectBaseOffset(ToMP->getMostRecentCXXRecordDecl(),
                              FromMP->getMostRecentCXXRecordDecl());

    if (!this->delegate(SubExpr))
      return false;

    return this->emitGetMemberPtrBasePop(DerivedOffset, CE);
  }

  case CK_BaseToDerivedMemberPointer: {
    assert(classifyPrim(CE) == PT_MemberPtr);
    assert(classifyPrim(SubExpr) == PT_MemberPtr);
    const auto *FromMP = SubExpr->getType()->castAs<MemberPointerType>();
    const auto *ToMP = CE->getType()->castAs<MemberPointerType>();

    unsigned DerivedOffset =
        Ctx.collectBaseOffset(FromMP->getMostRecentCXXRecordDecl(),
                              ToMP->getMostRecentCXXRecordDecl());

    if (!this->delegate(SubExpr))
      return false;
    return this->emitGetMemberPtrBasePop(-DerivedOffset, CE);
  }

  case CK_UncheckedDerivedToBase:
  case CK_DerivedToBase: {
    if (!this->delegate(SubExpr))
      return false;

    const auto extractRecordDecl = [](QualType Ty) -> const CXXRecordDecl * {
      if (const auto *PT = dyn_cast<PointerType>(Ty))
        return PT->getPointeeType()->getAsCXXRecordDecl();
      return Ty->getAsCXXRecordDecl();
    };

    // FIXME: We can express a series of non-virtual casts as a single
    // GetPtrBasePop op.
    QualType CurType = SubExpr->getType();
    for (const CXXBaseSpecifier *B : CE->path()) {
      if (B->isVirtual()) {
        if (!this->emitGetPtrVirtBasePop(extractRecordDecl(B->getType()), CE))
          return false;
        CurType = B->getType();
      } else {
        unsigned DerivedOffset = collectBaseOffset(B->getType(), CurType);
        if (!this->emitGetPtrBasePop(
                DerivedOffset, /*NullOK=*/CE->getType()->isPointerType(), CE))
          return false;
        CurType = B->getType();
      }
    }

    return true;
  }

  case CK_BaseToDerived: {
    if (!this->delegate(SubExpr))
      return false;
    unsigned DerivedOffset =
        collectBaseOffset(SubExpr->getType(), CE->getType());

    const Type *TargetType = CE->getType().getTypePtr();
    if (TargetType->isPointerOrReferenceType())
      TargetType = TargetType->getPointeeType().getTypePtr();
    return this->emitGetPtrDerivedPop(DerivedOffset,
                                      /*NullOK=*/CE->getType()->isPointerType(),
                                      TargetType, CE);
  }

  case CK_FloatingCast: {
    // HLSL uses CK_FloatingCast to cast between vectors.
    if (!SubExpr->getType()->isFloatingType() ||
        !CE->getType()->isFloatingType())
      return false;
    if (!this->visit(SubExpr))
      return false;
    const auto *TargetSemantics = &Ctx.getFloatSemantics(CE->getType());
    return this->emitCastFP(TargetSemantics, getRoundingMode(CE), CE);
  }

  case CK_IntegralToFloating: {
    if (!CE->getType()->isRealFloatingType())
      return false;
    if (!this->visit(SubExpr))
      return false;
    const auto *TargetSemantics = &Ctx.getFloatSemantics(CE->getType());
    return this->emitCastIntegralFloating(
        classifyPrim(SubExpr), TargetSemantics, getFPOptions(CE), CE);
  }

  case CK_FloatingToBoolean: {
    if (!SubExpr->getType()->isRealFloatingType() ||
        !CE->getType()->isBooleanType())
      return false;
    if (const auto *FL = dyn_cast<FloatingLiteral>(SubExpr))
      return this->emitConstBool(FL->getValue().isNonZero(), CE);
    if (!this->visit(SubExpr))
      return false;
    return this->emitCastFloatingIntegralBool(getFPOptions(CE), CE);
  }

  case CK_FloatingToIntegral: {
    if (!this->visit(SubExpr))
      return false;
    PrimType ToT = classifyPrim(CE);
    if (ToT == PT_IntAP)
      return this->emitCastFloatingIntegralAP(Ctx.getBitWidth(CE->getType()),
                                              getFPOptions(CE), CE);
    if (ToT == PT_IntAPS)
      return this->emitCastFloatingIntegralAPS(Ctx.getBitWidth(CE->getType()),
                                               getFPOptions(CE), CE);

    return this->emitCastFloatingIntegral(ToT, getFPOptions(CE), CE);
  }

  case CK_NullToPointer:
  case CK_NullToMemberPointer: {
    if (!this->discard(SubExpr))
      return false;
    const Descriptor *Desc = nullptr;
    const QualType PointeeType = CE->getType()->getPointeeType();
    if (!PointeeType.isNull()) {
      if (std::optional<PrimType> T = classify(PointeeType))
        Desc = P.createDescriptor(SubExpr, *T);
      else
        Desc = P.createDescriptor(SubExpr, PointeeType.getTypePtr(),
                                  std::nullopt, /*IsConst=*/true);
    }

    uint64_t Val = Ctx.getASTContext().getTargetNullPointerValue(CE->getType());
    return this->emitNull(classifyPrim(CE->getType()), Val, Desc, CE);
  }

  case CK_PointerToIntegral: {
    if (!this->visit(SubExpr))
      return false;

    // If SubExpr doesn't result in a pointer, make it one.
    if (PrimType FromT = classifyPrim(SubExpr->getType()); FromT != PT_Ptr) {
      assert(isPtrType(FromT));
      if (!this->emitDecayPtr(FromT, PT_Ptr, CE))
        return false;
    }

    PrimType T = classifyPrim(CE->getType());
    if (T == PT_IntAP)
      return this->emitCastPointerIntegralAP(Ctx.getBitWidth(CE->getType()),
                                             CE);
    if (T == PT_IntAPS)
      return this->emitCastPointerIntegralAPS(Ctx.getBitWidth(CE->getType()),
                                              CE);
    return this->emitCastPointerIntegral(T, CE);
  }

  case CK_ArrayToPointerDecay: {
    if (!this->visit(SubExpr))
      return false;
    return this->emitArrayDecay(CE);
  }

  case CK_IntegralToPointer: {
    QualType IntType = SubExpr->getType();
    assert(IntType->isIntegralOrEnumerationType());
    if (!this->visit(SubExpr))
      return false;
    // FIXME: I think the discard is wrong since the int->ptr cast might cause a
    // diagnostic.
    PrimType T = classifyPrim(IntType);
    QualType PtrType = CE->getType();
    const Descriptor *Desc;
    if (std::optional<PrimType> T = classify(PtrType->getPointeeType()))
      Desc = P.createDescriptor(SubExpr, *T);
    else if (PtrType->getPointeeType()->isVoidType())
      Desc = nullptr;
    else
      Desc = P.createDescriptor(CE, PtrType->getPointeeType().getTypePtr(),
                                Descriptor::InlineDescMD, /*IsConst=*/true);

    if (!this->emitGetIntPtr(T, Desc, CE))
      return false;

    PrimType DestPtrT = classifyPrim(PtrType);
    if (DestPtrT == PT_Ptr)
      return true;

    // In case we're converting the integer to a non-Pointer.
    return this->emitDecayPtr(PT_Ptr, DestPtrT, CE);
  }

  case CK_AtomicToNonAtomic:
  case CK_ConstructorConversion:
  case CK_FunctionToPointerDecay:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_AddressSpaceConversion:
  case CK_CPointerToObjCPointerCast:
    return this->delegate(SubExpr);

  case CK_BitCast: {
    // Reject bitcasts to atomic types.
    if (CE->getType()->isAtomicType()) {
      if (!this->discard(SubExpr))
        return false;
      return this->emitInvalidCast(CastKind::Reinterpret, /*Fatal=*/true, CE);
    }
    QualType SubExprTy = SubExpr->getType();
    std::optional<PrimType> FromT = classify(SubExprTy);
    // Casts from integer/vector to vector.
    if (CE->getType()->isVectorType())
      return this->emitBuiltinBitCast(CE);

    std::optional<PrimType> ToT = classify(CE->getType());
    if (!FromT || !ToT)
      return false;

    assert(isPtrType(*FromT));
    assert(isPtrType(*ToT));
    if (FromT == ToT) {
      if (CE->getType()->isVoidPointerType())
        return this->delegate(SubExpr);

      if (!this->visit(SubExpr))
        return false;
      if (CE->getType()->isFunctionPointerType())
        return true;
      if (FromT == PT_Ptr)
        return this->emitPtrPtrCast(SubExprTy->isVoidPointerType(), CE);
      return true;
    }

    if (!this->visit(SubExpr))
      return false;
    return this->emitDecayPtr(*FromT, *ToT, CE);
  }
  case CK_IntegralToBoolean:
  case CK_FixedPointToBoolean: {
    // HLSL uses this to cast to one-element vectors.
    std::optional<PrimType> FromT = classify(SubExpr->getType());
    if (!FromT)
      return false;

    if (const auto *IL = dyn_cast<IntegerLiteral>(SubExpr))
      return this->emitConst(IL->getValue(), CE);
    if (!this->visit(SubExpr))
      return false;
    return this->emitCast(*FromT, classifyPrim(CE), CE);
  }

  case CK_BooleanToSignedIntegral:
  case CK_IntegralCast: {
    std::optional<PrimType> FromT = classify(SubExpr->getType());
    std::optional<PrimType> ToT = classify(CE->getType());
    if (!FromT || !ToT)
      return false;

    // Try to emit a casted known constant value directly.
    if (const auto *IL = dyn_cast<IntegerLiteral>(SubExpr)) {
      if (ToT != PT_IntAP && ToT != PT_IntAPS && FromT != PT_IntAP &&
          FromT != PT_IntAPS && !CE->getType()->isEnumeralType())
        return this->emitConst(IL->getValue(), CE);
      if (!this->emitConst(IL->getValue(), SubExpr))
        return false;
    } else {
      if (!this->visit(SubExpr))
        return false;
    }

    // Possibly diagnose casts to enum types if the target type does not
    // have a fixed size.
    if (Ctx.getLangOpts().CPlusPlus && CE->getType()->isEnumeralType()) {
      if (const auto *ET = CE->getType().getCanonicalType()->castAs<EnumType>();
          !ET->getDecl()->isFixed()) {
        if (!this->emitCheckEnumValue(*FromT, ET->getDecl(), CE))
          return false;
      }
    }

    if (ToT == PT_IntAP) {
      if (!this->emitCastAP(*FromT, Ctx.getBitWidth(CE->getType()), CE))
        return false;
    } else if (ToT == PT_IntAPS) {
      if (!this->emitCastAPS(*FromT, Ctx.getBitWidth(CE->getType()), CE))
        return false;
    } else {
      if (FromT == ToT)
        return true;
      if (!this->emitCast(*FromT, *ToT, CE))
        return false;
    }
    if (CE->getCastKind() == CK_BooleanToSignedIntegral)
      return this->emitNeg(*ToT, CE);
    return true;
  }

  case CK_PointerToBoolean:
  case CK_MemberPointerToBoolean: {
    PrimType PtrT = classifyPrim(SubExpr->getType());

    if (!this->visit(SubExpr))
      return false;
    return this->emitIsNonNull(PtrT, CE);
  }

  case CK_IntegralComplexToBoolean:
  case CK_FloatingComplexToBoolean: {
    if (!this->visit(SubExpr))
      return false;
    return this->emitComplexBoolCast(SubExpr);
  }

  case CK_IntegralComplexToReal:
  case CK_FloatingComplexToReal:
    return this->emitComplexReal(SubExpr);

  case CK_IntegralRealToComplex:
  case CK_FloatingRealToComplex: {
    // We're creating a complex value here, so we need to
    // allocate storage for it.
    if (!Initializing) {
      std::optional<unsigned> LocalIndex = allocateTemporary(CE);
      if (!LocalIndex)
        return false;
      if (!this->emitGetPtrLocal(*LocalIndex, CE))
        return false;
    }

    PrimType T = classifyPrim(SubExpr->getType());
    // Init the complex value to {SubExpr, 0}.
    if (!this->visitArrayElemInit(0, SubExpr, T))
      return false;
    // Zero-init the second element.
    if (!this->visitZeroInitializer(T, SubExpr->getType(), SubExpr))
      return false;
    return this->emitInitElem(T, 1, SubExpr);
  }

  case CK_IntegralComplexCast:
  case CK_FloatingComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_FloatingComplexToIntegralComplex: {
    assert(CE->getType()->isAnyComplexType());
    assert(SubExpr->getType()->isAnyComplexType());
    if (!Initializing) {
      std::optional<unsigned> LocalIndex = allocateLocal(CE);
      if (!LocalIndex)
        return false;
      if (!this->emitGetPtrLocal(*LocalIndex, CE))
        return false;
    }

    // Location for the SubExpr.
    // Since SubExpr is of complex type, visiting it results in a pointer
    // anyway, so we just create a temporary pointer variable.
    unsigned SubExprOffset =
        allocateLocalPrimitive(SubExpr, PT_Ptr, /*IsConst=*/true);
    if (!this->visit(SubExpr))
      return false;
    if (!this->emitSetLocal(PT_Ptr, SubExprOffset, CE))
      return false;

    PrimType SourceElemT = classifyComplexElementType(SubExpr->getType());
    QualType DestElemType =
        CE->getType()->getAs<ComplexType>()->getElementType();
    PrimType DestElemT = classifyPrim(DestElemType);
    // Cast both elements individually.
    for (unsigned I = 0; I != 2; ++I) {
      if (!this->emitGetLocal(PT_Ptr, SubExprOffset, CE))
        return false;
      if (!this->emitArrayElemPop(SourceElemT, I, CE))
        return false;

      // Do the cast.
      if (!this->emitPrimCast(SourceElemT, DestElemT, DestElemType, CE))
        return false;

      // Save the value.
      if (!this->emitInitElem(DestElemT, I, CE))
        return false;
    }
    return true;
  }

  case CK_VectorSplat: {
    assert(!classify(CE->getType()));
    assert(classify(SubExpr->getType()));
    assert(CE->getType()->isVectorType());

    if (!Initializing) {
      std::optional<unsigned> LocalIndex = allocateLocal(CE);
      if (!LocalIndex)
        return false;
      if (!this->emitGetPtrLocal(*LocalIndex, CE))
        return false;
    }

    const auto *VT = CE->getType()->getAs<VectorType>();
    PrimType ElemT = classifyPrim(SubExpr->getType());
    unsigned ElemOffset =
        allocateLocalPrimitive(SubExpr, ElemT, /*IsConst=*/true);

    // Prepare a local variable for the scalar value.
    if (!this->visit(SubExpr))
      return false;
    if (classifyPrim(SubExpr) == PT_Ptr && !this->emitLoadPop(ElemT, CE))
      return false;

    if (!this->emitSetLocal(ElemT, ElemOffset, CE))
      return false;

    for (unsigned I = 0; I != VT->getNumElements(); ++I) {
      if (!this->emitGetLocal(ElemT, ElemOffset, CE))
        return false;
      if (!this->emitInitElem(ElemT, I, CE))
        return false;
    }

    return true;
  }

  case CK_HLSLVectorTruncation: {
    assert(SubExpr->getType()->isVectorType());
    if (std::optional<PrimType> ResultT = classify(CE)) {
      assert(!DiscardResult);
      // Result must be either a float or integer. Take the first element.
      if (!this->visit(SubExpr))
        return false;
      return this->emitArrayElemPop(*ResultT, 0, CE);
    }
    // Otherwise, this truncates from one vector type to another.
    assert(CE->getType()->isVectorType());

    if (!Initializing) {
      std::optional<unsigned> LocalIndex = allocateTemporary(CE);
      if (!LocalIndex)
        return false;
      if (!this->emitGetPtrLocal(*LocalIndex, CE))
        return false;
    }
    unsigned ToSize = CE->getType()->getAs<VectorType>()->getNumElements();
    assert(SubExpr->getType()->getAs<VectorType>()->getNumElements() > ToSize);
    if (!this->visit(SubExpr))
      return false;
    return this->emitCopyArray(classifyVectorElementType(CE->getType()), 0, 0,
                               ToSize, CE);
  };

  case CK_IntegralToFixedPoint: {
    if (!this->visit(SubExpr))
      return false;

    auto Sem =
        Ctx.getASTContext().getFixedPointSemantics(CE->getType()).toOpaqueInt();
    return this->emitCastIntegralFixedPoint(classifyPrim(SubExpr->getType()),
                                            Sem, CE);
  }
  case CK_FloatingToFixedPoint: {
    if (!this->visit(SubExpr))
      return false;

    auto Sem =
        Ctx.getASTContext().getFixedPointSemantics(CE->getType()).toOpaqueInt();
    return this->emitCastFloatingFixedPoint(Sem, CE);
  }
  case CK_FixedPointToFloating: {
    if (!this->visit(SubExpr))
      return false;
    const auto *TargetSemantics = &Ctx.getFloatSemantics(CE->getType());
    return this->emitCastFixedPointFloating(TargetSemantics, CE);
  }
  case CK_FixedPointToIntegral: {
    if (!this->visit(SubExpr))
      return false;
    return this->emitCastFixedPointIntegral(classifyPrim(CE->getType()), CE);
  }
  case CK_FixedPointCast: {
    if (!this->visit(SubExpr))
      return false;
    auto Sem =
        Ctx.getASTContext().getFixedPointSemantics(CE->getType()).toOpaqueInt();
    return this->emitCastFixedPoint(Sem, CE);
  }

  case CK_ToVoid:
    return discard(SubExpr);

  default:
    return this->emitInvalid(CE);
  }
  llvm_unreachable("Unhandled clang::CastKind enum");
}

template <class Emitter>
bool Compiler<Emitter>::VisitBuiltinBitCastExpr(const BuiltinBitCastExpr *E) {
  return this->emitBuiltinBitCast(E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitIntegerLiteral(const IntegerLiteral *LE) {
  if (DiscardResult)
    return true;

  return this->emitConst(LE->getValue(), LE);
}

template <class Emitter>
bool Compiler<Emitter>::VisitFloatingLiteral(const FloatingLiteral *E) {
  if (DiscardResult)
    return true;

  APFloat F = E->getValue();
  return this->emitFloat(F, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitImaginaryLiteral(const ImaginaryLiteral *E) {
  assert(E->getType()->isAnyComplexType());
  if (DiscardResult)
    return true;

  if (!Initializing) {
    std::optional<unsigned> LocalIndex = allocateTemporary(E);
    if (!LocalIndex)
      return false;
    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
  }

  const Expr *SubExpr = E->getSubExpr();
  PrimType SubExprT = classifyPrim(SubExpr->getType());

  if (!this->visitZeroInitializer(SubExprT, SubExpr->getType(), SubExpr))
    return false;
  if (!this->emitInitElem(SubExprT, 0, SubExpr))
    return false;
  return this->visitArrayElemInit(1, SubExpr, SubExprT);
}

template <class Emitter>
bool Compiler<Emitter>::VisitFixedPointLiteral(const FixedPointLiteral *E) {
  assert(E->getType()->isFixedPointType());
  assert(classifyPrim(E) == PT_FixedPoint);

  if (DiscardResult)
    return true;

  auto Sem = Ctx.getASTContext().getFixedPointSemantics(E->getType());
  APInt Value = E->getValue();
  return this->emitConstFixedPoint(FixedPoint(Value, Sem), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitParenExpr(const ParenExpr *E) {
  return this->delegate(E->getSubExpr());
}

template <class Emitter>
bool Compiler<Emitter>::VisitBinaryOperator(const BinaryOperator *BO) {
  // Need short-circuiting for these.
  if (BO->isLogicalOp() && !BO->getType()->isVectorType())
    return this->VisitLogicalBinOp(BO);

  const Expr *LHS = BO->getLHS();
  const Expr *RHS = BO->getRHS();

  // Handle comma operators. Just discard the LHS
  // and delegate to RHS.
  if (BO->isCommaOp()) {
    if (!this->discard(LHS))
      return false;
    if (RHS->getType()->isVoidType())
      return this->discard(RHS);

    return this->delegate(RHS);
  }

  if (BO->getType()->isAnyComplexType())
    return this->VisitComplexBinOp(BO);
  if (BO->getType()->isVectorType())
    return this->VisitVectorBinOp(BO);
  if ((LHS->getType()->isAnyComplexType() ||
       RHS->getType()->isAnyComplexType()) &&
      BO->isComparisonOp())
    return this->emitComplexComparison(LHS, RHS, BO);
  if (LHS->getType()->isFixedPointType() || RHS->getType()->isFixedPointType())
    return this->VisitFixedPointBinOp(BO);

  if (BO->isPtrMemOp()) {
    if (!this->visit(LHS))
      return false;

    if (!this->visit(RHS))
      return false;

    if (!this->emitToMemberPtr(BO))
      return false;

    if (classifyPrim(BO) == PT_MemberPtr)
      return true;

    if (!this->emitCastMemberPtrPtr(BO))
      return false;
    return DiscardResult ? this->emitPopPtr(BO) : true;
  }

  // Typecheck the args.
  std::optional<PrimType> LT = classify(LHS);
  std::optional<PrimType> RT = classify(RHS);
  std::optional<PrimType> T = classify(BO->getType());

  // Special case for C++'s three-way/spaceship operator <=>, which
  // returns a std::{strong,weak,partial}_ordering (which is a class, so doesn't
  // have a PrimType).
  if (!T && BO->getOpcode() == BO_Cmp) {
    if (DiscardResult)
      return true;
    const ComparisonCategoryInfo *CmpInfo =
        Ctx.getASTContext().CompCategories.lookupInfoForType(BO->getType());
    assert(CmpInfo);

    // We need a temporary variable holding our return value.
    if (!Initializing) {
      std::optional<unsigned> ResultIndex = this->allocateLocal(BO);
      if (!this->emitGetPtrLocal(*ResultIndex, BO))
        return false;
    }

    if (!visit(LHS) || !visit(RHS))
      return false;

    return this->emitCMP3(*LT, CmpInfo, BO);
  }

  if (!LT || !RT || !T)
    return false;

  // Pointer arithmetic special case.
  if (BO->getOpcode() == BO_Add || BO->getOpcode() == BO_Sub) {
    if (isPtrType(*T) || (isPtrType(*LT) && isPtrType(*RT)))
      return this->VisitPointerArithBinOp(BO);
  }

  // Assignments require us to evalute the RHS first.
  if (BO->getOpcode() == BO_Assign) {

    if (!visit(RHS) || !visit(LHS))
      return false;

    // We don't support assignments in C.
    if (!Ctx.getLangOpts().CPlusPlus && !this->emitInvalid(BO))
      return false;

    if (!this->emitFlip(*LT, *RT, BO))
      return false;
  } else {
    if (!visit(LHS) || !visit(RHS))
      return false;
  }

  // For languages such as C, cast the result of one
  // of our comparision opcodes to T (which is usually int).
  auto MaybeCastToBool = [this, T, BO](bool Result) {
    if (!Result)
      return false;
    if (DiscardResult)
      return this->emitPop(*T, BO);
    if (T != PT_Bool)
      return this->emitCast(PT_Bool, *T, BO);
    return true;
  };

  auto Discard = [this, T, BO](bool Result) {
    if (!Result)
      return false;
    return DiscardResult ? this->emitPop(*T, BO) : true;
  };

  switch (BO->getOpcode()) {
  case BO_EQ:
    return MaybeCastToBool(this->emitEQ(*LT, BO));
  case BO_NE:
    return MaybeCastToBool(this->emitNE(*LT, BO));
  case BO_LT:
    return MaybeCastToBool(this->emitLT(*LT, BO));
  case BO_LE:
    return MaybeCastToBool(this->emitLE(*LT, BO));
  case BO_GT:
    return MaybeCastToBool(this->emitGT(*LT, BO));
  case BO_GE:
    return MaybeCastToBool(this->emitGE(*LT, BO));
  case BO_Sub:
    if (BO->getType()->isFloatingType())
      return Discard(this->emitSubf(getFPOptions(BO), BO));
    return Discard(this->emitSub(*T, BO));
  case BO_Add:
    if (BO->getType()->isFloatingType())
      return Discard(this->emitAddf(getFPOptions(BO), BO));
    return Discard(this->emitAdd(*T, BO));
  case BO_Mul:
    if (BO->getType()->isFloatingType())
      return Discard(this->emitMulf(getFPOptions(BO), BO));
    return Discard(this->emitMul(*T, BO));
  case BO_Rem:
    return Discard(this->emitRem(*T, BO));
  case BO_Div:
    if (BO->getType()->isFloatingType())
      return Discard(this->emitDivf(getFPOptions(BO), BO));
    return Discard(this->emitDiv(*T, BO));
  case BO_Assign:
    if (DiscardResult)
      return LHS->refersToBitField() ? this->emitStoreBitFieldPop(*T, BO)
                                     : this->emitStorePop(*T, BO);
    if (LHS->refersToBitField()) {
      if (!this->emitStoreBitField(*T, BO))
        return false;
    } else {
      if (!this->emitStore(*T, BO))
        return false;
    }
    // Assignments aren't necessarily lvalues in C.
    // Load from them in that case.
    if (!BO->isLValue())
      return this->emitLoadPop(*T, BO);
    return true;
  case BO_And:
    return Discard(this->emitBitAnd(*T, BO));
  case BO_Or:
    return Discard(this->emitBitOr(*T, BO));
  case BO_Shl:
    return Discard(this->emitShl(*LT, *RT, BO));
  case BO_Shr:
    return Discard(this->emitShr(*LT, *RT, BO));
  case BO_Xor:
    return Discard(this->emitBitXor(*T, BO));
  case BO_LOr:
  case BO_LAnd:
    llvm_unreachable("Already handled earlier");
  default:
    return false;
  }

  llvm_unreachable("Unhandled binary op");
}

/// Perform addition/subtraction of a pointer and an integer or
/// subtraction of two pointers.
template <class Emitter>
bool Compiler<Emitter>::VisitPointerArithBinOp(const BinaryOperator *E) {
  BinaryOperatorKind Op = E->getOpcode();
  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();

  if ((Op != BO_Add && Op != BO_Sub) ||
      (!LHS->getType()->isPointerType() && !RHS->getType()->isPointerType()))
    return false;

  std::optional<PrimType> LT = classify(LHS);
  std::optional<PrimType> RT = classify(RHS);

  if (!LT || !RT)
    return false;

  // Visit the given pointer expression and optionally convert to a PT_Ptr.
  auto visitAsPointer = [&](const Expr *E, PrimType T) -> bool {
    if (!this->visit(E))
      return false;
    if (T != PT_Ptr)
      return this->emitDecayPtr(T, PT_Ptr, E);
    return true;
  };

  if (LHS->getType()->isPointerType() && RHS->getType()->isPointerType()) {
    if (Op != BO_Sub)
      return false;

    assert(E->getType()->isIntegerType());
    if (!visitAsPointer(RHS, *RT) || !visitAsPointer(LHS, *LT))
      return false;

    PrimType IntT = classifyPrim(E->getType());
    if (!this->emitSubPtr(IntT, E))
      return false;
    return DiscardResult ? this->emitPop(IntT, E) : true;
  }

  PrimType OffsetType;
  if (LHS->getType()->isIntegerType()) {
    if (!visitAsPointer(RHS, *RT))
      return false;
    if (!this->visit(LHS))
      return false;
    OffsetType = *LT;
  } else if (RHS->getType()->isIntegerType()) {
    if (!visitAsPointer(LHS, *LT))
      return false;
    if (!this->visit(RHS))
      return false;
    OffsetType = *RT;
  } else {
    return false;
  }

  // Do the operation and optionally transform to
  // result pointer type.
  if (Op == BO_Add) {
    if (!this->emitAddOffset(OffsetType, E))
      return false;

    if (classifyPrim(E) != PT_Ptr)
      return this->emitDecayPtr(PT_Ptr, classifyPrim(E), E);
    return true;
  } else if (Op == BO_Sub) {
    if (!this->emitSubOffset(OffsetType, E))
      return false;

    if (classifyPrim(E) != PT_Ptr)
      return this->emitDecayPtr(PT_Ptr, classifyPrim(E), E);
    return true;
  }

  return false;
}

template <class Emitter>
bool Compiler<Emitter>::VisitLogicalBinOp(const BinaryOperator *E) {
  assert(E->isLogicalOp());
  BinaryOperatorKind Op = E->getOpcode();
  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  std::optional<PrimType> T = classify(E->getType());

  if (Op == BO_LOr) {
    // Logical OR. Visit LHS and only evaluate RHS if LHS was FALSE.
    LabelTy LabelTrue = this->getLabel();
    LabelTy LabelEnd = this->getLabel();

    if (!this->visitBool(LHS))
      return false;
    if (!this->jumpTrue(LabelTrue))
      return false;

    if (!this->visitBool(RHS))
      return false;
    if (!this->jump(LabelEnd))
      return false;

    this->emitLabel(LabelTrue);
    this->emitConstBool(true, E);
    this->fallthrough(LabelEnd);
    this->emitLabel(LabelEnd);

  } else {
    assert(Op == BO_LAnd);
    // Logical AND.
    // Visit LHS. Only visit RHS if LHS was TRUE.
    LabelTy LabelFalse = this->getLabel();
    LabelTy LabelEnd = this->getLabel();

    if (!this->visitBool(LHS))
      return false;
    if (!this->jumpFalse(LabelFalse))
      return false;

    if (!this->visitBool(RHS))
      return false;
    if (!this->jump(LabelEnd))
      return false;

    this->emitLabel(LabelFalse);
    this->emitConstBool(false, E);
    this->fallthrough(LabelEnd);
    this->emitLabel(LabelEnd);
  }

  if (DiscardResult)
    return this->emitPopBool(E);

  // For C, cast back to integer type.
  assert(T);
  if (T != PT_Bool)
    return this->emitCast(PT_Bool, *T, E);
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitComplexBinOp(const BinaryOperator *E) {
  // Prepare storage for result.
  if (!Initializing) {
    std::optional<unsigned> LocalIndex = allocateTemporary(E);
    if (!LocalIndex)
      return false;
    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
  }

  // Both LHS and RHS might _not_ be of complex type, but one of them
  // needs to be.
  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();

  PrimType ResultElemT = this->classifyComplexElementType(E->getType());
  unsigned ResultOffset = ~0u;
  if (!DiscardResult)
    ResultOffset = this->allocateLocalPrimitive(E, PT_Ptr, /*IsConst=*/true);

  // Save result pointer in ResultOffset
  if (!this->DiscardResult) {
    if (!this->emitDupPtr(E))
      return false;
    if (!this->emitSetLocal(PT_Ptr, ResultOffset, E))
      return false;
  }
  QualType LHSType = LHS->getType();
  if (const auto *AT = LHSType->getAs<AtomicType>())
    LHSType = AT->getValueType();
  QualType RHSType = RHS->getType();
  if (const auto *AT = RHSType->getAs<AtomicType>())
    RHSType = AT->getValueType();

  bool LHSIsComplex = LHSType->isAnyComplexType();
  unsigned LHSOffset;
  bool RHSIsComplex = RHSType->isAnyComplexType();

  // For ComplexComplex Mul, we have special ops to make their implementation
  // easier.
  BinaryOperatorKind Op = E->getOpcode();
  if (Op == BO_Mul && LHSIsComplex && RHSIsComplex) {
    assert(classifyPrim(LHSType->getAs<ComplexType>()->getElementType()) ==
           classifyPrim(RHSType->getAs<ComplexType>()->getElementType()));
    PrimType ElemT =
        classifyPrim(LHSType->getAs<ComplexType>()->getElementType());
    if (!this->visit(LHS))
      return false;
    if (!this->visit(RHS))
      return false;
    return this->emitMulc(ElemT, E);
  }

  if (Op == BO_Div && RHSIsComplex) {
    QualType ElemQT = RHSType->getAs<ComplexType>()->getElementType();
    PrimType ElemT = classifyPrim(ElemQT);
    // If the LHS is not complex, we still need to do the full complex
    // division, so just stub create a complex value and stub it out with
    // the LHS and a zero.

    if (!LHSIsComplex) {
      // This is using the RHS type for the fake-complex LHS.
      std::optional<unsigned> LocalIndex = allocateTemporary(RHS);
      if (!LocalIndex)
        return false;
      LHSOffset = *LocalIndex;

      if (!this->emitGetPtrLocal(LHSOffset, E))
        return false;

      if (!this->visit(LHS))
        return false;
      // real is LHS
      if (!this->emitInitElem(ElemT, 0, E))
        return false;
      // imag is zero
      if (!this->visitZeroInitializer(ElemT, ElemQT, E))
        return false;
      if (!this->emitInitElem(ElemT, 1, E))
        return false;
    } else {
      if (!this->visit(LHS))
        return false;
    }

    if (!this->visit(RHS))
      return false;
    return this->emitDivc(ElemT, E);
  }

  // Evaluate LHS and save value to LHSOffset.
  if (LHSType->isAnyComplexType()) {
    LHSOffset = this->allocateLocalPrimitive(LHS, PT_Ptr, /*IsConst=*/true);
    if (!this->visit(LHS))
      return false;
    if (!this->emitSetLocal(PT_Ptr, LHSOffset, E))
      return false;
  } else {
    PrimType LHST = classifyPrim(LHSType);
    LHSOffset = this->allocateLocalPrimitive(LHS, LHST, /*IsConst=*/true);
    if (!this->visit(LHS))
      return false;
    if (!this->emitSetLocal(LHST, LHSOffset, E))
      return false;
  }

  // Same with RHS.
  unsigned RHSOffset;
  if (RHSType->isAnyComplexType()) {
    RHSOffset = this->allocateLocalPrimitive(RHS, PT_Ptr, /*IsConst=*/true);
    if (!this->visit(RHS))
      return false;
    if (!this->emitSetLocal(PT_Ptr, RHSOffset, E))
      return false;
  } else {
    PrimType RHST = classifyPrim(RHSType);
    RHSOffset = this->allocateLocalPrimitive(RHS, RHST, /*IsConst=*/true);
    if (!this->visit(RHS))
      return false;
    if (!this->emitSetLocal(RHST, RHSOffset, E))
      return false;
  }

  // For both LHS and RHS, either load the value from the complex pointer, or
  // directly from the local variable. For index 1 (i.e. the imaginary part),
  // just load 0 and do the operation anyway.
  auto loadComplexValue = [this](bool IsComplex, bool LoadZero,
                                 unsigned ElemIndex, unsigned Offset,
                                 const Expr *E) -> bool {
    if (IsComplex) {
      if (!this->emitGetLocal(PT_Ptr, Offset, E))
        return false;
      return this->emitArrayElemPop(classifyComplexElementType(E->getType()),
                                    ElemIndex, E);
    }
    if (ElemIndex == 0 || !LoadZero)
      return this->emitGetLocal(classifyPrim(E->getType()), Offset, E);
    return this->visitZeroInitializer(classifyPrim(E->getType()), E->getType(),
                                      E);
  };

  // Now we can get pointers to the LHS and RHS from the offsets above.
  for (unsigned ElemIndex = 0; ElemIndex != 2; ++ElemIndex) {
    // Result pointer for the store later.
    if (!this->DiscardResult) {
      if (!this->emitGetLocal(PT_Ptr, ResultOffset, E))
        return false;
    }

    // The actual operation.
    switch (Op) {
    case BO_Add:
      if (!loadComplexValue(LHSIsComplex, true, ElemIndex, LHSOffset, LHS))
        return false;

      if (!loadComplexValue(RHSIsComplex, true, ElemIndex, RHSOffset, RHS))
        return false;
      if (ResultElemT == PT_Float) {
        if (!this->emitAddf(getFPOptions(E), E))
          return false;
      } else {
        if (!this->emitAdd(ResultElemT, E))
          return false;
      }
      break;
    case BO_Sub:
      if (!loadComplexValue(LHSIsComplex, true, ElemIndex, LHSOffset, LHS))
        return false;

      if (!loadComplexValue(RHSIsComplex, true, ElemIndex, RHSOffset, RHS))
        return false;
      if (ResultElemT == PT_Float) {
        if (!this->emitSubf(getFPOptions(E), E))
          return false;
      } else {
        if (!this->emitSub(ResultElemT, E))
          return false;
      }
      break;
    case BO_Mul:
      if (!loadComplexValue(LHSIsComplex, false, ElemIndex, LHSOffset, LHS))
        return false;

      if (!loadComplexValue(RHSIsComplex, false, ElemIndex, RHSOffset, RHS))
        return false;

      if (ResultElemT == PT_Float) {
        if (!this->emitMulf(getFPOptions(E), E))
          return false;
      } else {
        if (!this->emitMul(ResultElemT, E))
          return false;
      }
      break;
    case BO_Div:
      assert(!RHSIsComplex);
      if (!loadComplexValue(LHSIsComplex, false, ElemIndex, LHSOffset, LHS))
        return false;

      if (!loadComplexValue(RHSIsComplex, false, ElemIndex, RHSOffset, RHS))
        return false;

      if (ResultElemT == PT_Float) {
        if (!this->emitDivf(getFPOptions(E), E))
          return false;
      } else {
        if (!this->emitDiv(ResultElemT, E))
          return false;
      }
      break;

    default:
      return false;
    }

    if (!this->DiscardResult) {
      // Initialize array element with the value we just computed.
      if (!this->emitInitElemPop(ResultElemT, ElemIndex, E))
        return false;
    } else {
      if (!this->emitPop(ResultElemT, E))
        return false;
    }
  }
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitVectorBinOp(const BinaryOperator *E) {
  assert(!E->isCommaOp() &&
         "Comma op should be handled in VisitBinaryOperator");
  assert(E->getType()->isVectorType());
  assert(E->getLHS()->getType()->isVectorType());
  assert(E->getRHS()->getType()->isVectorType());

  // Prepare storage for result.
  if (!Initializing && !E->isCompoundAssignmentOp()) {
    std::optional<unsigned> LocalIndex = allocateTemporary(E);
    if (!LocalIndex)
      return false;
    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
  }

  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  const auto *VecTy = E->getType()->getAs<VectorType>();
  auto Op = E->isCompoundAssignmentOp()
                ? BinaryOperator::getOpForCompoundAssignment(E->getOpcode())
                : E->getOpcode();

  PrimType ElemT = this->classifyVectorElementType(LHS->getType());
  PrimType RHSElemT = this->classifyVectorElementType(RHS->getType());
  PrimType ResultElemT = this->classifyVectorElementType(E->getType());

  // Evaluate LHS and save value to LHSOffset.
  unsigned LHSOffset =
      this->allocateLocalPrimitive(LHS, PT_Ptr, /*IsConst=*/true);
  if (!this->visit(LHS))
    return false;
  if (!this->emitSetLocal(PT_Ptr, LHSOffset, E))
    return false;

  // Evaluate RHS and save value to RHSOffset.
  unsigned RHSOffset =
      this->allocateLocalPrimitive(RHS, PT_Ptr, /*IsConst=*/true);
  if (!this->visit(RHS))
    return false;
  if (!this->emitSetLocal(PT_Ptr, RHSOffset, E))
    return false;

  if (E->isCompoundAssignmentOp() && !this->emitGetLocal(PT_Ptr, LHSOffset, E))
    return false;

  // BitAdd/BitOr/BitXor/Shl/Shr doesn't support bool type, we need perform the
  // integer promotion.
  bool NeedIntPromot = ElemT == PT_Bool && (E->isBitwiseOp() || E->isShiftOp());
  QualType PromotTy =
      Ctx.getASTContext().getPromotedIntegerType(Ctx.getASTContext().BoolTy);
  PrimType PromotT = classifyPrim(PromotTy);
  PrimType OpT = NeedIntPromot ? PromotT : ElemT;

  auto getElem = [=](unsigned Offset, PrimType ElemT, unsigned Index) {
    if (!this->emitGetLocal(PT_Ptr, Offset, E))
      return false;
    if (!this->emitArrayElemPop(ElemT, Index, E))
      return false;
    if (E->isLogicalOp()) {
      if (!this->emitPrimCast(ElemT, PT_Bool, Ctx.getASTContext().BoolTy, E))
        return false;
      if (!this->emitPrimCast(PT_Bool, ResultElemT, VecTy->getElementType(), E))
        return false;
    } else if (NeedIntPromot) {
      if (!this->emitPrimCast(ElemT, PromotT, PromotTy, E))
        return false;
    }
    return true;
  };

#define EMIT_ARITH_OP(OP)                                                      \
  {                                                                            \
    if (ElemT == PT_Float) {                                                   \
      if (!this->emit##OP##f(getFPOptions(E), E))                              \
        return false;                                                          \
    } else {                                                                   \
      if (!this->emit##OP(ElemT, E))                                           \
        return false;                                                          \
    }                                                                          \
    break;                                                                     \
  }

  for (unsigned I = 0; I != VecTy->getNumElements(); ++I) {
    if (!getElem(LHSOffset, ElemT, I))
      return false;
    if (!getElem(RHSOffset, RHSElemT, I))
      return false;
    switch (Op) {
    case BO_Add:
      EMIT_ARITH_OP(Add)
    case BO_Sub:
      EMIT_ARITH_OP(Sub)
    case BO_Mul:
      EMIT_ARITH_OP(Mul)
    case BO_Div:
      EMIT_ARITH_OP(Div)
    case BO_Rem:
      if (!this->emitRem(ElemT, E))
        return false;
      break;
    case BO_And:
      if (!this->emitBitAnd(OpT, E))
        return false;
      break;
    case BO_Or:
      if (!this->emitBitOr(OpT, E))
        return false;
      break;
    case BO_Xor:
      if (!this->emitBitXor(OpT, E))
        return false;
      break;
    case BO_Shl:
      if (!this->emitShl(OpT, RHSElemT, E))
        return false;
      break;
    case BO_Shr:
      if (!this->emitShr(OpT, RHSElemT, E))
        return false;
      break;
    case BO_EQ:
      if (!this->emitEQ(ElemT, E))
        return false;
      break;
    case BO_NE:
      if (!this->emitNE(ElemT, E))
        return false;
      break;
    case BO_LE:
      if (!this->emitLE(ElemT, E))
        return false;
      break;
    case BO_LT:
      if (!this->emitLT(ElemT, E))
        return false;
      break;
    case BO_GE:
      if (!this->emitGE(ElemT, E))
        return false;
      break;
    case BO_GT:
      if (!this->emitGT(ElemT, E))
        return false;
      break;
    case BO_LAnd:
      // a && b is equivalent to a!=0 & b!=0
      if (!this->emitBitAnd(ResultElemT, E))
        return false;
      break;
    case BO_LOr:
      // a || b is equivalent to a!=0 | b!=0
      if (!this->emitBitOr(ResultElemT, E))
        return false;
      break;
    default:
      return this->emitInvalid(E);
    }

    // The result of the comparison is a vector of the same width and number
    // of elements as the comparison operands with a signed integral element
    // type.
    //
    // https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html
    if (E->isComparisonOp()) {
      if (!this->emitPrimCast(PT_Bool, ResultElemT, VecTy->getElementType(), E))
        return false;
      if (!this->emitNeg(ResultElemT, E))
        return false;
    }

    // If we performed an integer promotion, we need to cast the compute result
    // into result vector element type.
    if (NeedIntPromot &&
        !this->emitPrimCast(PromotT, ResultElemT, VecTy->getElementType(), E))
      return false;

    // Initialize array element with the value we just computed.
    if (!this->emitInitElem(ResultElemT, I, E))
      return false;
  }

  if (DiscardResult && E->isCompoundAssignmentOp() && !this->emitPopPtr(E))
    return false;
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitFixedPointBinOp(const BinaryOperator *E) {
  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  const ASTContext &ASTCtx = Ctx.getASTContext();

  assert(LHS->getType()->isFixedPointType() ||
         RHS->getType()->isFixedPointType());

  auto LHSSema = ASTCtx.getFixedPointSemantics(LHS->getType());
  auto LHSSemaInt = LHSSema.toOpaqueInt();
  auto RHSSema = ASTCtx.getFixedPointSemantics(RHS->getType());
  auto RHSSemaInt = RHSSema.toOpaqueInt();

  if (!this->visit(LHS))
    return false;
  if (!LHS->getType()->isFixedPointType()) {
    if (!this->emitCastIntegralFixedPoint(classifyPrim(LHS->getType()),
                                          LHSSemaInt, E))
      return false;
  }

  if (!this->visit(RHS))
    return false;
  if (!RHS->getType()->isFixedPointType()) {
    if (!this->emitCastIntegralFixedPoint(classifyPrim(RHS->getType()),
                                          RHSSemaInt, E))
      return false;
  }

  // Convert the result to the target semantics.
  auto ConvertResult = [&](bool R) -> bool {
    if (!R)
      return false;
    auto ResultSema = ASTCtx.getFixedPointSemantics(E->getType()).toOpaqueInt();
    auto CommonSema = LHSSema.getCommonSemantics(RHSSema).toOpaqueInt();
    if (ResultSema != CommonSema)
      return this->emitCastFixedPoint(ResultSema, E);
    return true;
  };

  auto MaybeCastToBool = [&](bool Result) {
    if (!Result)
      return false;
    PrimType T = classifyPrim(E);
    if (DiscardResult)
      return this->emitPop(T, E);
    if (T != PT_Bool)
      return this->emitCast(PT_Bool, T, E);
    return true;
  };

  switch (E->getOpcode()) {
  case BO_EQ:
    return MaybeCastToBool(this->emitEQFixedPoint(E));
  case BO_NE:
    return MaybeCastToBool(this->emitNEFixedPoint(E));
  case BO_LT:
    return MaybeCastToBool(this->emitLTFixedPoint(E));
  case BO_LE:
    return MaybeCastToBool(this->emitLEFixedPoint(E));
  case BO_GT:
    return MaybeCastToBool(this->emitGTFixedPoint(E));
  case BO_GE:
    return MaybeCastToBool(this->emitGEFixedPoint(E));
  case BO_Add:
    return ConvertResult(this->emitAddFixedPoint(E));
  case BO_Sub:
    return ConvertResult(this->emitSubFixedPoint(E));
  case BO_Mul:
    return ConvertResult(this->emitMulFixedPoint(E));
  case BO_Div:
    return ConvertResult(this->emitDivFixedPoint(E));
  case BO_Shl:
    return ConvertResult(this->emitShiftFixedPoint(/*Left=*/true, E));
  case BO_Shr:
    return ConvertResult(this->emitShiftFixedPoint(/*Left=*/false, E));

  default:
    return this->emitInvalid(E);
  }

  llvm_unreachable("unhandled binop opcode");
}

template <class Emitter>
bool Compiler<Emitter>::VisitFixedPointUnaryOperator(const UnaryOperator *E) {
  const Expr *SubExpr = E->getSubExpr();
  assert(SubExpr->getType()->isFixedPointType());

  switch (E->getOpcode()) {
  case UO_Plus:
    return this->delegate(SubExpr);
  case UO_Minus:
    if (!this->visit(SubExpr))
      return false;
    return this->emitNegFixedPoint(E);
  default:
    return false;
  }

  llvm_unreachable("Unhandled unary opcode");
}

template <class Emitter>
bool Compiler<Emitter>::VisitImplicitValueInitExpr(
    const ImplicitValueInitExpr *E) {
  QualType QT = E->getType();

  if (std::optional<PrimType> T = classify(QT))
    return this->visitZeroInitializer(*T, QT, E);

  if (QT->isRecordType()) {
    const RecordDecl *RD = QT->getAsRecordDecl();
    assert(RD);
    if (RD->isInvalidDecl())
      return false;

    if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD);
        CXXRD && CXXRD->getNumVBases() > 0) {
      // TODO: Diagnose.
      return false;
    }

    const Record *R = getRecord(QT);
    if (!R)
      return false;

    assert(Initializing);
    return this->visitZeroRecordInitializer(R, E);
  }

  if (QT->isIncompleteArrayType())
    return true;

  if (QT->isArrayType())
    return this->visitZeroArrayInitializer(QT, E);

  if (const auto *ComplexTy = E->getType()->getAs<ComplexType>()) {
    assert(Initializing);
    QualType ElemQT = ComplexTy->getElementType();
    PrimType ElemT = classifyPrim(ElemQT);
    for (unsigned I = 0; I < 2; ++I) {
      if (!this->visitZeroInitializer(ElemT, ElemQT, E))
        return false;
      if (!this->emitInitElem(ElemT, I, E))
        return false;
    }
    return true;
  }

  if (const auto *VecT = E->getType()->getAs<VectorType>()) {
    unsigned NumVecElements = VecT->getNumElements();
    QualType ElemQT = VecT->getElementType();
    PrimType ElemT = classifyPrim(ElemQT);

    for (unsigned I = 0; I < NumVecElements; ++I) {
      if (!this->visitZeroInitializer(ElemT, ElemQT, E))
        return false;
      if (!this->emitInitElem(ElemT, I, E))
        return false;
    }
    return true;
  }

  return false;
}

template <class Emitter>
bool Compiler<Emitter>::VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  const Expr *Index = E->getIdx();
  const Expr *Base = E->getBase();

  // C++17's rules require us to evaluate the LHS first, regardless of which
  // side is the base.
  bool Success = true;
  for (const Expr *SubExpr : {LHS, RHS}) {
    if (!this->visit(SubExpr)) {
      Success = false;
      continue;
    }

    // Expand the base if this is a subscript on a
    // pointer expression.
    if (SubExpr == Base && Base->getType()->isPointerType()) {
      if (!this->emitExpandPtr(E))
        Success = false;
    }
  }

  if (!Success)
    return false;

  std::optional<PrimType> IndexT = classify(Index->getType());
  // In error-recovery cases, the index expression has a dependent type.
  if (!IndexT)
    return this->emitError(E);
  // If the index is first, we need to change that.
  if (LHS == Index) {
    if (!this->emitFlip(PT_Ptr, *IndexT, E))
      return false;
  }

  if (!this->emitArrayElemPtrPop(*IndexT, E))
    return false;
  if (DiscardResult)
    return this->emitPopPtr(E);
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::visitInitList(ArrayRef<const Expr *> Inits,
                                      const Expr *ArrayFiller, const Expr *E) {
  InitLinkScope<Emitter> ILS(this, InitLink::InitList());

  QualType QT = E->getType();
  if (const auto *AT = QT->getAs<AtomicType>())
    QT = AT->getValueType();

  if (QT->isVoidType()) {
    if (Inits.size() == 0)
      return true;
    return this->emitInvalid(E);
  }

  // Handle discarding first.
  if (DiscardResult) {
    for (const Expr *Init : Inits) {
      if (!this->discard(Init))
        return false;
    }
    return true;
  }

  // Primitive values.
  if (std::optional<PrimType> T = classify(QT)) {
    assert(!DiscardResult);
    if (Inits.size() == 0)
      return this->visitZeroInitializer(*T, QT, E);
    assert(Inits.size() == 1);
    return this->delegate(Inits[0]);
  }

  if (QT->isRecordType()) {
    const Record *R = getRecord(QT);

    if (Inits.size() == 1 && E->getType() == Inits[0]->getType())
      return this->delegate(Inits[0]);

    auto initPrimitiveField = [=](const Record::Field *FieldToInit,
                                  const Expr *Init, PrimType T) -> bool {
      InitStackScope<Emitter> ISS(this, isa<CXXDefaultInitExpr>(Init));
      InitLinkScope<Emitter> ILS(this, InitLink::Field(FieldToInit->Offset));
      if (!this->visit(Init))
        return false;

      if (FieldToInit->isBitField())
        return this->emitInitBitField(T, FieldToInit, E);
      return this->emitInitField(T, FieldToInit->Offset, E);
    };

    auto initCompositeField = [=](const Record::Field *FieldToInit,
                                  const Expr *Init) -> bool {
      InitStackScope<Emitter> ISS(this, isa<CXXDefaultInitExpr>(Init));
      InitLinkScope<Emitter> ILS(this, InitLink::Field(FieldToInit->Offset));

      // Non-primitive case. Get a pointer to the field-to-initialize
      // on the stack and recurse into visitInitializer().
      if (!this->emitGetPtrField(FieldToInit->Offset, Init))
        return false;
      if (!this->visitInitializer(Init))
        return false;
      return this->emitPopPtr(E);
    };

    if (R->isUnion()) {
      if (Inits.size() == 0) {
        if (!this->visitZeroRecordInitializer(R, E))
          return false;
      } else {
        const Expr *Init = Inits[0];
        const FieldDecl *FToInit = nullptr;
        if (const auto *ILE = dyn_cast<InitListExpr>(E))
          FToInit = ILE->getInitializedFieldInUnion();
        else
          FToInit = cast<CXXParenListInitExpr>(E)->getInitializedFieldInUnion();

        const Record::Field *FieldToInit = R->getField(FToInit);
        if (std::optional<PrimType> T = classify(Init)) {
          if (!initPrimitiveField(FieldToInit, Init, *T))
            return false;
        } else {
          if (!initCompositeField(FieldToInit, Init))
            return false;
        }
      }
      return this->emitFinishInit(E);
    }

    assert(!R->isUnion());
    unsigned InitIndex = 0;
    for (const Expr *Init : Inits) {
      // Skip unnamed bitfields.
      while (InitIndex < R->getNumFields() &&
             R->getField(InitIndex)->isUnnamedBitField())
        ++InitIndex;

      if (std::optional<PrimType> T = classify(Init)) {
        const Record::Field *FieldToInit = R->getField(InitIndex);
        if (!initPrimitiveField(FieldToInit, Init, *T))
          return false;
        ++InitIndex;
      } else {
        // Initializer for a direct base class.
        if (const Record::Base *B = R->getBase(Init->getType())) {
          if (!this->emitGetPtrBase(B->Offset, Init))
            return false;

          if (!this->visitInitializer(Init))
            return false;

          if (!this->emitFinishInitPop(E))
            return false;
          // Base initializers don't increase InitIndex, since they don't count
          // into the Record's fields.
        } else {
          const Record::Field *FieldToInit = R->getField(InitIndex);
          if (!initCompositeField(FieldToInit, Init))
            return false;
          ++InitIndex;
        }
      }
    }
    return this->emitFinishInit(E);
  }

  if (QT->isArrayType()) {
    if (Inits.size() == 1 && QT == Inits[0]->getType())
      return this->delegate(Inits[0]);

    const ConstantArrayType *CAT =
        Ctx.getASTContext().getAsConstantArrayType(QT);
    uint64_t NumElems = CAT->getZExtSize();

    if (!this->emitCheckArraySize(NumElems, E))
      return false;

    std::optional<PrimType> InitT = classify(CAT->getElementType());
    unsigned ElementIndex = 0;
    for (const Expr *Init : Inits) {
      if (const auto *EmbedS =
              dyn_cast<EmbedExpr>(Init->IgnoreParenImpCasts())) {
        PrimType TargetT = classifyPrim(Init->getType());

        auto Eval = [&](const Expr *Init, unsigned ElemIndex) {
          PrimType InitT = classifyPrim(Init->getType());
          if (!this->visit(Init))
            return false;
          if (InitT != TargetT) {
            if (!this->emitCast(InitT, TargetT, E))
              return false;
          }
          return this->emitInitElem(TargetT, ElemIndex, Init);
        };
        if (!EmbedS->doForEachDataElement(Eval, ElementIndex))
          return false;
      } else {
        if (!this->visitArrayElemInit(ElementIndex, Init, InitT))
          return false;
        ++ElementIndex;
      }
    }

    // Expand the filler expression.
    // FIXME: This should go away.
    if (ArrayFiller) {
      for (; ElementIndex != NumElems; ++ElementIndex) {
        if (!this->visitArrayElemInit(ElementIndex, ArrayFiller, InitT))
          return false;
      }
    }

    return this->emitFinishInit(E);
  }

  if (const auto *ComplexTy = QT->getAs<ComplexType>()) {
    unsigned NumInits = Inits.size();

    if (NumInits == 1)
      return this->delegate(Inits[0]);

    QualType ElemQT = ComplexTy->getElementType();
    PrimType ElemT = classifyPrim(ElemQT);
    if (NumInits == 0) {
      // Zero-initialize both elements.
      for (unsigned I = 0; I < 2; ++I) {
        if (!this->visitZeroInitializer(ElemT, ElemQT, E))
          return false;
        if (!this->emitInitElem(ElemT, I, E))
          return false;
      }
    } else if (NumInits == 2) {
      unsigned InitIndex = 0;
      for (const Expr *Init : Inits) {
        if (!this->visit(Init))
          return false;

        if (!this->emitInitElem(ElemT, InitIndex, E))
          return false;
        ++InitIndex;
      }
    }
    return true;
  }

  if (const auto *VecT = QT->getAs<VectorType>()) {
    unsigned NumVecElements = VecT->getNumElements();
    assert(NumVecElements >= Inits.size());

    QualType ElemQT = VecT->getElementType();
    PrimType ElemT = classifyPrim(ElemQT);

    // All initializer elements.
    unsigned InitIndex = 0;
    for (const Expr *Init : Inits) {
      if (!this->visit(Init))
        return false;

      // If the initializer is of vector type itself, we have to deconstruct
      // that and initialize all the target fields from the initializer fields.
      if (const auto *InitVecT = Init->getType()->getAs<VectorType>()) {
        if (!this->emitCopyArray(ElemT, 0, InitIndex,
                                 InitVecT->getNumElements(), E))
          return false;
        InitIndex += InitVecT->getNumElements();
      } else {
        if (!this->emitInitElem(ElemT, InitIndex, E))
          return false;
        ++InitIndex;
      }
    }

    assert(InitIndex <= NumVecElements);

    // Fill the rest with zeroes.
    for (; InitIndex != NumVecElements; ++InitIndex) {
      if (!this->visitZeroInitializer(ElemT, ElemQT, E))
        return false;
      if (!this->emitInitElem(ElemT, InitIndex, E))
        return false;
    }
    return true;
  }

  return false;
}

/// Pointer to the array(not the element!) must be on the stack when calling
/// this.
template <class Emitter>
bool Compiler<Emitter>::visitArrayElemInit(unsigned ElemIndex, const Expr *Init,
                                           std::optional<PrimType> InitT) {
  if (InitT) {
    // Visit the primitive element like normal.
    if (!this->visit(Init))
      return false;
    return this->emitInitElem(*InitT, ElemIndex, Init);
  }

  InitLinkScope<Emitter> ILS(this, InitLink::Elem(ElemIndex));
  // Advance the pointer currently on the stack to the given
  // dimension.
  if (!this->emitConstUint32(ElemIndex, Init))
    return false;
  if (!this->emitArrayElemPtrUint32(Init))
    return false;
  if (!this->visitInitializer(Init))
    return false;
  return this->emitFinishInitPop(Init);
}

template <class Emitter>
bool Compiler<Emitter>::visitCallArgs(ArrayRef<const Expr *> Args,
                                      const FunctionDecl *FuncDecl) {
  assert(VarScope->getKind() == ScopeKind::Call);
  llvm::BitVector NonNullArgs = collectNonNullArgs(FuncDecl, Args);

  unsigned ArgIndex = 0;
  for (const Expr *Arg : Args) {
    if (std::optional<PrimType> T = classify(Arg)) {
      if (!this->visit(Arg))
        return false;
    } else {

      std::optional<unsigned> LocalIndex = allocateLocal(
          Arg, Arg->getType(), /*ExtendingDecl=*/nullptr, ScopeKind::Call);
      if (!LocalIndex)
        return false;

      if (!this->emitGetPtrLocal(*LocalIndex, Arg))
        return false;
      InitLinkScope<Emitter> ILS(this, InitLink::Temp(*LocalIndex));
      if (!this->visitInitializer(Arg))
        return false;
    }

    if (FuncDecl && NonNullArgs[ArgIndex]) {
      PrimType ArgT = classify(Arg).value_or(PT_Ptr);
      if (ArgT == PT_Ptr) {
        if (!this->emitCheckNonNullArg(ArgT, Arg))
          return false;
      }
    }

    ++ArgIndex;
  }

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitInitListExpr(const InitListExpr *E) {
  return this->visitInitList(E->inits(), E->getArrayFiller(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXParenListInitExpr(
    const CXXParenListInitExpr *E) {
  return this->visitInitList(E->getInitExprs(), E->getArrayFiller(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitSubstNonTypeTemplateParmExpr(
    const SubstNonTypeTemplateParmExpr *E) {
  return this->delegate(E->getReplacement());
}

template <class Emitter>
bool Compiler<Emitter>::VisitConstantExpr(const ConstantExpr *E) {
  std::optional<PrimType> T = classify(E->getType());
  if (T && E->hasAPValueResult()) {
    // Try to emit the APValue directly, without visiting the subexpr.
    // This will only fail if we can't emit the APValue, so won't emit any
    // diagnostics or any double values.
    if (DiscardResult)
      return true;

    if (this->visitAPValue(E->getAPValueResult(), *T, E))
      return true;
  }
  return this->delegate(E->getSubExpr());
}

template <class Emitter>
bool Compiler<Emitter>::VisitEmbedExpr(const EmbedExpr *E) {
  auto It = E->begin();
  return this->visit(*It);
}

static CharUnits AlignOfType(QualType T, const ASTContext &ASTCtx,
                             UnaryExprOrTypeTrait Kind) {
  bool AlignOfReturnsPreferred =
      ASTCtx.getLangOpts().getClangABICompat() <= LangOptions::ClangABI::Ver7;

  // C++ [expr.alignof]p3:
  //     When alignof is applied to a reference type, the result is the
  //     alignment of the referenced type.
  if (const auto *Ref = T->getAs<ReferenceType>())
    T = Ref->getPointeeType();

  if (T.getQualifiers().hasUnaligned())
    return CharUnits::One();

  // __alignof is defined to return the preferred alignment.
  // Before 8, clang returned the preferred alignment for alignof and
  // _Alignof as well.
  if (Kind == UETT_PreferredAlignOf || AlignOfReturnsPreferred)
    return ASTCtx.toCharUnitsFromBits(ASTCtx.getPreferredTypeAlign(T));

  return ASTCtx.getTypeAlignInChars(T);
}

template <class Emitter>
bool Compiler<Emitter>::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *E) {
  UnaryExprOrTypeTrait Kind = E->getKind();
  const ASTContext &ASTCtx = Ctx.getASTContext();

  if (Kind == UETT_SizeOf || Kind == UETT_DataSizeOf) {
    QualType ArgType = E->getTypeOfArgument();

    // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
    //   the result is the size of the referenced type."
    if (const auto *Ref = ArgType->getAs<ReferenceType>())
      ArgType = Ref->getPointeeType();

    CharUnits Size;
    if (ArgType->isVoidType() || ArgType->isFunctionType())
      Size = CharUnits::One();
    else {
      if (ArgType->isDependentType() || !ArgType->isConstantSizeType())
        return this->emitInvalid(E);

      if (Kind == UETT_SizeOf)
        Size = ASTCtx.getTypeSizeInChars(ArgType);
      else
        Size = ASTCtx.getTypeInfoDataSizeInChars(ArgType).Width;
    }

    if (DiscardResult)
      return true;

    return this->emitConst(Size.getQuantity(), E);
  }

  if (Kind == UETT_CountOf) {
    QualType Ty = E->getTypeOfArgument();
    assert(Ty->isArrayType());

    // We don't need to worry about array element qualifiers, so getting the
    // unsafe array type is fine.
    if (const auto *CAT =
            dyn_cast<ConstantArrayType>(Ty->getAsArrayTypeUnsafe())) {
      if (DiscardResult)
        return true;
      return this->emitConst(CAT->getSize(), E);
    }

    assert(!Ty->isConstantSizeType());

    // If it's a variable-length array type, we need to check whether it is a
    // multidimensional array. If so, we need to check the size expression of
    // the VLA to see if it's a constant size. If so, we can return that value.
    const auto *VAT = ASTCtx.getAsVariableArrayType(Ty);
    assert(VAT);
    if (VAT->getElementType()->isArrayType()) {
      std::optional<APSInt> Res =
          VAT->getSizeExpr()->getIntegerConstantExpr(ASTCtx);
      if (Res) {
        if (DiscardResult)
          return true;
        return this->emitConst(*Res, E);
      }
    }
  }

  if (Kind == UETT_AlignOf || Kind == UETT_PreferredAlignOf) {
    CharUnits Size;

    if (E->isArgumentType()) {
      QualType ArgType = E->getTypeOfArgument();

      Size = AlignOfType(ArgType, ASTCtx, Kind);
    } else {
      // Argument is an expression, not a type.
      const Expr *Arg = E->getArgumentExpr()->IgnoreParens();

      // The kinds of expressions that we have special-case logic here for
      // should be kept up to date with the special checks for those
      // expressions in Sema.

      // alignof decl is always accepted, even if it doesn't make sense: we
      // default to 1 in those cases.
      if (const auto *DRE = dyn_cast<DeclRefExpr>(Arg))
        Size = ASTCtx.getDeclAlign(DRE->getDecl(),
                                   /*RefAsPointee*/ true);
      else if (const auto *ME = dyn_cast<MemberExpr>(Arg))
        Size = ASTCtx.getDeclAlign(ME->getMemberDecl(),
                                   /*RefAsPointee*/ true);
      else
        Size = AlignOfType(Arg->getType(), ASTCtx, Kind);
    }

    if (DiscardResult)
      return true;

    return this->emitConst(Size.getQuantity(), E);
  }

  if (Kind == UETT_VectorElements) {
    if (const auto *VT = E->getTypeOfArgument()->getAs<VectorType>())
      return this->emitConst(VT->getNumElements(), E);
    assert(E->getTypeOfArgument()->isSizelessVectorType());
    return this->emitSizelessVectorElementSize(E);
  }

  if (Kind == UETT_VecStep) {
    if (const auto *VT = E->getTypeOfArgument()->getAs<VectorType>()) {
      unsigned N = VT->getNumElements();

      // The vec_step built-in functions that take a 3-component
      // vector return 4. (OpenCL 1.1 spec 6.11.12)
      if (N == 3)
        N = 4;

      return this->emitConst(N, E);
    }
    return this->emitConst(1, E);
  }

  if (Kind == UETT_OpenMPRequiredSimdAlign) {
    assert(E->isArgumentType());
    unsigned Bits = ASTCtx.getOpenMPDefaultSimdAlign(E->getArgumentType());

    return this->emitConst(ASTCtx.toCharUnitsFromBits(Bits).getQuantity(), E);
  }

  if (Kind == UETT_PtrAuthTypeDiscriminator) {
    if (E->getArgumentType()->isDependentType())
      return this->emitInvalid(E);

    return this->emitConst(
        const_cast<ASTContext &>(ASTCtx).getPointerAuthTypeDiscriminator(
            E->getArgumentType()),
        E);
  }

  return false;
}

template <class Emitter>
bool Compiler<Emitter>::VisitMemberExpr(const MemberExpr *E) {
  // 'Base.Member'
  const Expr *Base = E->getBase();
  const ValueDecl *Member = E->getMemberDecl();

  if (DiscardResult)
    return this->discard(Base);

  // MemberExprs are almost always lvalues, in which case we don't need to
  // do the load. But sometimes they aren't.
  const auto maybeLoadValue = [&]() -> bool {
    if (E->isGLValue())
      return true;
    if (std::optional<PrimType> T = classify(E))
      return this->emitLoadPop(*T, E);
    return false;
  };

  if (const auto *VD = dyn_cast<VarDecl>(Member)) {
    // I am almost confident in saying that a var decl must be static
    // and therefore registered as a global variable. But this will probably
    // turn out to be wrong some time in the future, as always.
    if (auto GlobalIndex = P.getGlobal(VD))
      return this->emitGetPtrGlobal(*GlobalIndex, E) && maybeLoadValue();
    return false;
  }

  if (!isa<FieldDecl>(Member)) {
    if (!this->discard(Base) && !this->emitSideEffect(E))
      return false;

    return this->visitDeclRef(Member, E);
  }

  if (Initializing) {
    if (!this->delegate(Base))
      return false;
  } else {
    if (!this->visit(Base))
      return false;
  }

  // Base above gives us a pointer on the stack.
  const auto *FD = cast<FieldDecl>(Member);
  const RecordDecl *RD = FD->getParent();
  const Record *R = getRecord(RD);
  if (!R)
    return false;
  const Record::Field *F = R->getField(FD);
  // Leave a pointer to the field on the stack.
  if (F->Decl->getType()->isReferenceType())
    return this->emitGetFieldPop(PT_Ptr, F->Offset, E) && maybeLoadValue();
  return this->emitGetPtrFieldPop(F->Offset, E) && maybeLoadValue();
}

template <class Emitter>
bool Compiler<Emitter>::VisitArrayInitIndexExpr(const ArrayInitIndexExpr *E) {
  // ArrayIndex might not be set if a ArrayInitIndexExpr is being evaluated
  // stand-alone, e.g. via EvaluateAsInt().
  if (!ArrayIndex)
    return false;
  return this->emitConst(*ArrayIndex, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitArrayInitLoopExpr(const ArrayInitLoopExpr *E) {
  assert(Initializing);
  assert(!DiscardResult);

  // We visit the common opaque expression here once so we have its value
  // cached.
  if (!this->discard(E->getCommonExpr()))
    return false;

  // TODO: This compiles to quite a lot of bytecode if the array is larger.
  //   Investigate compiling this to a loop.
  const Expr *SubExpr = E->getSubExpr();
  size_t Size = E->getArraySize().getZExtValue();
  std::optional<PrimType> SubExprT = classify(SubExpr);

  // So, every iteration, we execute an assignment here
  // where the LHS is on the stack (the target array)
  // and the RHS is our SubExpr.
  for (size_t I = 0; I != Size; ++I) {
    ArrayIndexScope<Emitter> IndexScope(this, I);
    BlockScope<Emitter> BS(this);

    if (!this->visitArrayElemInit(I, SubExpr, SubExprT))
      return false;
    if (!BS.destroyLocals())
      return false;
  }
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitOpaqueValueExpr(const OpaqueValueExpr *E) {
  const Expr *SourceExpr = E->getSourceExpr();
  if (!SourceExpr)
    return false;

  if (Initializing)
    return this->visitInitializer(SourceExpr);

  PrimType SubExprT = classify(SourceExpr).value_or(PT_Ptr);
  if (auto It = OpaqueExprs.find(E); It != OpaqueExprs.end())
    return this->emitGetLocal(SubExprT, It->second, E);

  if (!this->visit(SourceExpr))
    return false;

  // At this point we either have the evaluated source expression or a pointer
  // to an object on the stack. We want to create a local variable that stores
  // this value.
  unsigned LocalIndex = allocateLocalPrimitive(E, SubExprT, /*IsConst=*/true);
  if (!this->emitSetLocal(SubExprT, LocalIndex, E))
    return false;

  // Here the local variable is created but the value is removed from the stack,
  // so we put it back if the caller needs it.
  if (!DiscardResult) {
    if (!this->emitGetLocal(SubExprT, LocalIndex, E))
      return false;
  }

  // This is cleaned up when the local variable is destroyed.
  OpaqueExprs.insert({E, LocalIndex});

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *E) {
  const Expr *Condition = E->getCond();
  const Expr *TrueExpr = E->getTrueExpr();
  const Expr *FalseExpr = E->getFalseExpr();

  auto visitChildExpr = [&](const Expr *E) -> bool {
    LocalScope<Emitter> S(this);
    if (!this->delegate(E))
      return false;
    return S.destroyLocals();
  };

  if (std::optional<bool> BoolValue = getBoolValue(Condition)) {
    if (BoolValue)
      return visitChildExpr(TrueExpr);
    return visitChildExpr(FalseExpr);
  }

  bool IsBcpCall = false;
  if (const auto *CE = dyn_cast<CallExpr>(Condition->IgnoreParenCasts());
      CE && CE->getBuiltinCallee() == Builtin::BI__builtin_constant_p) {
    IsBcpCall = true;
  }

  LabelTy LabelEnd = this->getLabel();   // Label after the operator.
  LabelTy LabelFalse = this->getLabel(); // Label for the false expr.

  if (IsBcpCall) {
    if (!this->emitStartSpeculation(E))
      return false;
  }

  if (!this->visitBool(Condition)) {
    // If the condition failed and we're checking for undefined behavior
    // (which only happens with EvalEmitter) check the TrueExpr and FalseExpr
    // as well.
    if (this->checkingForUndefinedBehavior()) {
      if (!this->discard(TrueExpr))
        return false;
      if (!this->discard(FalseExpr))
        return false;
    }
    return false;
  }

  if (!this->jumpFalse(LabelFalse))
    return false;
  if (!visitChildExpr(TrueExpr))
    return false;
  if (!this->jump(LabelEnd))
    return false;
  this->emitLabel(LabelFalse);
  if (!visitChildExpr(FalseExpr))
    return false;
  this->fallthrough(LabelEnd);
  this->emitLabel(LabelEnd);

  if (IsBcpCall)
    return this->emitEndSpeculation(E);
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitStringLiteral(const StringLiteral *E) {
  if (DiscardResult)
    return true;

  if (!Initializing) {
    unsigned StringIndex = P.createGlobalString(E);
    return this->emitGetPtrGlobal(StringIndex, E);
  }

  // We are initializing an array on the stack.
  const ConstantArrayType *CAT =
      Ctx.getASTContext().getAsConstantArrayType(E->getType());
  assert(CAT && "a string literal that's not a constant array?");

  // If the initializer string is too long, a diagnostic has already been
  // emitted. Read only the array length from the string literal.
  unsigned ArraySize = CAT->getZExtSize();
  unsigned N = std::min(ArraySize, E->getLength());
  unsigned CharWidth = E->getCharByteWidth();

  for (unsigned I = 0; I != N; ++I) {
    uint32_t CodeUnit = E->getCodeUnit(I);

    if (CharWidth == 1) {
      this->emitConstSint8(CodeUnit, E);
      this->emitInitElemSint8(I, E);
    } else if (CharWidth == 2) {
      this->emitConstUint16(CodeUnit, E);
      this->emitInitElemUint16(I, E);
    } else if (CharWidth == 4) {
      this->emitConstUint32(CodeUnit, E);
      this->emitInitElemUint32(I, E);
    } else {
      llvm_unreachable("unsupported character width");
    }
  }

  // Fill up the rest of the char array with NUL bytes.
  for (unsigned I = N; I != ArraySize; ++I) {
    if (CharWidth == 1) {
      this->emitConstSint8(0, E);
      this->emitInitElemSint8(I, E);
    } else if (CharWidth == 2) {
      this->emitConstUint16(0, E);
      this->emitInitElemUint16(I, E);
    } else if (CharWidth == 4) {
      this->emitConstUint32(0, E);
      this->emitInitElemUint32(I, E);
    } else {
      llvm_unreachable("unsupported character width");
    }
  }

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitObjCStringLiteral(const ObjCStringLiteral *E) {
  if (DiscardResult)
    return true;
  return this->emitDummyPtr(E, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitObjCEncodeExpr(const ObjCEncodeExpr *E) {
  auto &A = Ctx.getASTContext();
  std::string Str;
  A.getObjCEncodingForType(E->getEncodedType(), Str);
  StringLiteral *SL =
      StringLiteral::Create(A, Str, StringLiteralKind::Ordinary,
                            /*Pascal=*/false, E->getType(), E->getAtLoc());
  return this->delegate(SL);
}

template <class Emitter>
bool Compiler<Emitter>::VisitSYCLUniqueStableNameExpr(
    const SYCLUniqueStableNameExpr *E) {
  if (DiscardResult)
    return true;

  assert(!Initializing);

  auto &A = Ctx.getASTContext();
  std::string ResultStr = E->ComputeName(A);

  QualType CharTy = A.CharTy.withConst();
  APInt Size(A.getTypeSize(A.getSizeType()), ResultStr.size() + 1);
  QualType ArrayTy = A.getConstantArrayType(CharTy, Size, nullptr,
                                            ArraySizeModifier::Normal, 0);

  StringLiteral *SL =
      StringLiteral::Create(A, ResultStr, StringLiteralKind::Ordinary,
                            /*Pascal=*/false, ArrayTy, E->getLocation());

  unsigned StringIndex = P.createGlobalString(SL);
  return this->emitGetPtrGlobal(StringIndex, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCharacterLiteral(const CharacterLiteral *E) {
  if (DiscardResult)
    return true;
  return this->emitConst(E->getValue(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitFloatCompoundAssignOperator(
    const CompoundAssignOperator *E) {

  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  QualType LHSType = LHS->getType();
  QualType LHSComputationType = E->getComputationLHSType();
  QualType ResultType = E->getComputationResultType();
  std::optional<PrimType> LT = classify(LHSComputationType);
  std::optional<PrimType> RT = classify(ResultType);

  assert(ResultType->isFloatingType());

  if (!LT || !RT)
    return false;

  PrimType LHST = classifyPrim(LHSType);

  // C++17 onwards require that we evaluate the RHS first.
  // Compute RHS and save it in a temporary variable so we can
  // load it again later.
  if (!visit(RHS))
    return false;

  unsigned TempOffset = this->allocateLocalPrimitive(E, *RT, /*IsConst=*/true);
  if (!this->emitSetLocal(*RT, TempOffset, E))
    return false;

  // First, visit LHS.
  if (!visit(LHS))
    return false;
  if (!this->emitLoad(LHST, E))
    return false;

  // If necessary, convert LHS to its computation type.
  if (!this->emitPrimCast(LHST, classifyPrim(LHSComputationType),
                          LHSComputationType, E))
    return false;

  // Now load RHS.
  if (!this->emitGetLocal(*RT, TempOffset, E))
    return false;

  switch (E->getOpcode()) {
  case BO_AddAssign:
    if (!this->emitAddf(getFPOptions(E), E))
      return false;
    break;
  case BO_SubAssign:
    if (!this->emitSubf(getFPOptions(E), E))
      return false;
    break;
  case BO_MulAssign:
    if (!this->emitMulf(getFPOptions(E), E))
      return false;
    break;
  case BO_DivAssign:
    if (!this->emitDivf(getFPOptions(E), E))
      return false;
    break;
  default:
    return false;
  }

  if (!this->emitPrimCast(classifyPrim(ResultType), LHST, LHS->getType(), E))
    return false;

  if (DiscardResult)
    return this->emitStorePop(LHST, E);
  return this->emitStore(LHST, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitPointerCompoundAssignOperator(
    const CompoundAssignOperator *E) {
  BinaryOperatorKind Op = E->getOpcode();
  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  std::optional<PrimType> LT = classify(LHS->getType());
  std::optional<PrimType> RT = classify(RHS->getType());

  if (Op != BO_AddAssign && Op != BO_SubAssign)
    return false;

  if (!LT || !RT)
    return false;

  if (!visit(LHS))
    return false;

  if (!this->emitLoad(*LT, LHS))
    return false;

  if (!visit(RHS))
    return false;

  if (Op == BO_AddAssign) {
    if (!this->emitAddOffset(*RT, E))
      return false;
  } else {
    if (!this->emitSubOffset(*RT, E))
      return false;
  }

  if (DiscardResult)
    return this->emitStorePopPtr(E);
  return this->emitStorePtr(E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCompoundAssignOperator(
    const CompoundAssignOperator *E) {
  if (E->getType()->isVectorType())
    return VisitVectorBinOp(E);

  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  std::optional<PrimType> LHSComputationT =
      classify(E->getComputationLHSType());
  std::optional<PrimType> LT = classify(LHS->getType());
  std::optional<PrimType> RT = classify(RHS->getType());
  std::optional<PrimType> ResultT = classify(E->getType());

  if (!Ctx.getLangOpts().CPlusPlus14)
    return this->visit(RHS) && this->visit(LHS) && this->emitError(E);

  if (!LT || !RT || !ResultT || !LHSComputationT)
    return false;

  // Handle floating point operations separately here, since they
  // require special care.

  if (ResultT == PT_Float || RT == PT_Float)
    return VisitFloatCompoundAssignOperator(E);

  if (E->getType()->isPointerType())
    return VisitPointerCompoundAssignOperator(E);

  assert(!E->getType()->isPointerType() && "Handled above");
  assert(!E->getType()->isFloatingType() && "Handled above");

  // C++17 onwards require that we evaluate the RHS first.
  // Compute RHS and save it in a temporary variable so we can
  // load it again later.
  // FIXME: Compound assignments are unsequenced in C, so we might
  //   have to figure out how to reject them.
  if (!visit(RHS))
    return false;

  unsigned TempOffset = this->allocateLocalPrimitive(E, *RT, /*IsConst=*/true);

  if (!this->emitSetLocal(*RT, TempOffset, E))
    return false;

  // Get LHS pointer, load its value and cast it to the
  // computation type if necessary.
  if (!visit(LHS))
    return false;
  if (!this->emitLoad(*LT, E))
    return false;
  if (LT != LHSComputationT) {
    if (!this->emitCast(*LT, *LHSComputationT, E))
      return false;
  }

  // Get the RHS value on the stack.
  if (!this->emitGetLocal(*RT, TempOffset, E))
    return false;

  // Perform operation.
  switch (E->getOpcode()) {
  case BO_AddAssign:
    if (!this->emitAdd(*LHSComputationT, E))
      return false;
    break;
  case BO_SubAssign:
    if (!this->emitSub(*LHSComputationT, E))
      return false;
    break;
  case BO_MulAssign:
    if (!this->emitMul(*LHSComputationT, E))
      return false;
    break;
  case BO_DivAssign:
    if (!this->emitDiv(*LHSComputationT, E))
      return false;
    break;
  case BO_RemAssign:
    if (!this->emitRem(*LHSComputationT, E))
      return false;
    break;
  case BO_ShlAssign:
    if (!this->emitShl(*LHSComputationT, *RT, E))
      return false;
    break;
  case BO_ShrAssign:
    if (!this->emitShr(*LHSComputationT, *RT, E))
      return false;
    break;
  case BO_AndAssign:
    if (!this->emitBitAnd(*LHSComputationT, E))
      return false;
    break;
  case BO_XorAssign:
    if (!this->emitBitXor(*LHSComputationT, E))
      return false;
    break;
  case BO_OrAssign:
    if (!this->emitBitOr(*LHSComputationT, E))
      return false;
    break;
  default:
    llvm_unreachable("Unimplemented compound assign operator");
  }

  // And now cast from LHSComputationT to ResultT.
  if (ResultT != LHSComputationT) {
    if (!this->emitCast(*LHSComputationT, *ResultT, E))
      return false;
  }

  // And store the result in LHS.
  if (DiscardResult) {
    if (LHS->refersToBitField())
      return this->emitStoreBitFieldPop(*ResultT, E);
    return this->emitStorePop(*ResultT, E);
  }
  if (LHS->refersToBitField())
    return this->emitStoreBitField(*ResultT, E);
  return this->emitStore(*ResultT, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitExprWithCleanups(const ExprWithCleanups *E) {
  LocalScope<Emitter> ES(this);
  const Expr *SubExpr = E->getSubExpr();

  return this->delegate(SubExpr) && ES.destroyLocals(E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *E) {
  const Expr *SubExpr = E->getSubExpr();

  if (Initializing) {
    // We already have a value, just initialize that.
    return this->delegate(SubExpr);
  }
  // If we don't end up using the materialized temporary anyway, don't
  // bother creating it.
  if (DiscardResult)
    return this->discard(SubExpr);

  // When we're initializing a global variable *or* the storage duration of
  // the temporary is explicitly static, create a global variable.
  std::optional<PrimType> SubExprT = classify(SubExpr);
  bool IsStatic = E->getStorageDuration() == SD_Static;
  if (IsStatic) {
    std::optional<unsigned> GlobalIndex = P.createGlobal(E);
    if (!GlobalIndex)
      return false;

    const LifetimeExtendedTemporaryDecl *TempDecl =
        E->getLifetimeExtendedTemporaryDecl();
    if (IsStatic)
      assert(TempDecl);

    if (SubExprT) {
      if (!this->visit(SubExpr))
        return false;
      if (IsStatic) {
        if (!this->emitInitGlobalTemp(*SubExprT, *GlobalIndex, TempDecl, E))
          return false;
      } else {
        if (!this->emitInitGlobal(*SubExprT, *GlobalIndex, E))
          return false;
      }
      return this->emitGetPtrGlobal(*GlobalIndex, E);
    }

    if (!this->checkLiteralType(SubExpr))
      return false;
    // Non-primitive values.
    if (!this->emitGetPtrGlobal(*GlobalIndex, E))
      return false;
    if (!this->visitInitializer(SubExpr))
      return false;
    if (IsStatic)
      return this->emitInitGlobalTempComp(TempDecl, E);
    return true;
  }

  // For everyhing else, use local variables.
  if (SubExprT) {
    bool IsConst = SubExpr->getType().isConstQualified();
    unsigned LocalIndex =
        allocateLocalPrimitive(E, *SubExprT, IsConst, E->getExtendingDecl());
    if (!this->visit(SubExpr))
      return false;
    if (!this->emitSetLocal(*SubExprT, LocalIndex, E))
      return false;
    return this->emitGetPtrLocal(LocalIndex, E);
  } else {

    if (!this->checkLiteralType(SubExpr))
      return false;

    const Expr *Inner = E->getSubExpr()->skipRValueSubobjectAdjustments();
    if (std::optional<unsigned> LocalIndex =
            allocateLocal(E, Inner->getType(), E->getExtendingDecl())) {
      InitLinkScope<Emitter> ILS(this, InitLink::Temp(*LocalIndex));
      if (!this->emitGetPtrLocal(*LocalIndex, E))
        return false;
      return this->visitInitializer(SubExpr) && this->emitFinishInit(E);
    }
  }
  return false;
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXBindTemporaryExpr(
    const CXXBindTemporaryExpr *E) {
  return this->delegate(E->getSubExpr());
}

template <class Emitter>
bool Compiler<Emitter>::VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
  const Expr *Init = E->getInitializer();
  if (DiscardResult)
    return this->discard(Init);

  if (Initializing) {
    // We already have a value, just initialize that.
    return this->visitInitializer(Init) && this->emitFinishInit(E);
  }

  std::optional<PrimType> T = classify(E->getType());
  if (E->isFileScope()) {
    // Avoid creating a variable if this is a primitive RValue anyway.
    if (T && !E->isLValue())
      return this->delegate(Init);

    if (std::optional<unsigned> GlobalIndex = P.createGlobal(E)) {
      if (!this->emitGetPtrGlobal(*GlobalIndex, E))
        return false;

      if (T) {
        if (!this->visit(Init))
          return false;
        return this->emitInitGlobal(*T, *GlobalIndex, E);
      }

      return this->visitInitializer(Init) && this->emitFinishInit(E);
    }

    return false;
  }

  // Otherwise, use a local variable.
  if (T && !E->isLValue()) {
    // For primitive types, we just visit the initializer.
    return this->delegate(Init);
  }

  unsigned LocalIndex;
  if (T)
    LocalIndex = this->allocateLocalPrimitive(Init, *T, /*IsConst=*/false);
  else if (std::optional<unsigned> MaybeIndex = this->allocateLocal(Init))
    LocalIndex = *MaybeIndex;
  else
    return false;

  if (!this->emitGetPtrLocal(LocalIndex, E))
    return false;

  if (T)
    return this->visit(Init) && this->emitInit(*T, E);
  return this->visitInitializer(Init) && this->emitFinishInit(E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitTypeTraitExpr(const TypeTraitExpr *E) {
  if (DiscardResult)
    return true;
  if (E->isStoredAsBoolean()) {
    if (E->getType()->isBooleanType())
      return this->emitConstBool(E->getBoolValue(), E);
    return this->emitConst(E->getBoolValue(), E);
  }
  PrimType T = classifyPrim(E->getType());
  return this->visitAPValue(E->getAPValue(), T, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *E) {
  if (DiscardResult)
    return true;
  return this->emitConst(E->getValue(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitLambdaExpr(const LambdaExpr *E) {
  if (DiscardResult)
    return true;

  assert(Initializing);
  const Record *R = P.getOrCreateRecord(E->getLambdaClass());
  if (!R)
    return false;

  auto *CaptureInitIt = E->capture_init_begin();
  // Initialize all fields (which represent lambda captures) of the
  // record with their initializers.
  for (const Record::Field &F : R->fields()) {
    const Expr *Init = *CaptureInitIt;
    if (!Init || Init->containsErrors())
      continue;
    ++CaptureInitIt;

    if (std::optional<PrimType> T = classify(Init)) {
      if (!this->visit(Init))
        return false;

      if (!this->emitInitField(*T, F.Offset, E))
        return false;
    } else {
      if (!this->emitGetPtrField(F.Offset, E))
        return false;

      if (!this->visitInitializer(Init))
        return false;

      if (!this->emitPopPtr(E))
        return false;
    }
  }

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitPredefinedExpr(const PredefinedExpr *E) {
  if (DiscardResult)
    return true;

  if (!Initializing) {
    unsigned StringIndex = P.createGlobalString(E->getFunctionName(), E);
    return this->emitGetPtrGlobal(StringIndex, E);
  }

  return this->delegate(E->getFunctionName());
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXThrowExpr(const CXXThrowExpr *E) {
  if (E->getSubExpr() && !this->discard(E->getSubExpr()))
    return false;

  return this->emitInvalid(E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXReinterpretCastExpr(
    const CXXReinterpretCastExpr *E) {
  const Expr *SubExpr = E->getSubExpr();

  std::optional<PrimType> FromT = classify(SubExpr);
  std::optional<PrimType> ToT = classify(E);

  if (!FromT || !ToT)
    return this->emitInvalidCast(CastKind::Reinterpret, /*Fatal=*/true, E);

  if (FromT == PT_Ptr || ToT == PT_Ptr) {
    // Both types could be PT_Ptr because their expressions are glvalues.
    std::optional<PrimType> PointeeFromT;
    if (SubExpr->getType()->isPointerOrReferenceType())
      PointeeFromT = classify(SubExpr->getType()->getPointeeType());
    else
      PointeeFromT = classify(SubExpr->getType());

    std::optional<PrimType> PointeeToT;
    if (E->getType()->isPointerOrReferenceType())
      PointeeToT = classify(E->getType()->getPointeeType());
    else
      PointeeToT = classify(E->getType());

    bool Fatal = true;
    if (PointeeToT && PointeeFromT) {
      if (isIntegralType(*PointeeFromT) && isIntegralType(*PointeeToT))
        Fatal = false;
    } else {
      Fatal = SubExpr->getType().getTypePtr() != E->getType().getTypePtr();
    }

    if (!this->emitInvalidCast(CastKind::Reinterpret, Fatal, E))
      return false;

    if (E->getCastKind() == CK_LValueBitCast)
      return this->delegate(SubExpr);
    return this->VisitCastExpr(E);
  }

  // Try to actually do the cast.
  bool Fatal = (ToT != FromT);
  if (!this->emitInvalidCast(CastKind::Reinterpret, Fatal, E))
    return false;

  return this->VisitCastExpr(E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXDynamicCastExpr(const CXXDynamicCastExpr *E) {

  if (!Ctx.getLangOpts().CPlusPlus20) {
    if (!this->emitInvalidCast(CastKind::Dynamic, /*Fatal=*/false, E))
      return false;
  }

  return this->VisitCastExpr(E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXNoexceptExpr(const CXXNoexceptExpr *E) {
  assert(E->getType()->isBooleanType());

  if (DiscardResult)
    return true;
  return this->emitConstBool(E->getValue(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXConstructExpr(const CXXConstructExpr *E) {
  QualType T = E->getType();
  assert(!classify(T));

  if (T->isRecordType()) {
    const CXXConstructorDecl *Ctor = E->getConstructor();

    // Trivial copy/move constructor. Avoid copy.
    if (Ctor->isDefaulted() && Ctor->isCopyOrMoveConstructor() &&
        Ctor->isTrivial() &&
        E->getArg(0)->isTemporaryObject(Ctx.getASTContext(),
                                        T->getAsCXXRecordDecl()))
      return this->visitInitializer(E->getArg(0));

    // If we're discarding a construct expression, we still need
    // to allocate a variable and call the constructor and destructor.
    if (DiscardResult) {
      if (Ctor->isTrivial())
        return true;
      assert(!Initializing);
      std::optional<unsigned> LocalIndex = allocateLocal(E);

      if (!LocalIndex)
        return false;

      if (!this->emitGetPtrLocal(*LocalIndex, E))
        return false;
    }

    // Zero initialization.
    if (E->requiresZeroInitialization()) {
      const Record *R = getRecord(E->getType());

      if (!this->visitZeroRecordInitializer(R, E))
        return false;

      // If the constructor is trivial anyway, we're done.
      if (Ctor->isTrivial())
        return true;
    }

    const Function *Func = getFunction(Ctor);

    if (!Func)
      return false;

    assert(Func->hasThisPointer());
    assert(!Func->hasRVO());

    //  The This pointer is already on the stack because this is an initializer,
    //  but we need to dup() so the call() below has its own copy.
    if (!this->emitDupPtr(E))
      return false;

    // Constructor arguments.
    for (const auto *Arg : E->arguments()) {
      if (!this->visit(Arg))
        return false;
    }

    if (Func->isVariadic()) {
      uint32_t VarArgSize = 0;
      unsigned NumParams = Func->getNumWrittenParams();
      for (unsigned I = NumParams, N = E->getNumArgs(); I != N; ++I) {
        VarArgSize +=
            align(primSize(classify(E->getArg(I)->getType()).value_or(PT_Ptr)));
      }
      if (!this->emitCallVar(Func, VarArgSize, E))
        return false;
    } else {
      if (!this->emitCall(Func, 0, E)) {
        // When discarding, we don't need the result anyway, so clean up
        // the instance dup we did earlier in case surrounding code wants
        // to keep evaluating.
        if (DiscardResult)
          (void)this->emitPopPtr(E);
        return false;
      }
    }

    if (DiscardResult)
      return this->emitPopPtr(E);
    return this->emitFinishInit(E);
  }

  if (T->isArrayType()) {
    const ConstantArrayType *CAT =
        Ctx.getASTContext().getAsConstantArrayType(E->getType());
    if (!CAT)
      return false;

    size_t NumElems = CAT->getZExtSize();
    const Function *Func = getFunction(E->getConstructor());
    if (!Func)
      return false;

    // FIXME(perf): We're calling the constructor once per array element here,
    //   in the old intepreter we had a special-case for trivial constructors.
    for (size_t I = 0; I != NumElems; ++I) {
      if (!this->emitConstUint64(I, E))
        return false;
      if (!this->emitArrayElemPtrUint64(E))
        return false;

      // Constructor arguments.
      for (const auto *Arg : E->arguments()) {
        if (!this->visit(Arg))
          return false;
      }

      if (!this->emitCall(Func, 0, E))
        return false;
    }
    return true;
  }

  return false;
}

template <class Emitter>
bool Compiler<Emitter>::VisitSourceLocExpr(const SourceLocExpr *E) {
  if (DiscardResult)
    return true;

  const APValue Val =
      E->EvaluateInContext(Ctx.getASTContext(), SourceLocDefaultExpr);

  // Things like __builtin_LINE().
  if (E->getType()->isIntegerType()) {
    assert(Val.isInt());
    const APSInt &I = Val.getInt();
    return this->emitConst(I, E);
  }
  // Otherwise, the APValue is an LValue, with only one element.
  // Theoretically, we don't need the APValue at all of course.
  assert(E->getType()->isPointerType());
  assert(Val.isLValue());
  const APValue::LValueBase &Base = Val.getLValueBase();
  if (const Expr *LValueExpr = Base.dyn_cast<const Expr *>())
    return this->visit(LValueExpr);

  // Otherwise, we have a decl (which is the case for
  // __builtin_source_location).
  assert(Base.is<const ValueDecl *>());
  assert(Val.getLValuePath().size() == 0);
  const auto *BaseDecl = Base.dyn_cast<const ValueDecl *>();
  assert(BaseDecl);

  auto *UGCD = cast<UnnamedGlobalConstantDecl>(BaseDecl);

  std::optional<unsigned> GlobalIndex = P.getOrCreateGlobal(UGCD);
  if (!GlobalIndex)
    return false;

  if (!this->emitGetPtrGlobal(*GlobalIndex, E))
    return false;

  const Record *R = getRecord(E->getType());
  const APValue &V = UGCD->getValue();
  for (unsigned I = 0, N = R->getNumFields(); I != N; ++I) {
    const Record::Field *F = R->getField(I);
    const APValue &FieldValue = V.getStructField(I);

    PrimType FieldT = classifyPrim(F->Decl->getType());

    if (!this->visitAPValue(FieldValue, FieldT, E))
      return false;
    if (!this->emitInitField(FieldT, F->Offset, E))
      return false;
  }

  // Leave the pointer to the global on the stack.
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitOffsetOfExpr(const OffsetOfExpr *E) {
  unsigned N = E->getNumComponents();
  if (N == 0)
    return false;

  for (unsigned I = 0; I != N; ++I) {
    const OffsetOfNode &Node = E->getComponent(I);
    if (Node.getKind() == OffsetOfNode::Array) {
      const Expr *ArrayIndexExpr = E->getIndexExpr(Node.getArrayExprIndex());
      PrimType IndexT = classifyPrim(ArrayIndexExpr->getType());

      if (DiscardResult) {
        if (!this->discard(ArrayIndexExpr))
          return false;
        continue;
      }

      if (!this->visit(ArrayIndexExpr))
        return false;
      // Cast to Sint64.
      if (IndexT != PT_Sint64) {
        if (!this->emitCast(IndexT, PT_Sint64, E))
          return false;
      }
    }
  }

  if (DiscardResult)
    return true;

  PrimType T = classifyPrim(E->getType());
  return this->emitOffsetOf(T, E, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXScalarValueInitExpr(
    const CXXScalarValueInitExpr *E) {
  QualType Ty = E->getType();

  if (DiscardResult || Ty->isVoidType())
    return true;

  if (std::optional<PrimType> T = classify(Ty))
    return this->visitZeroInitializer(*T, Ty, E);

  if (const auto *CT = Ty->getAs<ComplexType>()) {
    if (!Initializing) {
      std::optional<unsigned> LocalIndex = allocateLocal(E);
      if (!LocalIndex)
        return false;
      if (!this->emitGetPtrLocal(*LocalIndex, E))
        return false;
    }

    // Initialize both fields to 0.
    QualType ElemQT = CT->getElementType();
    PrimType ElemT = classifyPrim(ElemQT);

    for (unsigned I = 0; I != 2; ++I) {
      if (!this->visitZeroInitializer(ElemT, ElemQT, E))
        return false;
      if (!this->emitInitElem(ElemT, I, E))
        return false;
    }
    return true;
  }

  if (const auto *VT = Ty->getAs<VectorType>()) {
    // FIXME: Code duplication with the _Complex case above.
    if (!Initializing) {
      std::optional<unsigned> LocalIndex = allocateLocal(E);
      if (!LocalIndex)
        return false;
      if (!this->emitGetPtrLocal(*LocalIndex, E))
        return false;
    }

    // Initialize all fields to 0.
    QualType ElemQT = VT->getElementType();
    PrimType ElemT = classifyPrim(ElemQT);

    for (unsigned I = 0, N = VT->getNumElements(); I != N; ++I) {
      if (!this->visitZeroInitializer(ElemT, ElemQT, E))
        return false;
      if (!this->emitInitElem(ElemT, I, E))
        return false;
    }
    return true;
  }

  return false;
}

template <class Emitter>
bool Compiler<Emitter>::VisitSizeOfPackExpr(const SizeOfPackExpr *E) {
  return this->emitConst(E->getPackLength(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitGenericSelectionExpr(
    const GenericSelectionExpr *E) {
  return this->delegate(E->getResultExpr());
}

template <class Emitter>
bool Compiler<Emitter>::VisitChooseExpr(const ChooseExpr *E) {
  return this->delegate(E->getChosenSubExpr());
}

template <class Emitter>
bool Compiler<Emitter>::VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *E) {
  if (DiscardResult)
    return true;

  return this->emitConst(E->getValue(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXInheritedCtorInitExpr(
    const CXXInheritedCtorInitExpr *E) {
  const CXXConstructorDecl *Ctor = E->getConstructor();
  assert(!Ctor->isTrivial() &&
         "Trivial CXXInheritedCtorInitExpr, implement. (possible?)");
  const Function *F = this->getFunction(Ctor);
  assert(F);
  assert(!F->hasRVO());
  assert(F->hasThisPointer());

  if (!this->emitDupPtr(SourceInfo{}))
    return false;

  // Forward all arguments of the current function (which should be a
  // constructor itself) to the inherited ctor.
  // This is necessary because the calling code has pushed the pointer
  // of the correct base for  us already, but the arguments need
  // to come after.
  unsigned Offset = align(primSize(PT_Ptr)); // instance pointer.
  for (const ParmVarDecl *PD : Ctor->parameters()) {
    PrimType PT = this->classify(PD->getType()).value_or(PT_Ptr);

    if (!this->emitGetParam(PT, Offset, E))
      return false;
    Offset += align(primSize(PT));
  }

  return this->emitCall(F, 0, E);
}

// FIXME: This function has become rather unwieldy, especially
// the part where we initialize an array allocation of dynamic size.
template <class Emitter>
bool Compiler<Emitter>::VisitCXXNewExpr(const CXXNewExpr *E) {
  assert(classifyPrim(E->getType()) == PT_Ptr);
  const Expr *Init = E->getInitializer();
  QualType ElementType = E->getAllocatedType();
  std::optional<PrimType> ElemT = classify(ElementType);
  unsigned PlacementArgs = E->getNumPlacementArgs();
  const FunctionDecl *OperatorNew = E->getOperatorNew();
  const Expr *PlacementDest = nullptr;
  bool IsNoThrow = false;

  if (PlacementArgs != 0) {
    // FIXME: There is no restriction on this, but it's not clear that any
    // other form makes any sense. We get here for cases such as:
    //
    //   new (std::align_val_t{N}) X(int)
    //
    // (which should presumably be valid only if N is a multiple of
    // alignof(int), and in any case can't be deallocated unless N is
    // alignof(X) and X has new-extended alignment).
    if (PlacementArgs == 1) {
      const Expr *Arg1 = E->getPlacementArg(0);
      if (Arg1->getType()->isNothrowT()) {
        if (!this->discard(Arg1))
          return false;
        IsNoThrow = true;
      } else {
        // Invalid unless we have C++26 or are in a std:: function.
        if (!this->emitInvalidNewDeleteExpr(E, E))
          return false;

        // If we have a placement-new destination, we'll later use that instead
        // of allocating.
        if (OperatorNew->isReservedGlobalPlacementOperator())
          PlacementDest = Arg1;
      }
    } else {
      // Always invalid.
      return this->emitInvalid(E);
    }
  } else if (!OperatorNew
                  ->isUsableAsGlobalAllocationFunctionInConstantEvaluation())
    return this->emitInvalidNewDeleteExpr(E, E);

  const Descriptor *Desc;
  if (!PlacementDest) {
    if (ElemT) {
      if (E->isArray())
        Desc = nullptr; // We're not going to use it in this case.
      else
        Desc = P.createDescriptor(E, *ElemT, /*SourceTy=*/nullptr,
                                  Descriptor::InlineDescMD);
    } else {
      Desc = P.createDescriptor(
          E, ElementType.getTypePtr(),
          E->isArray() ? std::nullopt : Descriptor::InlineDescMD,
          /*IsConst=*/false, /*IsTemporary=*/false, /*IsMutable=*/false,
          /*IsVolatile=*/false, Init);
    }
  }

  if (E->isArray()) {
    std::optional<const Expr *> ArraySizeExpr = E->getArraySize();
    if (!ArraySizeExpr)
      return false;

    const Expr *Stripped = *ArraySizeExpr;
    for (; auto *ICE = dyn_cast<ImplicitCastExpr>(Stripped);
         Stripped = ICE->getSubExpr())
      if (ICE->getCastKind() != CK_NoOp &&
          ICE->getCastKind() != CK_IntegralCast)
        break;

    PrimType SizeT = classifyPrim(Stripped->getType());

    // Save evaluated array size to a variable.
    unsigned ArrayLen =
        allocateLocalPrimitive(Stripped, SizeT, /*IsConst=*/false);
    if (!this->visit(Stripped))
      return false;
    if (!this->emitSetLocal(SizeT, ArrayLen, E))
      return false;

    if (PlacementDest) {
      if (!this->visit(PlacementDest))
        return false;
      if (!this->emitStartLifetime(E))
        return false;
      if (!this->emitGetLocal(SizeT, ArrayLen, E))
        return false;
      if (!this->emitCheckNewTypeMismatchArray(SizeT, E, E))
        return false;
    } else {
      if (!this->emitGetLocal(SizeT, ArrayLen, E))
        return false;

      if (ElemT) {
        // N primitive elements.
        if (!this->emitAllocN(SizeT, *ElemT, E, IsNoThrow, E))
          return false;
      } else {
        // N Composite elements.
        if (!this->emitAllocCN(SizeT, Desc, IsNoThrow, E))
          return false;
      }
    }

    if (Init) {
      QualType InitType = Init->getType();
      size_t StaticInitElems = 0;
      const Expr *DynamicInit = nullptr;
      if (const ConstantArrayType *CAT =
              Ctx.getASTContext().getAsConstantArrayType(InitType)) {
        StaticInitElems = CAT->getZExtSize();
        if (!this->visitInitializer(Init))
          return false;

        if (const auto *ILE = dyn_cast<InitListExpr>(Init);
            ILE && ILE->hasArrayFiller())
          DynamicInit = ILE->getArrayFiller();
      }

      // The initializer initializes a certain number of elements, S.
      // However, the complete number of elements, N, might be larger than that.
      // In this case, we need to get an initializer for the remaining elements.
      // There are to cases:
      //   1) For the form 'new Struct[n];', the initializer is a
      //      CXXConstructExpr and its type is an IncompleteArrayType.
      //   2) For the form 'new Struct[n]{1,2,3}', the initializer is an
      //      InitListExpr and the initializer for the remaining elements
      //      is the array filler.

      if (DynamicInit || InitType->isIncompleteArrayType()) {
        const Function *CtorFunc = nullptr;
        if (const auto *CE = dyn_cast<CXXConstructExpr>(Init)) {
          CtorFunc = getFunction(CE->getConstructor());
          if (!CtorFunc)
            return false;
        } else if (!DynamicInit)
          DynamicInit = Init;

        LabelTy EndLabel = this->getLabel();
        LabelTy StartLabel = this->getLabel();

        // In the nothrow case, the alloc above might have returned nullptr.
        // Don't call any constructors that case.
        if (IsNoThrow) {
          if (!this->emitDupPtr(E))
            return false;
          if (!this->emitNullPtr(0, nullptr, E))
            return false;
          if (!this->emitEQPtr(E))
            return false;
          if (!this->jumpTrue(EndLabel))
            return false;
        }

        // Create loop variables.
        unsigned Iter =
            allocateLocalPrimitive(Stripped, SizeT, /*IsConst=*/false);
        if (!this->emitConst(StaticInitElems, SizeT, E))
          return false;
        if (!this->emitSetLocal(SizeT, Iter, E))
          return false;

        this->fallthrough(StartLabel);
        this->emitLabel(StartLabel);
        // Condition. Iter < ArrayLen?
        if (!this->emitGetLocal(SizeT, Iter, E))
          return false;
        if (!this->emitGetLocal(SizeT, ArrayLen, E))
          return false;
        if (!this->emitLT(SizeT, E))
          return false;
        if (!this->jumpFalse(EndLabel))
          return false;

        // Pointer to the allocated array is already on the stack.
        if (!this->emitGetLocal(SizeT, Iter, E))
          return false;
        if (!this->emitArrayElemPtr(SizeT, E))
          return false;

        if (isa_and_nonnull<ImplicitValueInitExpr>(DynamicInit) &&
            DynamicInit->getType()->isArrayType()) {
          QualType ElemType =
              DynamicInit->getType()->getAsArrayTypeUnsafe()->getElementType();
          PrimType InitT = classifyPrim(ElemType);
          if (!this->visitZeroInitializer(InitT, ElemType, E))
            return false;
          if (!this->emitStorePop(InitT, E))
            return false;
        } else if (DynamicInit) {
          if (std::optional<PrimType> InitT = classify(DynamicInit)) {
            if (!this->visit(DynamicInit))
              return false;
            if (!this->emitStorePop(*InitT, E))
              return false;
          } else {
            if (!this->visitInitializer(DynamicInit))
              return false;
            if (!this->emitPopPtr(E))
              return false;
          }
        } else {
          assert(CtorFunc);
          if (!this->emitCall(CtorFunc, 0, E))
            return false;
        }

        // ++Iter;
        if (!this->emitGetPtrLocal(Iter, E))
          return false;
        if (!this->emitIncPop(SizeT, false, E))
          return false;

        if (!this->jump(StartLabel))
          return false;

        this->fallthrough(EndLabel);
        this->emitLabel(EndLabel);
      }
    }
  } else { // Non-array.
    if (PlacementDest) {
      if (!this->visit(PlacementDest))
        return false;
      if (!this->emitStartLifetime(E))
        return false;
      if (!this->emitCheckNewTypeMismatch(E, E))
        return false;
    } else {
      // Allocate just one element.
      if (!this->emitAlloc(Desc, E))
        return false;
    }

    if (Init) {
      if (ElemT) {
        if (!this->visit(Init))
          return false;

        if (!this->emitInit(*ElemT, E))
          return false;
      } else {
        // Composite.
        if (!this->visitInitializer(Init))
          return false;
      }
    }
  }

  if (DiscardResult)
    return this->emitPopPtr(E);

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXDeleteExpr(const CXXDeleteExpr *E) {
  const Expr *Arg = E->getArgument();

  const FunctionDecl *OperatorDelete = E->getOperatorDelete();

  if (!OperatorDelete->isUsableAsGlobalAllocationFunctionInConstantEvaluation())
    return this->emitInvalidNewDeleteExpr(E, E);

  // Arg must be an lvalue.
  if (!this->visit(Arg))
    return false;

  return this->emitFree(E->isArrayForm(), E->isGlobalDelete(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitBlockExpr(const BlockExpr *E) {
  if (DiscardResult)
    return true;

  const Function *Func = nullptr;
  if (auto F = Ctx.getOrCreateObjCBlock(E))
    Func = F;

  if (!Func)
    return false;
  return this->emitGetFnPtr(Func, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXTypeidExpr(const CXXTypeidExpr *E) {
  const Type *TypeInfoType = E->getType().getTypePtr();

  auto canonType = [](const Type *T) {
    return T->getCanonicalTypeUnqualified().getTypePtr();
  };

  if (!E->isPotentiallyEvaluated()) {
    if (DiscardResult)
      return true;

    if (E->isTypeOperand())
      return this->emitGetTypeid(
          canonType(E->getTypeOperand(Ctx.getASTContext()).getTypePtr()),
          TypeInfoType, E);

    return this->emitGetTypeid(
        canonType(E->getExprOperand()->getType().getTypePtr()), TypeInfoType,
        E);
  }

  // Otherwise, we need to evaluate the expression operand.
  assert(E->getExprOperand());
  assert(E->getExprOperand()->isLValue());

  if (!Ctx.getLangOpts().CPlusPlus20 && !this->emitDiagTypeid(E))
    return false;

  if (!this->visit(E->getExprOperand()))
    return false;

  if (!this->emitGetTypeidPtr(TypeInfoType, E))
    return false;
  if (DiscardResult)
    return this->emitPopPtr(E);
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitExpressionTraitExpr(const ExpressionTraitExpr *E) {
  assert(Ctx.getLangOpts().CPlusPlus);
  return this->emitConstBool(E->getValue(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXUuidofExpr(const CXXUuidofExpr *E) {
  if (DiscardResult)
    return true;
  assert(!Initializing);

  const MSGuidDecl *GuidDecl = E->getGuidDecl();
  const RecordDecl *RD = GuidDecl->getType()->getAsRecordDecl();
  assert(RD);
  // If the definiton of the result type is incomplete, just return a dummy.
  // If (and when) that is read from, we will fail, but not now.
  if (!RD->isCompleteDefinition())
    return this->emitDummyPtr(GuidDecl, E);

  std::optional<unsigned> GlobalIndex = P.getOrCreateGlobal(GuidDecl);
  if (!GlobalIndex)
    return false;
  if (!this->emitGetPtrGlobal(*GlobalIndex, E))
    return false;

  assert(this->getRecord(E->getType()));

  const APValue &V = GuidDecl->getAsAPValue();
  if (V.getKind() == APValue::None)
    return true;

  assert(V.isStruct());
  assert(V.getStructNumBases() == 0);
  if (!this->visitAPValueInitializer(V, E, E->getType()))
    return false;

  return this->emitFinishInit(E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitRequiresExpr(const RequiresExpr *E) {
  assert(classifyPrim(E->getType()) == PT_Bool);
  if (DiscardResult)
    return true;
  return this->emitConstBool(E->isSatisfied(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitConceptSpecializationExpr(
    const ConceptSpecializationExpr *E) {
  assert(classifyPrim(E->getType()) == PT_Bool);
  if (DiscardResult)
    return true;
  return this->emitConstBool(E->isSatisfied(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXRewrittenBinaryOperator(
    const CXXRewrittenBinaryOperator *E) {
  return this->delegate(E->getSemanticForm());
}

template <class Emitter>
bool Compiler<Emitter>::VisitPseudoObjectExpr(const PseudoObjectExpr *E) {

  for (const Expr *SemE : E->semantics()) {
    if (auto *OVE = dyn_cast<OpaqueValueExpr>(SemE)) {
      if (SemE == E->getResultExpr())
        return false;

      if (OVE->isUnique())
        continue;

      if (!this->discard(OVE))
        return false;
    } else if (SemE == E->getResultExpr()) {
      if (!this->delegate(SemE))
        return false;
    } else {
      if (!this->discard(SemE))
        return false;
    }
  }
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitPackIndexingExpr(const PackIndexingExpr *E) {
  return this->delegate(E->getSelectedExpr());
}

template <class Emitter>
bool Compiler<Emitter>::VisitRecoveryExpr(const RecoveryExpr *E) {
  return this->emitError(E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitAddrLabelExpr(const AddrLabelExpr *E) {
  assert(E->getType()->isVoidPointerType());

  unsigned Offset =
      allocateLocalPrimitive(E->getLabel(), PT_Ptr, /*IsConst=*/true);

  return this->emitGetLocal(PT_Ptr, Offset, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitConvertVectorExpr(const ConvertVectorExpr *E) {
  assert(Initializing);
  const auto *VT = E->getType()->castAs<VectorType>();
  QualType ElemType = VT->getElementType();
  PrimType ElemT = classifyPrim(ElemType);
  const Expr *Src = E->getSrcExpr();
  QualType SrcType = Src->getType();
  PrimType SrcElemT = classifyVectorElementType(SrcType);

  unsigned SrcOffset =
      this->allocateLocalPrimitive(Src, PT_Ptr, /*IsConst=*/true);
  if (!this->visit(Src))
    return false;
  if (!this->emitSetLocal(PT_Ptr, SrcOffset, E))
    return false;

  for (unsigned I = 0; I != VT->getNumElements(); ++I) {
    if (!this->emitGetLocal(PT_Ptr, SrcOffset, E))
      return false;
    if (!this->emitArrayElemPop(SrcElemT, I, E))
      return false;

    // Cast to the desired result element type.
    if (SrcElemT != ElemT) {
      if (!this->emitPrimCast(SrcElemT, ElemT, ElemType, E))
        return false;
    } else if (ElemType->isFloatingType() && SrcType != ElemType) {
      const auto *TargetSemantics = &Ctx.getFloatSemantics(ElemType);
      if (!this->emitCastFP(TargetSemantics, getRoundingMode(E), E))
        return false;
    }
    if (!this->emitInitElem(ElemT, I, E))
      return false;
  }

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitShuffleVectorExpr(const ShuffleVectorExpr *E) {
  assert(Initializing);
  assert(E->getNumSubExprs() > 2);

  const Expr *Vecs[] = {E->getExpr(0), E->getExpr(1)};
  const VectorType *VT = Vecs[0]->getType()->castAs<VectorType>();
  PrimType ElemT = classifyPrim(VT->getElementType());
  unsigned NumInputElems = VT->getNumElements();
  unsigned NumOutputElems = E->getNumSubExprs() - 2;
  assert(NumOutputElems > 0);

  // Save both input vectors to a local variable.
  unsigned VectorOffsets[2];
  for (unsigned I = 0; I != 2; ++I) {
    VectorOffsets[I] =
        this->allocateLocalPrimitive(Vecs[I], PT_Ptr, /*IsConst=*/true);
    if (!this->visit(Vecs[I]))
      return false;
    if (!this->emitSetLocal(PT_Ptr, VectorOffsets[I], E))
      return false;
  }
  for (unsigned I = 0; I != NumOutputElems; ++I) {
    APSInt ShuffleIndex = E->getShuffleMaskIdx(I);
    assert(ShuffleIndex >= -1);
    if (ShuffleIndex == -1)
      return this->emitInvalidShuffleVectorIndex(I, E);

    assert(ShuffleIndex < (NumInputElems * 2));
    if (!this->emitGetLocal(PT_Ptr,
                            VectorOffsets[ShuffleIndex >= NumInputElems], E))
      return false;
    unsigned InputVectorIndex = ShuffleIndex.getZExtValue() % NumInputElems;
    if (!this->emitArrayElemPop(ElemT, InputVectorIndex, E))
      return false;

    if (!this->emitInitElem(ElemT, I, E))
      return false;
  }

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitExtVectorElementExpr(
    const ExtVectorElementExpr *E) {
  const Expr *Base = E->getBase();
  assert(
      Base->getType()->isVectorType() ||
      Base->getType()->getAs<PointerType>()->getPointeeType()->isVectorType());

  SmallVector<uint32_t, 4> Indices;
  E->getEncodedElementAccess(Indices);

  if (Indices.size() == 1) {
    if (!this->visit(Base))
      return false;

    if (E->isGLValue()) {
      if (!this->emitConstUint32(Indices[0], E))
        return false;
      return this->emitArrayElemPtrPop(PT_Uint32, E);
    }
    // Else, also load the value.
    return this->emitArrayElemPop(classifyPrim(E->getType()), Indices[0], E);
  }

  // Create a local variable for the base.
  unsigned BaseOffset = allocateLocalPrimitive(Base, PT_Ptr, /*IsConst=*/true);
  if (!this->visit(Base))
    return false;
  if (!this->emitSetLocal(PT_Ptr, BaseOffset, E))
    return false;

  // Now the vector variable for the return value.
  if (!Initializing) {
    std::optional<unsigned> ResultIndex;
    ResultIndex = allocateLocal(E);
    if (!ResultIndex)
      return false;
    if (!this->emitGetPtrLocal(*ResultIndex, E))
      return false;
  }

  assert(Indices.size() == E->getType()->getAs<VectorType>()->getNumElements());

  PrimType ElemT =
      classifyPrim(E->getType()->getAs<VectorType>()->getElementType());
  uint32_t DstIndex = 0;
  for (uint32_t I : Indices) {
    if (!this->emitGetLocal(PT_Ptr, BaseOffset, E))
      return false;
    if (!this->emitArrayElemPop(ElemT, I, E))
      return false;
    if (!this->emitInitElem(ElemT, DstIndex, E))
      return false;
    ++DstIndex;
  }

  // Leave the result pointer on the stack.
  assert(!DiscardResult);
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitObjCBoxedExpr(const ObjCBoxedExpr *E) {
  const Expr *SubExpr = E->getSubExpr();
  if (!E->isExpressibleAsConstantInitializer())
    return this->discard(SubExpr) && this->emitInvalid(E);

  if (DiscardResult)
    return true;

  assert(classifyPrim(E) == PT_Ptr);
  return this->emitDummyPtr(E, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXStdInitializerListExpr(
    const CXXStdInitializerListExpr *E) {
  const Expr *SubExpr = E->getSubExpr();
  const ConstantArrayType *ArrayType =
      Ctx.getASTContext().getAsConstantArrayType(SubExpr->getType());
  const Record *R = getRecord(E->getType());
  assert(Initializing);
  assert(SubExpr->isGLValue());

  if (!this->visit(SubExpr))
    return false;
  if (!this->emitConstUint8(0, E))
    return false;
  if (!this->emitArrayElemPtrPopUint8(E))
    return false;
  if (!this->emitInitFieldPtr(R->getField(0u)->Offset, E))
    return false;

  PrimType SecondFieldT = classifyPrim(R->getField(1u)->Decl->getType());
  if (isIntegralType(SecondFieldT)) {
    if (!this->emitConst(static_cast<APSInt>(ArrayType->getSize()),
                         SecondFieldT, E))
      return false;
    return this->emitInitField(SecondFieldT, R->getField(1u)->Offset, E);
  }
  assert(SecondFieldT == PT_Ptr);

  if (!this->emitGetFieldPtr(R->getField(0u)->Offset, E))
    return false;
  if (!this->emitExpandPtr(E))
    return false;
  if (!this->emitConst(static_cast<APSInt>(ArrayType->getSize()), PT_Uint64, E))
    return false;
  if (!this->emitArrayElemPtrPop(PT_Uint64, E))
    return false;
  return this->emitInitFieldPtr(R->getField(1u)->Offset, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitStmtExpr(const StmtExpr *E) {
  BlockScope<Emitter> BS(this);
  StmtExprScope<Emitter> SS(this);

  const CompoundStmt *CS = E->getSubStmt();
  const Stmt *Result = CS->getStmtExprResult();
  for (const Stmt *S : CS->body()) {
    if (S != Result) {
      if (!this->visitStmt(S))
        return false;
      continue;
    }

    assert(S == Result);
    if (const Expr *ResultExpr = dyn_cast<Expr>(S))
      return this->delegate(ResultExpr);
    return this->emitUnsupported(E);
  }

  return BS.destroyLocals();
}

template <class Emitter> bool Compiler<Emitter>::discard(const Expr *E) {
  OptionScope<Emitter> Scope(this, /*NewDiscardResult=*/true,
                             /*NewInitializing=*/false);
  return this->Visit(E);
}

template <class Emitter> bool Compiler<Emitter>::delegate(const Expr *E) {
  // We're basically doing:
  // OptionScope<Emitter> Scope(this, DicardResult, Initializing);
  // but that's unnecessary of course.
  return this->Visit(E);
}

template <class Emitter> bool Compiler<Emitter>::visit(const Expr *E) {
  if (E->getType().isNull())
    return false;

  if (E->getType()->isVoidType())
    return this->discard(E);

  // Create local variable to hold the return value.
  if (!E->isGLValue() && !E->getType()->isAnyComplexType() &&
      !classify(E->getType())) {
    std::optional<unsigned> LocalIndex = allocateLocal(E);
    if (!LocalIndex)
      return false;

    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
    InitLinkScope<Emitter> ILS(this, InitLink::Temp(*LocalIndex));
    return this->visitInitializer(E);
  }

  //  Otherwise,we have a primitive return value, produce the value directly
  //  and push it on the stack.
  OptionScope<Emitter> Scope(this, /*NewDiscardResult=*/false,
                             /*NewInitializing=*/false);
  return this->Visit(E);
}

template <class Emitter>
bool Compiler<Emitter>::visitInitializer(const Expr *E) {
  assert(!classify(E->getType()));

  OptionScope<Emitter> Scope(this, /*NewDiscardResult=*/false,
                             /*NewInitializing=*/true);
  return this->Visit(E);
}

template <class Emitter> bool Compiler<Emitter>::visitBool(const Expr *E) {
  std::optional<PrimType> T = classify(E->getType());
  if (!T) {
    // Convert complex values to bool.
    if (E->getType()->isAnyComplexType()) {
      if (!this->visit(E))
        return false;
      return this->emitComplexBoolCast(E);
    }
    return false;
  }

  if (!this->visit(E))
    return false;

  if (T == PT_Bool)
    return true;

  // Convert pointers to bool.
  if (T == PT_Ptr)
    return this->emitIsNonNullPtr(E);

  // Or Floats.
  if (T == PT_Float)
    return this->emitCastFloatingIntegralBool(getFPOptions(E), E);

  // Or anything else we can.
  return this->emitCast(*T, PT_Bool, E);
}

template <class Emitter>
bool Compiler<Emitter>::visitZeroInitializer(PrimType T, QualType QT,
                                             const Expr *E) {
  if (const auto *AT = QT->getAs<AtomicType>())
    QT = AT->getValueType();

  switch (T) {
  case PT_Bool:
    return this->emitZeroBool(E);
  case PT_Sint8:
    return this->emitZeroSint8(E);
  case PT_Uint8:
    return this->emitZeroUint8(E);
  case PT_Sint16:
    return this->emitZeroSint16(E);
  case PT_Uint16:
    return this->emitZeroUint16(E);
  case PT_Sint32:
    return this->emitZeroSint32(E);
  case PT_Uint32:
    return this->emitZeroUint32(E);
  case PT_Sint64:
    return this->emitZeroSint64(E);
  case PT_Uint64:
    return this->emitZeroUint64(E);
  case PT_IntAP:
    return this->emitZeroIntAP(Ctx.getBitWidth(QT), E);
  case PT_IntAPS:
    return this->emitZeroIntAPS(Ctx.getBitWidth(QT), E);
  case PT_Ptr:
    return this->emitNullPtr(Ctx.getASTContext().getTargetNullPointerValue(QT),
                             nullptr, E);
  case PT_MemberPtr:
    return this->emitNullMemberPtr(0, nullptr, E);
  case PT_Float: {
    APFloat F = APFloat::getZero(Ctx.getFloatSemantics(QT));
    return this->emitFloat(F, E);
  }
  case PT_FixedPoint: {
    auto Sem = Ctx.getASTContext().getFixedPointSemantics(E->getType());
    return this->emitConstFixedPoint(FixedPoint::zero(Sem), E);
  }
  }
  llvm_unreachable("unknown primitive type");
}

template <class Emitter>
bool Compiler<Emitter>::visitZeroRecordInitializer(const Record *R,
                                                   const Expr *E) {
  assert(E);
  assert(R);
  // Fields
  for (const Record::Field &Field : R->fields()) {
    if (Field.isUnnamedBitField())
      continue;

    const Descriptor *D = Field.Desc;
    if (D->isPrimitive()) {
      QualType QT = D->getType();
      PrimType T = classifyPrim(D->getType());
      if (!this->visitZeroInitializer(T, QT, E))
        return false;
      if (!this->emitInitField(T, Field.Offset, E))
        return false;
      if (R->isUnion())
        break;
      continue;
    }

    if (!this->emitGetPtrField(Field.Offset, E))
      return false;

    if (D->isPrimitiveArray()) {
      QualType ET = D->getElemQualType();
      PrimType T = classifyPrim(ET);
      for (uint32_t I = 0, N = D->getNumElems(); I != N; ++I) {
        if (!this->visitZeroInitializer(T, ET, E))
          return false;
        if (!this->emitInitElem(T, I, E))
          return false;
      }
    } else if (D->isCompositeArray()) {
      // Can't be a vector or complex field.
      if (!this->visitZeroArrayInitializer(D->getType(), E))
        return false;
    } else if (D->isRecord()) {
      if (!this->visitZeroRecordInitializer(D->ElemRecord, E))
        return false;
    } else
      return false;

    if (!this->emitFinishInitPop(E))
      return false;

    // C++11 [dcl.init]p5: If T is a (possibly cv-qualified) union type, the
    // object's first non-static named data member is zero-initialized
    if (R->isUnion())
      break;
  }

  for (const Record::Base &B : R->bases()) {
    if (!this->emitGetPtrBase(B.Offset, E))
      return false;
    if (!this->visitZeroRecordInitializer(B.R, E))
      return false;
    if (!this->emitFinishInitPop(E))
      return false;
  }

  // FIXME: Virtual bases.

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::visitZeroArrayInitializer(QualType T, const Expr *E) {
  assert(T->isArrayType() || T->isAnyComplexType() || T->isVectorType());
  const ArrayType *AT = T->getAsArrayTypeUnsafe();
  QualType ElemType = AT->getElementType();
  size_t NumElems = cast<ConstantArrayType>(AT)->getZExtSize();

  if (std::optional<PrimType> ElemT = classify(ElemType)) {
    for (size_t I = 0; I != NumElems; ++I) {
      if (!this->visitZeroInitializer(*ElemT, ElemType, E))
        return false;
      if (!this->emitInitElem(*ElemT, I, E))
        return false;
    }
    return true;
  } else if (ElemType->isRecordType()) {
    const Record *R = getRecord(ElemType);

    for (size_t I = 0; I != NumElems; ++I) {
      if (!this->emitConstUint32(I, E))
        return false;
      if (!this->emitArrayElemPtr(PT_Uint32, E))
        return false;
      if (!this->visitZeroRecordInitializer(R, E))
        return false;
      if (!this->emitPopPtr(E))
        return false;
    }
    return true;
  } else if (ElemType->isArrayType()) {
    for (size_t I = 0; I != NumElems; ++I) {
      if (!this->emitConstUint32(I, E))
        return false;
      if (!this->emitArrayElemPtr(PT_Uint32, E))
        return false;
      if (!this->visitZeroArrayInitializer(ElemType, E))
        return false;
      if (!this->emitPopPtr(E))
        return false;
    }
    return true;
  }

  return false;
}

template <class Emitter>
template <typename T>
bool Compiler<Emitter>::emitConst(T Value, PrimType Ty, const Expr *E) {
  switch (Ty) {
  case PT_Sint8:
    return this->emitConstSint8(Value, E);
  case PT_Uint8:
    return this->emitConstUint8(Value, E);
  case PT_Sint16:
    return this->emitConstSint16(Value, E);
  case PT_Uint16:
    return this->emitConstUint16(Value, E);
  case PT_Sint32:
    return this->emitConstSint32(Value, E);
  case PT_Uint32:
    return this->emitConstUint32(Value, E);
  case PT_Sint64:
    return this->emitConstSint64(Value, E);
  case PT_Uint64:
    return this->emitConstUint64(Value, E);
  case PT_Bool:
    return this->emitConstBool(Value, E);
  case PT_Ptr:
  case PT_MemberPtr:
  case PT_Float:
  case PT_IntAP:
  case PT_IntAPS:
  case PT_FixedPoint:
    llvm_unreachable("Invalid integral type");
    break;
  }
  llvm_unreachable("unknown primitive type");
}

template <class Emitter>
template <typename T>
bool Compiler<Emitter>::emitConst(T Value, const Expr *E) {
  return this->emitConst(Value, classifyPrim(E->getType()), E);
}

template <class Emitter>
bool Compiler<Emitter>::emitConst(const APSInt &Value, PrimType Ty,
                                  const Expr *E) {
  if (Ty == PT_IntAPS)
    return this->emitConstIntAPS(Value, E);
  if (Ty == PT_IntAP)
    return this->emitConstIntAP(Value, E);

  if (Value.isSigned())
    return this->emitConst(Value.getSExtValue(), Ty, E);
  return this->emitConst(Value.getZExtValue(), Ty, E);
}

template <class Emitter>
bool Compiler<Emitter>::emitConst(const APSInt &Value, const Expr *E) {
  return this->emitConst(Value, classifyPrim(E->getType()), E);
}

template <class Emitter>
unsigned Compiler<Emitter>::allocateLocalPrimitive(
    DeclTy &&Src, PrimType Ty, bool IsConst, const ValueDecl *ExtendingDecl,
    ScopeKind SC, bool IsConstexprUnknown) {
  // Make sure we don't accidentally register the same decl twice.
  if (const auto *VD =
          dyn_cast_if_present<ValueDecl>(Src.dyn_cast<const Decl *>())) {
    assert(!P.getGlobal(VD));
    assert(!Locals.contains(VD));
    (void)VD;
  }

  // FIXME: There are cases where Src.is<Expr*>() is wrong, e.g.
  //   (int){12} in C. Consider using Expr::isTemporaryObject() instead
  //   or isa<MaterializeTemporaryExpr>().
  Descriptor *D = P.createDescriptor(Src, Ty, nullptr, Descriptor::InlineDescMD,
                                     IsConst, isa<const Expr *>(Src));
  D->IsConstexprUnknown = IsConstexprUnknown;
  Scope::Local Local = this->createLocal(D);
  if (auto *VD = dyn_cast_if_present<ValueDecl>(Src.dyn_cast<const Decl *>()))
    Locals.insert({VD, Local});
  if (ExtendingDecl)
    VarScope->addExtended(Local, ExtendingDecl);
  else
    VarScope->addForScopeKind(Local, SC);
  return Local.Offset;
}

template <class Emitter>
std::optional<unsigned>
Compiler<Emitter>::allocateLocal(DeclTy &&Src, QualType Ty,
                                 const ValueDecl *ExtendingDecl, ScopeKind SC,
                                 bool IsConstexprUnknown) {
  // Make sure we don't accidentally register the same decl twice.
  if ([[maybe_unused]] const auto *VD =
          dyn_cast_if_present<ValueDecl>(Src.dyn_cast<const Decl *>())) {
    assert(!P.getGlobal(VD));
    assert(!Locals.contains(VD));
  }

  const ValueDecl *Key = nullptr;
  const Expr *Init = nullptr;
  bool IsTemporary = false;
  if (auto *VD = dyn_cast_if_present<ValueDecl>(Src.dyn_cast<const Decl *>())) {
    Key = VD;
    Ty = VD->getType();

    if (const auto *VarD = dyn_cast<VarDecl>(VD))
      Init = VarD->getInit();
  }
  if (auto *E = Src.dyn_cast<const Expr *>()) {
    IsTemporary = true;
    if (Ty.isNull())
      Ty = E->getType();
  }

  Descriptor *D = P.createDescriptor(
      Src, Ty.getTypePtr(), Descriptor::InlineDescMD, Ty.isConstQualified(),
      IsTemporary, /*IsMutable=*/false, /*IsVolatile=*/false, Init);
  if (!D)
    return std::nullopt;
  D->IsConstexprUnknown = IsConstexprUnknown;

  Scope::Local Local = this->createLocal(D);
  if (Key)
    Locals.insert({Key, Local});
  if (ExtendingDecl)
    VarScope->addExtended(Local, ExtendingDecl);
  else
    VarScope->addForScopeKind(Local, SC);
  return Local.Offset;
}

template <class Emitter>
std::optional<unsigned> Compiler<Emitter>::allocateTemporary(const Expr *E) {
  QualType Ty = E->getType();
  assert(!Ty->isRecordType());

  Descriptor *D = P.createDescriptor(
      E, Ty.getTypePtr(), Descriptor::InlineDescMD, Ty.isConstQualified(),
      /*IsTemporary=*/true);

  if (!D)
    return std::nullopt;

  Scope::Local Local = this->createLocal(D);
  VariableScope<Emitter> *S = VarScope;
  assert(S);
  // Attach to topmost scope.
  while (S->getParent())
    S = S->getParent();
  assert(S && !S->getParent());
  S->addLocal(Local);
  return Local.Offset;
}

template <class Emitter>
const RecordType *Compiler<Emitter>::getRecordTy(QualType Ty) {
  if (const PointerType *PT = dyn_cast<PointerType>(Ty))
    return PT->getPointeeType()->getAs<RecordType>();
  return Ty->getAs<RecordType>();
}

template <class Emitter> Record *Compiler<Emitter>::getRecord(QualType Ty) {
  if (const auto *RecordTy = getRecordTy(Ty))
    return getRecord(RecordTy->getDecl());
  return nullptr;
}

template <class Emitter>
Record *Compiler<Emitter>::getRecord(const RecordDecl *RD) {
  return P.getOrCreateRecord(RD);
}

template <class Emitter>
const Function *Compiler<Emitter>::getFunction(const FunctionDecl *FD) {
  return Ctx.getOrCreateFunction(FD);
}

template <class Emitter>
bool Compiler<Emitter>::visitExpr(const Expr *E, bool DestroyToplevelScope) {
  LocalScope<Emitter> RootScope(this);

  // If we won't destroy the toplevel scope, check for memory leaks first.
  if (!DestroyToplevelScope) {
    if (!this->emitCheckAllocations(E))
      return false;
  }

  auto maybeDestroyLocals = [&]() -> bool {
    if (DestroyToplevelScope)
      return RootScope.destroyLocals() && this->emitCheckAllocations(E);
    return this->emitCheckAllocations(E);
  };

  // Void expressions.
  if (E->getType()->isVoidType()) {
    if (!visit(E))
      return false;
    return this->emitRetVoid(E) && maybeDestroyLocals();
  }

  // Expressions with a primitive return type.
  if (std::optional<PrimType> T = classify(E)) {
    if (!visit(E))
      return false;

    return this->emitRet(*T, E) && maybeDestroyLocals();
  }

  // Expressions with a composite return type.
  // For us, that means everything we don't
  // have a PrimType for.
  if (std::optional<unsigned> LocalOffset = this->allocateLocal(E)) {
    InitLinkScope<Emitter> ILS(this, InitLink::Temp(*LocalOffset));
    if (!this->emitGetPtrLocal(*LocalOffset, E))
      return false;

    if (!visitInitializer(E))
      return false;

    if (!this->emitFinishInit(E))
      return false;
    // We are destroying the locals AFTER the Ret op.
    // The Ret op needs to copy the (alive) values, but the
    // destructors may still turn the entire expression invalid.
    return this->emitRetValue(E) && maybeDestroyLocals();
  }

  return maybeDestroyLocals() && this->emitCheckAllocations(E) && false;
}

template <class Emitter>
VarCreationState Compiler<Emitter>::visitDecl(const VarDecl *VD,
                                              bool IsConstexprUnknown) {

  auto R = this->visitVarDecl(VD, /*Toplevel=*/true, IsConstexprUnknown);

  if (R.notCreated())
    return R;

  if (R)
    return true;

  if (!R && Context::shouldBeGloballyIndexed(VD)) {
    if (auto GlobalIndex = P.getGlobal(VD)) {
      Block *GlobalBlock = P.getGlobal(*GlobalIndex);
      GlobalInlineDescriptor &GD =
          *reinterpret_cast<GlobalInlineDescriptor *>(GlobalBlock->rawData());

      GD.InitState = GlobalInitState::InitializerFailed;
      GlobalBlock->invokeDtor();
    }
  }

  return R;
}

/// Toplevel visitDeclAndReturn().
/// We get here from evaluateAsInitializer().
/// We need to evaluate the initializer and return its value.
template <class Emitter>
bool Compiler<Emitter>::visitDeclAndReturn(const VarDecl *VD,
                                           bool ConstantContext) {

  // We only create variables if we're evaluating in a constant context.
  // Otherwise, just evaluate the initializer and return it.
  if (!ConstantContext) {
    DeclScope<Emitter> LS(this, VD);
    const Expr *Init = VD->getInit();
    if (!this->visit(Init))
      return false;
    return this->emitRet(classify(Init).value_or(PT_Ptr), VD) &&
           LS.destroyLocals() && this->emitCheckAllocations(VD);
  }

  LocalScope<Emitter> VDScope(this, VD);
  if (!this->visitVarDecl(VD, /*Toplevel=*/true))
    return false;

  std::optional<PrimType> VarT = classify(VD->getType());
  if (Context::shouldBeGloballyIndexed(VD)) {
    auto GlobalIndex = P.getGlobal(VD);
    assert(GlobalIndex); // visitVarDecl() didn't return false.
    if (VarT) {
      if (!this->emitGetGlobalUnchecked(*VarT, *GlobalIndex, VD))
        return false;
    } else {
      if (!this->emitGetPtrGlobal(*GlobalIndex, VD))
        return false;
    }
  } else {
    auto Local = Locals.find(VD);
    assert(Local != Locals.end()); // Same here.
    if (VarT) {
      if (!this->emitGetLocal(*VarT, Local->second.Offset, VD))
        return false;
    } else {
      if (!this->emitGetPtrLocal(Local->second.Offset, VD))
        return false;
    }
  }

  // Return the value.
  if (!this->emitRet(VarT.value_or(PT_Ptr), VD)) {
    // If the Ret above failed and this is a global variable, mark it as
    // uninitialized, even everything else succeeded.
    if (Context::shouldBeGloballyIndexed(VD)) {
      auto GlobalIndex = P.getGlobal(VD);
      assert(GlobalIndex);
      Block *GlobalBlock = P.getGlobal(*GlobalIndex);
      GlobalInlineDescriptor &GD =
          *reinterpret_cast<GlobalInlineDescriptor *>(GlobalBlock->rawData());

      GD.InitState = GlobalInitState::InitializerFailed;
      GlobalBlock->invokeDtor();
    }
    return false;
  }

  return VDScope.destroyLocals() && this->emitCheckAllocations(VD);
}

template <class Emitter>
VarCreationState Compiler<Emitter>::visitVarDecl(const VarDecl *VD,
                                                 bool Toplevel,
                                                 bool IsConstexprUnknown) {
  // We don't know what to do with these, so just return false.
  if (VD->getType().isNull())
    return false;

  // This case is EvalEmitter-only. If we won't create any instructions for the
  // initializer anyway, don't bother creating the variable in the first place.
  if (!this->isActive())
    return VarCreationState::NotCreated();

  const Expr *Init = VD->getInit();
  std::optional<PrimType> VarT = classify(VD->getType());

  if (Init && Init->isValueDependent())
    return false;

  if (Context::shouldBeGloballyIndexed(VD)) {
    auto checkDecl = [&]() -> bool {
      bool NeedsOp = !Toplevel && VD->isLocalVarDecl() && VD->isStaticLocal();
      return !NeedsOp || this->emitCheckDecl(VD, VD);
    };

    auto initGlobal = [&](unsigned GlobalIndex) -> bool {
      assert(Init);

      if (VarT) {
        if (!this->visit(Init))
          return checkDecl() && false;

        return checkDecl() && this->emitInitGlobal(*VarT, GlobalIndex, VD);
      }

      if (!checkDecl())
        return false;

      if (!this->emitGetPtrGlobal(GlobalIndex, Init))
        return false;

      if (!visitInitializer(Init))
        return false;

      return this->emitFinishInitGlobal(Init);
    };

    DeclScope<Emitter> LocalScope(this, VD);

    // We've already seen and initialized this global.
    if (std::optional<unsigned> GlobalIndex = P.getGlobal(VD)) {
      if (P.getPtrGlobal(*GlobalIndex).isInitialized())
        return checkDecl();

      // The previous attempt at initialization might've been unsuccessful,
      // so let's try this one.
      return Init && checkDecl() && initGlobal(*GlobalIndex);
    }

    std::optional<unsigned> GlobalIndex = P.createGlobal(VD, Init);

    if (!GlobalIndex)
      return false;

    return !Init || (checkDecl() && initGlobal(*GlobalIndex));
  }
  // Local variables.
  InitLinkScope<Emitter> ILS(this, InitLink::Decl(VD));

  if (VarT) {
    unsigned Offset = this->allocateLocalPrimitive(
        VD, *VarT, VD->getType().isConstQualified(), nullptr, ScopeKind::Block,
        IsConstexprUnknown);
    if (Init) {
      // If this is a toplevel declaration, create a scope for the
      // initializer.
      if (Toplevel) {
        LocalScope<Emitter> Scope(this);
        if (!this->visit(Init))
          return false;
        return this->emitSetLocal(*VarT, Offset, VD) && Scope.destroyLocals();
      } else {
        if (!this->visit(Init))
          return false;
        return this->emitSetLocal(*VarT, Offset, VD);
      }
    }
  } else {
    if (std::optional<unsigned> Offset = this->allocateLocal(
            VD, VD->getType(), nullptr, ScopeKind::Block, IsConstexprUnknown)) {
      if (!Init)
        return true;

      if (!this->emitGetPtrLocal(*Offset, Init))
        return false;

      if (!visitInitializer(Init))
        return false;

      return this->emitFinishInitPop(Init);
    }
    return false;
  }
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::visitAPValue(const APValue &Val, PrimType ValType,
                                     const Expr *E) {
  assert(!DiscardResult);
  if (Val.isInt())
    return this->emitConst(Val.getInt(), ValType, E);
  else if (Val.isFloat()) {
    APFloat F = Val.getFloat();
    return this->emitFloat(F, E);
  }

  if (Val.isLValue()) {
    if (Val.isNullPointer())
      return this->emitNull(ValType, 0, nullptr, E);
    APValue::LValueBase Base = Val.getLValueBase();
    if (const Expr *BaseExpr = Base.dyn_cast<const Expr *>())
      return this->visit(BaseExpr);
    else if (const auto *VD = Base.dyn_cast<const ValueDecl *>()) {
      return this->visitDeclRef(VD, E);
    }
  } else if (Val.isMemberPointer()) {
    if (const ValueDecl *MemberDecl = Val.getMemberPointerDecl())
      return this->emitGetMemberPtr(MemberDecl, E);
    return this->emitNullMemberPtr(0, nullptr, E);
  }

  return false;
}

template <class Emitter>
bool Compiler<Emitter>::visitAPValueInitializer(const APValue &Val,
                                                const Expr *E, QualType T) {
  if (Val.isStruct()) {
    const Record *R = this->getRecord(T);
    assert(R);
    for (unsigned I = 0, N = Val.getStructNumFields(); I != N; ++I) {
      const APValue &F = Val.getStructField(I);
      const Record::Field *RF = R->getField(I);
      QualType FieldType = RF->Decl->getType();

      if (std::optional<PrimType> PT = classify(FieldType)) {
        if (!this->visitAPValue(F, *PT, E))
          return false;
        if (!this->emitInitField(*PT, RF->Offset, E))
          return false;
      } else {
        if (!this->emitGetPtrField(RF->Offset, E))
          return false;
        if (!this->visitAPValueInitializer(F, E, FieldType))
          return false;
        if (!this->emitPopPtr(E))
          return false;
      }
    }
    return true;
  } else if (Val.isUnion()) {
    const FieldDecl *UnionField = Val.getUnionField();
    const Record *R = this->getRecord(UnionField->getParent());
    assert(R);
    const APValue &F = Val.getUnionValue();
    const Record::Field *RF = R->getField(UnionField);
    PrimType T = classifyPrim(RF->Decl->getType());
    if (!this->visitAPValue(F, T, E))
      return false;
    return this->emitInitField(T, RF->Offset, E);
  } else if (Val.isArray()) {
    const auto *ArrType = T->getAsArrayTypeUnsafe();
    QualType ElemType = ArrType->getElementType();
    for (unsigned A = 0, AN = Val.getArraySize(); A != AN; ++A) {
      const APValue &Elem = Val.getArrayInitializedElt(A);
      if (std::optional<PrimType> ElemT = classify(ElemType)) {
        if (!this->visitAPValue(Elem, *ElemT, E))
          return false;
        if (!this->emitInitElem(*ElemT, A, E))
          return false;
      } else {
        if (!this->emitConstUint32(A, E))
          return false;
        if (!this->emitArrayElemPtrUint32(E))
          return false;
        if (!this->visitAPValueInitializer(Elem, E, ElemType))
          return false;
        if (!this->emitPopPtr(E))
          return false;
      }
    }
    return true;
  }
  // TODO: Other types.

  return false;
}

template <class Emitter>
bool Compiler<Emitter>::VisitBuiltinCallExpr(const CallExpr *E,
                                             unsigned BuiltinID) {

  if (BuiltinID == Builtin::BI__builtin_constant_p) {
    // Void argument is always invalid and harder to handle later.
    if (E->getArg(0)->getType()->isVoidType()) {
      if (DiscardResult)
        return true;
      return this->emitConst(0, E);
    }

    if (!this->emitStartSpeculation(E))
      return false;
    LabelTy EndLabel = this->getLabel();
    if (!this->speculate(E, EndLabel))
      return false;
    this->fallthrough(EndLabel);
    if (!this->emitEndSpeculation(E))
      return false;
    if (DiscardResult)
      return this->emitPop(classifyPrim(E), E);
    return true;
  }

  // For these, we're expected to ultimately return an APValue pointing
  // to the CallExpr. This is needed to get the correct codegen.
  if (BuiltinID == Builtin::BI__builtin___CFStringMakeConstantString ||
      BuiltinID == Builtin::BI__builtin___NSStringMakeConstantString ||
      BuiltinID == Builtin::BI__builtin_ptrauth_sign_constant ||
      BuiltinID == Builtin::BI__builtin_function_start) {
    if (DiscardResult)
      return true;
    return this->emitDummyPtr(E, E);
  }

  QualType ReturnType = E->getType();
  std::optional<PrimType> ReturnT = classify(E);

  // Non-primitive return type. Prepare storage.
  if (!Initializing && !ReturnT && !ReturnType->isVoidType()) {
    std::optional<unsigned> LocalIndex = allocateLocal(E);
    if (!LocalIndex)
      return false;
    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
  }

  if (!Context::isUnevaluatedBuiltin(BuiltinID)) {
    // Put arguments on the stack.
    for (const auto *Arg : E->arguments()) {
      if (!this->visit(Arg))
        return false;
    }
  }

  if (!this->emitCallBI(E, BuiltinID, E))
    return false;

  if (DiscardResult && !ReturnType->isVoidType()) {
    assert(ReturnT);
    return this->emitPop(*ReturnT, E);
  }

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitCallExpr(const CallExpr *E) {
  const FunctionDecl *FuncDecl = E->getDirectCallee();

  if (FuncDecl) {
    if (unsigned BuiltinID = FuncDecl->getBuiltinID())
      return VisitBuiltinCallExpr(E, BuiltinID);

    // Calls to replaceable operator new/operator delete.
    if (FuncDecl->isUsableAsGlobalAllocationFunctionInConstantEvaluation()) {
      if (FuncDecl->getDeclName().isAnyOperatorNew()) {
        return VisitBuiltinCallExpr(E, Builtin::BI__builtin_operator_new);
      } else {
        assert(FuncDecl->getDeclName().getCXXOverloadedOperator() == OO_Delete);
        return VisitBuiltinCallExpr(E, Builtin::BI__builtin_operator_delete);
      }
    }

    // Explicit calls to trivial destructors
    if (const auto *DD = dyn_cast<CXXDestructorDecl>(FuncDecl);
        DD && DD->isTrivial()) {
      const auto *MemberCall = cast<CXXMemberCallExpr>(E);
      if (!this->visit(MemberCall->getImplicitObjectArgument()))
        return false;
      return this->emitCheckDestruction(E) && this->emitEndLifetime(E) &&
             this->emitPopPtr(E);
    }
  }

  BlockScope<Emitter> CallScope(this, ScopeKind::Call);

  QualType ReturnType = E->getCallReturnType(Ctx.getASTContext());
  std::optional<PrimType> T = classify(ReturnType);
  bool HasRVO = !ReturnType->isVoidType() && !T;

  if (HasRVO) {
    if (DiscardResult) {
      // If we need to discard the return value but the function returns its
      // value via an RVO pointer, we need to create one such pointer just
      // for this call.
      if (std::optional<unsigned> LocalIndex = allocateLocal(E)) {
        if (!this->emitGetPtrLocal(*LocalIndex, E))
          return false;
      }
    } else {
      // We need the result. Prepare a pointer to return or
      // dup the current one.
      if (!Initializing) {
        if (std::optional<unsigned> LocalIndex = allocateLocal(E)) {
          if (!this->emitGetPtrLocal(*LocalIndex, E))
            return false;
        }
      }
      if (!this->emitDupPtr(E))
        return false;
    }
  }

  SmallVector<const Expr *, 8> Args(ArrayRef(E->getArgs(), E->getNumArgs()));

  bool IsAssignmentOperatorCall = false;
  if (const auto *OCE = dyn_cast<CXXOperatorCallExpr>(E);
      OCE && OCE->isAssignmentOp()) {
    // Just like with regular assignments, we need to special-case assignment
    // operators here and evaluate the RHS (the second arg) before the LHS (the
    // first arg). We fix this by using a Flip op later.
    assert(Args.size() == 2);
    IsAssignmentOperatorCall = true;
    std::reverse(Args.begin(), Args.end());
  }
  // Calling a static operator will still
  // pass the instance, but we don't need it.
  // Discard it here.
  if (isa<CXXOperatorCallExpr>(E)) {
    if (const auto *MD = dyn_cast_if_present<CXXMethodDecl>(FuncDecl);
        MD && MD->isStatic()) {
      if (!this->discard(E->getArg(0)))
        return false;
      // Drop first arg.
      Args.erase(Args.begin());
    }
  }

  std::optional<unsigned> CalleeOffset;
  // Add the (optional, implicit) This pointer.
  if (const auto *MC = dyn_cast<CXXMemberCallExpr>(E)) {
    if (!FuncDecl && classifyPrim(E->getCallee()) == PT_MemberPtr) {
      // If we end up creating a CallPtr op for this, we need the base of the
      // member pointer as the instance pointer, and later extract the function
      // decl as the function pointer.
      const Expr *Callee = E->getCallee();
      CalleeOffset =
          this->allocateLocalPrimitive(Callee, PT_MemberPtr, /*IsConst=*/true);
      if (!this->visit(Callee))
        return false;
      if (!this->emitSetLocal(PT_MemberPtr, *CalleeOffset, E))
        return false;
      if (!this->emitGetLocal(PT_MemberPtr, *CalleeOffset, E))
        return false;
      if (!this->emitGetMemberPtrBase(E))
        return false;
    } else if (!this->visit(MC->getImplicitObjectArgument())) {
      return false;
    }
  } else if (const auto *PD =
                 dyn_cast<CXXPseudoDestructorExpr>(E->getCallee())) {
    if (!this->emitCheckPseudoDtor(E))
      return false;
    const Expr *Base = PD->getBase();
    if (!Base->isGLValue())
      return this->discard(Base);
    if (!this->visit(Base))
      return false;
    return this->emitEndLifetimePop(E);
  } else if (!FuncDecl) {
    const Expr *Callee = E->getCallee();
    CalleeOffset =
        this->allocateLocalPrimitive(Callee, PT_Ptr, /*IsConst=*/true);
    if (!this->visit(Callee))
      return false;
    if (!this->emitSetLocal(PT_Ptr, *CalleeOffset, E))
      return false;
  }

  if (!this->visitCallArgs(Args, FuncDecl))
    return false;

  // Undo the argument reversal we did earlier.
  if (IsAssignmentOperatorCall) {
    assert(Args.size() == 2);
    PrimType Arg1T = classify(Args[0]).value_or(PT_Ptr);
    PrimType Arg2T = classify(Args[1]).value_or(PT_Ptr);
    if (!this->emitFlip(Arg2T, Arg1T, E))
      return false;
  }

  if (FuncDecl) {
    const Function *Func = getFunction(FuncDecl);
    if (!Func)
      return false;
    assert(HasRVO == Func->hasRVO());

    bool HasQualifier = false;
    if (const auto *ME = dyn_cast<MemberExpr>(E->getCallee()))
      HasQualifier = ME->hasQualifier();

    bool IsVirtual = false;
    if (const auto *MD = dyn_cast<CXXMethodDecl>(FuncDecl))
      IsVirtual = MD->isVirtual();

    // In any case call the function. The return value will end up on the stack
    // and if the function has RVO, we already have the pointer on the stack to
    // write the result into.
    if (IsVirtual && !HasQualifier) {
      uint32_t VarArgSize = 0;
      unsigned NumParams =
          Func->getNumWrittenParams() + isa<CXXOperatorCallExpr>(E);
      for (unsigned I = NumParams, N = E->getNumArgs(); I != N; ++I)
        VarArgSize += align(primSize(classify(E->getArg(I)).value_or(PT_Ptr)));

      if (!this->emitCallVirt(Func, VarArgSize, E))
        return false;
    } else if (Func->isVariadic()) {
      uint32_t VarArgSize = 0;
      unsigned NumParams =
          Func->getNumWrittenParams() + isa<CXXOperatorCallExpr>(E);
      for (unsigned I = NumParams, N = E->getNumArgs(); I != N; ++I)
        VarArgSize += align(primSize(classify(E->getArg(I)).value_or(PT_Ptr)));
      if (!this->emitCallVar(Func, VarArgSize, E))
        return false;
    } else {
      if (!this->emitCall(Func, 0, E))
        return false;
    }
  } else {
    // Indirect call. Visit the callee, which will leave a FunctionPointer on
    // the stack. Cleanup of the returned value if necessary will be done after
    // the function call completed.

    // Sum the size of all args from the call expr.
    uint32_t ArgSize = 0;
    for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
      ArgSize += align(primSize(classify(E->getArg(I)).value_or(PT_Ptr)));

    // Get the callee, either from a member pointer or function pointer saved in
    // CalleeOffset.
    if (isa<CXXMemberCallExpr>(E) && CalleeOffset) {
      if (!this->emitGetLocal(PT_MemberPtr, *CalleeOffset, E))
        return false;
      if (!this->emitGetMemberPtrDecl(E))
        return false;
    } else {
      if (!this->emitGetLocal(PT_Ptr, *CalleeOffset, E))
        return false;
    }
    if (!this->emitCallPtr(ArgSize, E, E))
      return false;
  }

  // Cleanup for discarded return values.
  if (DiscardResult && !ReturnType->isVoidType() && T)
    return this->emitPop(*T, E) && CallScope.destroyLocals();

  return CallScope.destroyLocals();
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *E) {
  SourceLocScope<Emitter> SLS(this, E);

  return this->delegate(E->getExpr());
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *E) {
  SourceLocScope<Emitter> SLS(this, E);

  return this->delegate(E->getExpr());
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
  if (DiscardResult)
    return true;

  return this->emitConstBool(E->getValue(), E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXNullPtrLiteralExpr(
    const CXXNullPtrLiteralExpr *E) {
  if (DiscardResult)
    return true;

  uint64_t Val = Ctx.getASTContext().getTargetNullPointerValue(E->getType());
  return this->emitNullPtr(Val, nullptr, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitGNUNullExpr(const GNUNullExpr *E) {
  if (DiscardResult)
    return true;

  assert(E->getType()->isIntegerType());

  PrimType T = classifyPrim(E->getType());
  return this->emitZero(T, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitCXXThisExpr(const CXXThisExpr *E) {
  if (DiscardResult)
    return true;

  if (this->LambdaThisCapture.Offset > 0) {
    if (this->LambdaThisCapture.IsPtr)
      return this->emitGetThisFieldPtr(this->LambdaThisCapture.Offset, E);
    return this->emitGetPtrThisField(this->LambdaThisCapture.Offset, E);
  }

  // In some circumstances, the 'this' pointer does not actually refer to the
  // instance pointer of the current function frame, but e.g. to the declaration
  // currently being initialized. Here we emit the necessary instruction(s) for
  // this scenario.
  if (!InitStackActive)
    return this->emitThis(E);

  if (!InitStack.empty()) {
    // If our init stack is, for example:
    // 0 Stack: 3 (decl)
    // 1 Stack: 6 (init list)
    // 2 Stack: 1 (field)
    // 3 Stack: 6 (init list)
    // 4 Stack: 1 (field)
    //
    // We want to find the LAST element in it that's an init list,
    // which is marked with the K_InitList marker. The index right
    // before that points to an init list. We need to find the
    // elements before the K_InitList element that point to a base
    // (e.g. a decl or This), optionally followed by field, elem, etc.
    // In the example above, we want to emit elements [0..2].
    unsigned StartIndex = 0;
    unsigned EndIndex = 0;
    // Find the init list.
    for (StartIndex = InitStack.size() - 1; StartIndex > 0; --StartIndex) {
      if (InitStack[StartIndex].Kind == InitLink::K_InitList ||
          InitStack[StartIndex].Kind == InitLink::K_This) {
        EndIndex = StartIndex;
        --StartIndex;
        break;
      }
    }

    // Walk backwards to find the base.
    for (; StartIndex > 0; --StartIndex) {
      if (InitStack[StartIndex].Kind == InitLink::K_InitList)
        continue;

      if (InitStack[StartIndex].Kind != InitLink::K_Field &&
          InitStack[StartIndex].Kind != InitLink::K_Elem)
        break;
    }

    // Emit the instructions.
    for (unsigned I = StartIndex; I != EndIndex; ++I) {
      if (InitStack[I].Kind == InitLink::K_InitList)
        continue;
      if (!InitStack[I].template emit<Emitter>(this, E))
        return false;
    }
    return true;
  }
  return this->emitThis(E);
}

template <class Emitter> bool Compiler<Emitter>::visitStmt(const Stmt *S) {
  switch (S->getStmtClass()) {
  case Stmt::CompoundStmtClass:
    return visitCompoundStmt(cast<CompoundStmt>(S));
  case Stmt::DeclStmtClass:
    return visitDeclStmt(cast<DeclStmt>(S), /*EvaluateConditionDecl=*/true);
  case Stmt::ReturnStmtClass:
    return visitReturnStmt(cast<ReturnStmt>(S));
  case Stmt::IfStmtClass:
    return visitIfStmt(cast<IfStmt>(S));
  case Stmt::WhileStmtClass:
    return visitWhileStmt(cast<WhileStmt>(S));
  case Stmt::DoStmtClass:
    return visitDoStmt(cast<DoStmt>(S));
  case Stmt::ForStmtClass:
    return visitForStmt(cast<ForStmt>(S));
  case Stmt::CXXForRangeStmtClass:
    return visitCXXForRangeStmt(cast<CXXForRangeStmt>(S));
  case Stmt::BreakStmtClass:
    return visitBreakStmt(cast<BreakStmt>(S));
  case Stmt::ContinueStmtClass:
    return visitContinueStmt(cast<ContinueStmt>(S));
  case Stmt::SwitchStmtClass:
    return visitSwitchStmt(cast<SwitchStmt>(S));
  case Stmt::CaseStmtClass:
    return visitCaseStmt(cast<CaseStmt>(S));
  case Stmt::DefaultStmtClass:
    return visitDefaultStmt(cast<DefaultStmt>(S));
  case Stmt::AttributedStmtClass:
    return visitAttributedStmt(cast<AttributedStmt>(S));
  case Stmt::CXXTryStmtClass:
    return visitCXXTryStmt(cast<CXXTryStmt>(S));
  case Stmt::NullStmtClass:
    return true;
  // Always invalid statements.
  case Stmt::GCCAsmStmtClass:
  case Stmt::MSAsmStmtClass:
  case Stmt::GotoStmtClass:
    return this->emitInvalid(S);
  case Stmt::LabelStmtClass:
    return this->visitStmt(cast<LabelStmt>(S)->getSubStmt());
  default: {
    if (const auto *E = dyn_cast<Expr>(S))
      return this->discard(E);
    return false;
  }
  }
}

template <class Emitter>
bool Compiler<Emitter>::visitCompoundStmt(const CompoundStmt *S) {
  BlockScope<Emitter> Scope(this);
  for (const auto *InnerStmt : S->body())
    if (!visitStmt(InnerStmt))
      return false;
  return Scope.destroyLocals();
}

template <class Emitter>
bool Compiler<Emitter>::maybeEmitDeferredVarInit(const VarDecl *VD) {
  if (auto *DD = dyn_cast_if_present<DecompositionDecl>(VD)) {
    for (auto *BD : DD->flat_bindings())
      if (auto *KD = BD->getHoldingVar(); KD && !this->visitVarDecl(KD))
        return false;
  }
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::visitDeclStmt(const DeclStmt *DS,
                                      bool EvaluateConditionDecl) {
  for (const auto *D : DS->decls()) {
    if (isa<StaticAssertDecl, TagDecl, TypedefNameDecl, BaseUsingDecl,
            FunctionDecl, NamespaceAliasDecl, UsingDirectiveDecl>(D))
      continue;

    const auto *VD = dyn_cast<VarDecl>(D);
    if (!VD)
      return false;
    if (!this->visitVarDecl(VD))
      return false;

    // Register decomposition decl holding vars.
    if (EvaluateConditionDecl && !this->maybeEmitDeferredVarInit(VD))
      return false;
  }

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::visitReturnStmt(const ReturnStmt *RS) {
  if (this->InStmtExpr)
    return this->emitUnsupported(RS);

  if (const Expr *RE = RS->getRetValue()) {
    LocalScope<Emitter> RetScope(this);
    if (ReturnType) {
      // Primitive types are simply returned.
      if (!this->visit(RE))
        return false;
      this->emitCleanup();
      return this->emitRet(*ReturnType, RS);
    } else if (RE->getType()->isVoidType()) {
      if (!this->visit(RE))
        return false;
    } else {
      InitLinkScope<Emitter> ILS(this, InitLink::RVO());
      // RVO - construct the value in the return location.
      if (!this->emitRVOPtr(RE))
        return false;
      if (!this->visitInitializer(RE))
        return false;
      if (!this->emitPopPtr(RE))
        return false;

      this->emitCleanup();
      return this->emitRetVoid(RS);
    }
  }

  // Void return.
  this->emitCleanup();
  return this->emitRetVoid(RS);
}

template <class Emitter> bool Compiler<Emitter>::visitIfStmt(const IfStmt *IS) {
  auto visitChildStmt = [&](const Stmt *S) -> bool {
    LocalScope<Emitter> SScope(this);
    if (!visitStmt(S))
      return false;
    return SScope.destroyLocals();
  };
  if (auto *CondInit = IS->getInit())
    if (!visitStmt(CondInit))
      return false;

  if (const DeclStmt *CondDecl = IS->getConditionVariableDeclStmt())
    if (!visitDeclStmt(CondDecl))
      return false;

  // Save ourselves compiling some code and the jumps, etc. if the condition is
  // stataically known to be either true or false. We could look at more cases
  // here, but I think all the ones that actually happen are using a
  // ConstantExpr.
  if (std::optional<bool> BoolValue = getBoolValue(IS->getCond())) {
    if (*BoolValue)
      return visitChildStmt(IS->getThen());
    else if (const Stmt *Else = IS->getElse())
      return visitChildStmt(Else);
    return true;
  }

  // Otherwise, compile the condition.
  if (IS->isNonNegatedConsteval()) {
    if (!this->emitIsConstantContext(IS))
      return false;
  } else if (IS->isNegatedConsteval()) {
    if (!this->emitIsConstantContext(IS))
      return false;
    if (!this->emitInv(IS))
      return false;
  } else {
    if (!this->visitBool(IS->getCond()))
      return false;
  }

  if (!this->maybeEmitDeferredVarInit(IS->getConditionVariable()))
    return false;

  if (const Stmt *Else = IS->getElse()) {
    LabelTy LabelElse = this->getLabel();
    LabelTy LabelEnd = this->getLabel();
    if (!this->jumpFalse(LabelElse))
      return false;
    if (!visitChildStmt(IS->getThen()))
      return false;
    if (!this->jump(LabelEnd))
      return false;
    this->emitLabel(LabelElse);
    if (!visitChildStmt(Else))
      return false;
    this->emitLabel(LabelEnd);
  } else {
    LabelTy LabelEnd = this->getLabel();
    if (!this->jumpFalse(LabelEnd))
      return false;
    if (!visitChildStmt(IS->getThen()))
      return false;
    this->emitLabel(LabelEnd);
  }

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::visitWhileStmt(const WhileStmt *S) {
  const Expr *Cond = S->getCond();
  const Stmt *Body = S->getBody();

  LabelTy CondLabel = this->getLabel(); // Label before the condition.
  LabelTy EndLabel = this->getLabel();  // Label after the loop.
  LoopScope<Emitter> LS(this, EndLabel, CondLabel);

  this->fallthrough(CondLabel);
  this->emitLabel(CondLabel);

  {
    LocalScope<Emitter> CondScope(this);
    if (const DeclStmt *CondDecl = S->getConditionVariableDeclStmt())
      if (!visitDeclStmt(CondDecl))
        return false;

    if (!this->visitBool(Cond))
      return false;

    if (!this->maybeEmitDeferredVarInit(S->getConditionVariable()))
      return false;

    if (!this->jumpFalse(EndLabel))
      return false;

    if (!this->visitStmt(Body))
      return false;

    if (!CondScope.destroyLocals())
      return false;
  }
  if (!this->jump(CondLabel))
    return false;
  this->fallthrough(EndLabel);
  this->emitLabel(EndLabel);

  return true;
}

template <class Emitter> bool Compiler<Emitter>::visitDoStmt(const DoStmt *S) {
  const Expr *Cond = S->getCond();
  const Stmt *Body = S->getBody();

  LabelTy StartLabel = this->getLabel();
  LabelTy EndLabel = this->getLabel();
  LabelTy CondLabel = this->getLabel();
  LoopScope<Emitter> LS(this, EndLabel, CondLabel);

  this->fallthrough(StartLabel);
  this->emitLabel(StartLabel);

  {
    LocalScope<Emitter> CondScope(this);
    if (!this->visitStmt(Body))
      return false;
    this->fallthrough(CondLabel);
    this->emitLabel(CondLabel);
    if (!this->visitBool(Cond))
      return false;

    if (!CondScope.destroyLocals())
      return false;
  }
  if (!this->jumpTrue(StartLabel))
    return false;

  this->fallthrough(EndLabel);
  this->emitLabel(EndLabel);
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::visitForStmt(const ForStmt *S) {
  // for (Init; Cond; Inc) { Body }
  const Stmt *Init = S->getInit();
  const Expr *Cond = S->getCond();
  const Expr *Inc = S->getInc();
  const Stmt *Body = S->getBody();

  LabelTy EndLabel = this->getLabel();
  LabelTy CondLabel = this->getLabel();
  LabelTy IncLabel = this->getLabel();
  LoopScope<Emitter> LS(this, EndLabel, IncLabel);

  if (Init && !this->visitStmt(Init))
    return false;

  this->fallthrough(CondLabel);
  this->emitLabel(CondLabel);

  // Start of loop body.
  LocalScope<Emitter> CondScope(this);
  if (const DeclStmt *CondDecl = S->getConditionVariableDeclStmt())
    if (!visitDeclStmt(CondDecl))
      return false;

  if (Cond) {
    if (!this->visitBool(Cond))
      return false;
    if (!this->jumpFalse(EndLabel))
      return false;
  }
  if (!this->maybeEmitDeferredVarInit(S->getConditionVariable()))
    return false;

  if (Body && !this->visitStmt(Body))
    return false;

  this->fallthrough(IncLabel);
  this->emitLabel(IncLabel);
  if (Inc && !this->discard(Inc))
    return false;

  if (!CondScope.destroyLocals())
    return false;
  if (!this->jump(CondLabel))
    return false;
  // End of loop body.

  this->emitLabel(EndLabel);
  // If we jumped out of the loop above, we still need to clean up the condition
  // scope.
  return CondScope.destroyLocals();
}

template <class Emitter>
bool Compiler<Emitter>::visitCXXForRangeStmt(const CXXForRangeStmt *S) {
  const Stmt *Init = S->getInit();
  const Expr *Cond = S->getCond();
  const Expr *Inc = S->getInc();
  const Stmt *Body = S->getBody();
  const Stmt *BeginStmt = S->getBeginStmt();
  const Stmt *RangeStmt = S->getRangeStmt();
  const Stmt *EndStmt = S->getEndStmt();
  const VarDecl *LoopVar = S->getLoopVariable();

  LabelTy EndLabel = this->getLabel();
  LabelTy CondLabel = this->getLabel();
  LabelTy IncLabel = this->getLabel();
  LoopScope<Emitter> LS(this, EndLabel, IncLabel);

  // Emit declarations needed in the loop.
  if (Init && !this->visitStmt(Init))
    return false;
  if (!this->visitStmt(RangeStmt))
    return false;
  if (!this->visitStmt(BeginStmt))
    return false;
  if (!this->visitStmt(EndStmt))
    return false;

  // Now the condition as well as the loop variable assignment.
  this->fallthrough(CondLabel);
  this->emitLabel(CondLabel);
  if (!this->visitBool(Cond))
    return false;
  if (!this->jumpFalse(EndLabel))
    return false;

  if (!this->visitVarDecl(LoopVar))
    return false;

  // Body.
  {
    if (!this->visitStmt(Body))
      return false;

    this->fallthrough(IncLabel);
    this->emitLabel(IncLabel);
    if (!this->discard(Inc))
      return false;
  }

  if (!this->jump(CondLabel))
    return false;

  this->fallthrough(EndLabel);
  this->emitLabel(EndLabel);
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::visitBreakStmt(const BreakStmt *S) {
  if (!BreakLabel)
    return false;

  for (VariableScope<Emitter> *C = VarScope; C != BreakVarScope;
       C = C->getParent())
    C->emitDestruction();
  return this->jump(*BreakLabel);
}

template <class Emitter>
bool Compiler<Emitter>::visitContinueStmt(const ContinueStmt *S) {
  if (!ContinueLabel)
    return false;

  for (VariableScope<Emitter> *C = VarScope;
       C && C->getParent() != ContinueVarScope; C = C->getParent())
    C->emitDestruction();
  return this->jump(*ContinueLabel);
}

template <class Emitter>
bool Compiler<Emitter>::visitSwitchStmt(const SwitchStmt *S) {
  const Expr *Cond = S->getCond();
  PrimType CondT = this->classifyPrim(Cond->getType());
  LocalScope<Emitter> LS(this);

  LabelTy EndLabel = this->getLabel();
  OptLabelTy DefaultLabel = std::nullopt;
  unsigned CondVar =
      this->allocateLocalPrimitive(Cond, CondT, /*IsConst=*/true);

  if (const auto *CondInit = S->getInit())
    if (!visitStmt(CondInit))
      return false;

  if (const DeclStmt *CondDecl = S->getConditionVariableDeclStmt())
    if (!visitDeclStmt(CondDecl))
      return false;

  // Initialize condition variable.
  if (!this->visit(Cond))
    return false;
  if (!this->emitSetLocal(CondT, CondVar, S))
    return false;

  if (!this->maybeEmitDeferredVarInit(S->getConditionVariable()))
    return false;

  CaseMap CaseLabels;
  // Create labels and comparison ops for all case statements.
  for (const SwitchCase *SC = S->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase()) {
    if (const auto *CS = dyn_cast<CaseStmt>(SC)) {
      // FIXME: Implement ranges.
      if (CS->caseStmtIsGNURange())
        return false;
      CaseLabels[SC] = this->getLabel();

      const Expr *Value = CS->getLHS();
      PrimType ValueT = this->classifyPrim(Value->getType());

      // Compare the case statement's value to the switch condition.
      if (!this->emitGetLocal(CondT, CondVar, CS))
        return false;
      if (!this->visit(Value))
        return false;

      // Compare and jump to the case label.
      if (!this->emitEQ(ValueT, S))
        return false;
      if (!this->jumpTrue(CaseLabels[CS]))
        return false;
    } else {
      assert(!DefaultLabel);
      DefaultLabel = this->getLabel();
    }
  }

  // If none of the conditions above were true, fall through to the default
  // statement or jump after the switch statement.
  if (DefaultLabel) {
    if (!this->jump(*DefaultLabel))
      return false;
  } else {
    if (!this->jump(EndLabel))
      return false;
  }

  SwitchScope<Emitter> SS(this, std::move(CaseLabels), EndLabel, DefaultLabel);
  if (!this->visitStmt(S->getBody()))
    return false;
  this->emitLabel(EndLabel);

  return LS.destroyLocals();
}

template <class Emitter>
bool Compiler<Emitter>::visitCaseStmt(const CaseStmt *S) {
  this->emitLabel(CaseLabels[S]);
  return this->visitStmt(S->getSubStmt());
}

template <class Emitter>
bool Compiler<Emitter>::visitDefaultStmt(const DefaultStmt *S) {
  this->emitLabel(*DefaultLabel);
  return this->visitStmt(S->getSubStmt());
}

template <class Emitter>
bool Compiler<Emitter>::visitAttributedStmt(const AttributedStmt *S) {
  if (this->Ctx.getLangOpts().CXXAssumptions &&
      !this->Ctx.getLangOpts().MSVCCompat) {
    for (const Attr *A : S->getAttrs()) {
      auto *AA = dyn_cast<CXXAssumeAttr>(A);
      if (!AA)
        continue;

      assert(isa<NullStmt>(S->getSubStmt()));

      const Expr *Assumption = AA->getAssumption();
      if (Assumption->isValueDependent())
        return false;

      if (Assumption->HasSideEffects(this->Ctx.getASTContext()))
        continue;

      // Evaluate assumption.
      if (!this->visitBool(Assumption))
        return false;

      if (!this->emitAssume(Assumption))
        return false;
    }
  }

  // Ignore other attributes.
  return this->visitStmt(S->getSubStmt());
}

template <class Emitter>
bool Compiler<Emitter>::visitCXXTryStmt(const CXXTryStmt *S) {
  // Ignore all handlers.
  return this->visitStmt(S->getTryBlock());
}

template <class Emitter>
bool Compiler<Emitter>::emitLambdaStaticInvokerBody(const CXXMethodDecl *MD) {
  assert(MD->isLambdaStaticInvoker());
  assert(MD->hasBody());
  assert(cast<CompoundStmt>(MD->getBody())->body_empty());

  const CXXRecordDecl *ClosureClass = MD->getParent();
  const CXXMethodDecl *LambdaCallOp = ClosureClass->getLambdaCallOperator();
  assert(ClosureClass->captures_begin() == ClosureClass->captures_end());
  const Function *Func = this->getFunction(LambdaCallOp);
  if (!Func)
    return false;
  assert(Func->hasThisPointer());
  assert(Func->getNumParams() == (MD->getNumParams() + 1 + Func->hasRVO()));

  if (Func->hasRVO()) {
    if (!this->emitRVOPtr(MD))
      return false;
  }

  // The lambda call operator needs an instance pointer, but we don't have
  // one here, and we don't need one either because the lambda cannot have
  // any captures, as verified above. Emit a null pointer. This is then
  // special-cased when interpreting to not emit any misleading diagnostics.
  if (!this->emitNullPtr(0, nullptr, MD))
    return false;

  // Forward all arguments from the static invoker to the lambda call operator.
  for (const ParmVarDecl *PVD : MD->parameters()) {
    auto It = this->Params.find(PVD);
    assert(It != this->Params.end());

    // We do the lvalue-to-rvalue conversion manually here, so no need
    // to care about references.
    PrimType ParamType = this->classify(PVD->getType()).value_or(PT_Ptr);
    if (!this->emitGetParam(ParamType, It->second.Offset, MD))
      return false;
  }

  if (!this->emitCall(Func, 0, LambdaCallOp))
    return false;

  this->emitCleanup();
  if (ReturnType)
    return this->emitRet(*ReturnType, MD);

  // Nothing to do, since we emitted the RVO pointer above.
  return this->emitRetVoid(MD);
}

template <class Emitter>
bool Compiler<Emitter>::checkLiteralType(const Expr *E) {
  if (Ctx.getLangOpts().CPlusPlus23)
    return true;

  if (!E->isPRValue() || E->getType()->isLiteralType(Ctx.getASTContext()))
    return true;

  return this->emitCheckLiteralType(E->getType().getTypePtr(), E);
}

template <class Emitter>
bool Compiler<Emitter>::compileConstructor(const CXXConstructorDecl *Ctor) {
  assert(!ReturnType);

  auto emitFieldInitializer = [&](const Record::Field *F, unsigned FieldOffset,
                                  const Expr *InitExpr) -> bool {
    // We don't know what to do with these, so just return false.
    if (InitExpr->getType().isNull())
      return false;

    if (std::optional<PrimType> T = this->classify(InitExpr)) {
      if (!this->visit(InitExpr))
        return false;

      if (F->isBitField())
        return this->emitInitThisBitField(*T, F, FieldOffset, InitExpr);
      return this->emitInitThisField(*T, FieldOffset, InitExpr);
    }
    // Non-primitive case. Get a pointer to the field-to-initialize
    // on the stack and call visitInitialzer() for it.
    InitLinkScope<Emitter> FieldScope(this, InitLink::Field(F->Offset));
    if (!this->emitGetPtrThisField(FieldOffset, InitExpr))
      return false;

    if (!this->visitInitializer(InitExpr))
      return false;

    return this->emitFinishInitPop(InitExpr);
  };

  const RecordDecl *RD = Ctor->getParent();
  const Record *R = this->getRecord(RD);
  if (!R)
    return false;

  if (R->isUnion() && Ctor->isCopyOrMoveConstructor()) {
    // union copy and move ctors are special.
    assert(cast<CompoundStmt>(Ctor->getBody())->body_empty());
    if (!this->emitThis(Ctor))
      return false;

    auto PVD = Ctor->getParamDecl(0);
    ParamOffset PO = this->Params[PVD]; // Must exist.

    if (!this->emitGetParam(PT_Ptr, PO.Offset, Ctor))
      return false;

    return this->emitMemcpy(Ctor) && this->emitPopPtr(Ctor) &&
           this->emitRetVoid(Ctor);
  }

  InitLinkScope<Emitter> InitScope(this, InitLink::This());
  for (const auto *Init : Ctor->inits()) {
    // Scope needed for the initializers.
    BlockScope<Emitter> Scope(this);

    const Expr *InitExpr = Init->getInit();
    if (const FieldDecl *Member = Init->getMember()) {
      const Record::Field *F = R->getField(Member);

      if (!emitFieldInitializer(F, F->Offset, InitExpr))
        return false;
    } else if (const Type *Base = Init->getBaseClass()) {
      const auto *BaseDecl = Base->getAsCXXRecordDecl();
      assert(BaseDecl);

      if (Init->isBaseVirtual()) {
        assert(R->getVirtualBase(BaseDecl));
        if (!this->emitGetPtrThisVirtBase(BaseDecl, InitExpr))
          return false;

      } else {
        // Base class initializer.
        // Get This Base and call initializer on it.
        const Record::Base *B = R->getBase(BaseDecl);
        assert(B);
        if (!this->emitGetPtrThisBase(B->Offset, InitExpr))
          return false;
      }

      if (!this->visitInitializer(InitExpr))
        return false;
      if (!this->emitFinishInitPop(InitExpr))
        return false;
    } else if (const IndirectFieldDecl *IFD = Init->getIndirectMember()) {
      assert(IFD->getChainingSize() >= 2);

      unsigned NestedFieldOffset = 0;
      const Record::Field *NestedField = nullptr;
      for (const NamedDecl *ND : IFD->chain()) {
        const auto *FD = cast<FieldDecl>(ND);
        const Record *FieldRecord = this->P.getOrCreateRecord(FD->getParent());
        assert(FieldRecord);

        NestedField = FieldRecord->getField(FD);
        assert(NestedField);

        NestedFieldOffset += NestedField->Offset;
      }
      assert(NestedField);

      if (!emitFieldInitializer(NestedField, NestedFieldOffset, InitExpr))
        return false;

      // Mark all chain links as initialized.
      unsigned InitFieldOffset = 0;
      for (const NamedDecl *ND : IFD->chain().drop_back()) {
        const auto *FD = cast<FieldDecl>(ND);
        const Record *FieldRecord = this->P.getOrCreateRecord(FD->getParent());
        assert(FieldRecord);
        NestedField = FieldRecord->getField(FD);
        InitFieldOffset += NestedField->Offset;
        assert(NestedField);
        if (!this->emitGetPtrThisField(InitFieldOffset, InitExpr))
          return false;
        if (!this->emitFinishInitPop(InitExpr))
          return false;
      }

    } else {
      assert(Init->isDelegatingInitializer());
      if (!this->emitThis(InitExpr))
        return false;
      if (!this->visitInitializer(Init->getInit()))
        return false;
      if (!this->emitPopPtr(InitExpr))
        return false;
    }

    if (!Scope.destroyLocals())
      return false;
  }

  if (const auto *Body = Ctor->getBody())
    if (!visitStmt(Body))
      return false;

  return this->emitRetVoid(SourceInfo{});
}

template <class Emitter>
bool Compiler<Emitter>::compileDestructor(const CXXDestructorDecl *Dtor) {
  const RecordDecl *RD = Dtor->getParent();
  const Record *R = this->getRecord(RD);
  if (!R)
    return false;

  if (!Dtor->isTrivial() && Dtor->getBody()) {
    if (!this->visitStmt(Dtor->getBody()))
      return false;
  }

  if (!this->emitThis(Dtor))
    return false;

  if (!this->emitCheckDestruction(Dtor))
    return false;

  assert(R);
  if (!R->isUnion()) {
    // First, destroy all fields.
    for (const Record::Field &Field : llvm::reverse(R->fields())) {
      const Descriptor *D = Field.Desc;
      if (!D->isPrimitive() && !D->isPrimitiveArray()) {
        if (!this->emitGetPtrField(Field.Offset, SourceInfo{}))
          return false;
        if (!this->emitDestruction(D, SourceInfo{}))
          return false;
        if (!this->emitPopPtr(SourceInfo{}))
          return false;
      }
    }
  }

  for (const Record::Base &Base : llvm::reverse(R->bases())) {
    if (Base.R->isAnonymousUnion())
      continue;

    if (!this->emitGetPtrBase(Base.Offset, SourceInfo{}))
      return false;
    if (!this->emitRecordDestruction(Base.R, {}))
      return false;
    if (!this->emitPopPtr(SourceInfo{}))
      return false;
  }

  // FIXME: Virtual bases.
  return this->emitPopPtr(Dtor) && this->emitRetVoid(Dtor);
}

template <class Emitter>
bool Compiler<Emitter>::compileUnionAssignmentOperator(
    const CXXMethodDecl *MD) {
  if (!this->emitThis(MD))
    return false;

  auto PVD = MD->getParamDecl(0);
  ParamOffset PO = this->Params[PVD]; // Must exist.

  if (!this->emitGetParam(PT_Ptr, PO.Offset, MD))
    return false;

  return this->emitMemcpy(MD) && this->emitRet(PT_Ptr, MD);
}

template <class Emitter>
bool Compiler<Emitter>::visitFunc(const FunctionDecl *F) {
  // Classify the return type.
  ReturnType = this->classify(F->getReturnType());

  if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(F))
    return this->compileConstructor(Ctor);
  if (const auto *Dtor = dyn_cast<CXXDestructorDecl>(F))
    return this->compileDestructor(Dtor);

  // Emit custom code if this is a lambda static invoker.
  if (const auto *MD = dyn_cast<CXXMethodDecl>(F)) {
    const RecordDecl *RD = MD->getParent();

    if (RD->isUnion() &&
        (MD->isCopyAssignmentOperator() || MD->isMoveAssignmentOperator()))
      return this->compileUnionAssignmentOperator(MD);

    if (MD->isLambdaStaticInvoker())
      return this->emitLambdaStaticInvokerBody(MD);
  }

  // Regular functions.
  if (const auto *Body = F->getBody())
    if (!visitStmt(Body))
      return false;

  // Emit a guard return to protect against a code path missing one.
  if (F->getReturnType()->isVoidType())
    return this->emitRetVoid(SourceInfo{});
  return this->emitNoRet(SourceInfo{});
}

template <class Emitter>
bool Compiler<Emitter>::VisitUnaryOperator(const UnaryOperator *E) {
  const Expr *SubExpr = E->getSubExpr();
  if (SubExpr->getType()->isAnyComplexType())
    return this->VisitComplexUnaryOperator(E);
  if (SubExpr->getType()->isVectorType())
    return this->VisitVectorUnaryOperator(E);
  if (SubExpr->getType()->isFixedPointType())
    return this->VisitFixedPointUnaryOperator(E);
  std::optional<PrimType> T = classify(SubExpr->getType());

  switch (E->getOpcode()) {
  case UO_PostInc: { // x++
    if (!Ctx.getLangOpts().CPlusPlus14)
      return this->emitInvalid(E);
    if (!T)
      return this->emitError(E);

    if (!this->visit(SubExpr))
      return false;

    if (T == PT_Ptr) {
      if (!this->emitIncPtr(E))
        return false;

      return DiscardResult ? this->emitPopPtr(E) : true;
    }

    if (T == PT_Float) {
      return DiscardResult ? this->emitIncfPop(getFPOptions(E), E)
                           : this->emitIncf(getFPOptions(E), E);
    }

    return DiscardResult ? this->emitIncPop(*T, E->canOverflow(), E)
                         : this->emitInc(*T, E->canOverflow(), E);
  }
  case UO_PostDec: { // x--
    if (!Ctx.getLangOpts().CPlusPlus14)
      return this->emitInvalid(E);
    if (!T)
      return this->emitError(E);

    if (!this->visit(SubExpr))
      return false;

    if (T == PT_Ptr) {
      if (!this->emitDecPtr(E))
        return false;

      return DiscardResult ? this->emitPopPtr(E) : true;
    }

    if (T == PT_Float) {
      return DiscardResult ? this->emitDecfPop(getFPOptions(E), E)
                           : this->emitDecf(getFPOptions(E), E);
    }

    return DiscardResult ? this->emitDecPop(*T, E->canOverflow(), E)
                         : this->emitDec(*T, E->canOverflow(), E);
  }
  case UO_PreInc: { // ++x
    if (!Ctx.getLangOpts().CPlusPlus14)
      return this->emitInvalid(E);
    if (!T)
      return this->emitError(E);

    if (!this->visit(SubExpr))
      return false;

    if (T == PT_Ptr) {
      if (!this->emitLoadPtr(E))
        return false;
      if (!this->emitConstUint8(1, E))
        return false;
      if (!this->emitAddOffsetUint8(E))
        return false;
      return DiscardResult ? this->emitStorePopPtr(E) : this->emitStorePtr(E);
    }

    // Post-inc and pre-inc are the same if the value is to be discarded.
    if (DiscardResult) {
      if (T == PT_Float)
        return this->emitIncfPop(getFPOptions(E), E);
      return this->emitIncPop(*T, E->canOverflow(), E);
    }

    if (T == PT_Float) {
      const auto &TargetSemantics = Ctx.getFloatSemantics(E->getType());
      if (!this->emitLoadFloat(E))
        return false;
      APFloat F(TargetSemantics, 1);
      if (!this->emitFloat(F, E))
        return false;

      if (!this->emitAddf(getFPOptions(E), E))
        return false;
      if (!this->emitStoreFloat(E))
        return false;
    } else {
      assert(isIntegralType(*T));
      if (!this->emitPreInc(*T, E->canOverflow(), E))
        return false;
    }
    return E->isGLValue() || this->emitLoadPop(*T, E);
  }
  case UO_PreDec: { // --x
    if (!Ctx.getLangOpts().CPlusPlus14)
      return this->emitInvalid(E);
    if (!T)
      return this->emitError(E);

    if (!this->visit(SubExpr))
      return false;

    if (T == PT_Ptr) {
      if (!this->emitLoadPtr(E))
        return false;
      if (!this->emitConstUint8(1, E))
        return false;
      if (!this->emitSubOffsetUint8(E))
        return false;
      return DiscardResult ? this->emitStorePopPtr(E) : this->emitStorePtr(E);
    }

    // Post-dec and pre-dec are the same if the value is to be discarded.
    if (DiscardResult) {
      if (T == PT_Float)
        return this->emitDecfPop(getFPOptions(E), E);
      return this->emitDecPop(*T, E->canOverflow(), E);
    }

    if (T == PT_Float) {
      const auto &TargetSemantics = Ctx.getFloatSemantics(E->getType());
      if (!this->emitLoadFloat(E))
        return false;
      APFloat F(TargetSemantics, 1);
      if (!this->emitFloat(F, E))
        return false;

      if (!this->emitSubf(getFPOptions(E), E))
        return false;
      if (!this->emitStoreFloat(E))
        return false;
    } else {
      assert(isIntegralType(*T));
      if (!this->emitPreDec(*T, E->canOverflow(), E))
        return false;
    }
    return E->isGLValue() || this->emitLoadPop(*T, E);
  }
  case UO_LNot: // !x
    if (!T)
      return this->emitError(E);

    if (DiscardResult)
      return this->discard(SubExpr);

    if (!this->visitBool(SubExpr))
      return false;

    if (!this->emitInv(E))
      return false;

    if (PrimType ET = classifyPrim(E->getType()); ET != PT_Bool)
      return this->emitCast(PT_Bool, ET, E);
    return true;
  case UO_Minus: // -x
    if (!T)
      return this->emitError(E);

    if (!this->visit(SubExpr))
      return false;
    return DiscardResult ? this->emitPop(*T, E) : this->emitNeg(*T, E);
  case UO_Plus: // +x
    if (!T)
      return this->emitError(E);

    if (!this->visit(SubExpr)) // noop
      return false;
    return DiscardResult ? this->emitPop(*T, E) : true;
  case UO_AddrOf: // &x
    if (E->getType()->isMemberPointerType()) {
      // C++11 [expr.unary.op]p3 has very strict rules on how the address of a
      // member can be formed.
      return this->emitGetMemberPtr(cast<DeclRefExpr>(SubExpr)->getDecl(), E);
    }
    // We should already have a pointer when we get here.
    return this->delegate(SubExpr);
  case UO_Deref: // *x
    if (DiscardResult)
      return this->discard(SubExpr);

    if (!this->visit(SubExpr))
      return false;

    if (classifyPrim(SubExpr) == PT_Ptr)
      return this->emitNarrowPtr(E);
    return true;

  case UO_Not: // ~x
    if (!T)
      return this->emitError(E);

    if (!this->visit(SubExpr))
      return false;
    return DiscardResult ? this->emitPop(*T, E) : this->emitComp(*T, E);
  case UO_Real: // __real x
    assert(T);
    return this->delegate(SubExpr);
  case UO_Imag: { // __imag x
    assert(T);
    if (!this->discard(SubExpr))
      return false;
    return this->visitZeroInitializer(*T, SubExpr->getType(), SubExpr);
  }
  case UO_Extension:
    return this->delegate(SubExpr);
  case UO_Coawait:
    assert(false && "Unhandled opcode");
  }

  return false;
}

template <class Emitter>
bool Compiler<Emitter>::VisitComplexUnaryOperator(const UnaryOperator *E) {
  const Expr *SubExpr = E->getSubExpr();
  assert(SubExpr->getType()->isAnyComplexType());

  if (DiscardResult)
    return this->discard(SubExpr);

  std::optional<PrimType> ResT = classify(E);
  auto prepareResult = [=]() -> bool {
    if (!ResT && !Initializing) {
      std::optional<unsigned> LocalIndex = allocateLocal(SubExpr);
      if (!LocalIndex)
        return false;
      return this->emitGetPtrLocal(*LocalIndex, E);
    }

    return true;
  };

  // The offset of the temporary, if we created one.
  unsigned SubExprOffset = ~0u;
  auto createTemp = [=, &SubExprOffset]() -> bool {
    SubExprOffset =
        this->allocateLocalPrimitive(SubExpr, PT_Ptr, /*IsConst=*/true);
    if (!this->visit(SubExpr))
      return false;
    return this->emitSetLocal(PT_Ptr, SubExprOffset, E);
  };

  PrimType ElemT = classifyComplexElementType(SubExpr->getType());
  auto getElem = [=](unsigned Offset, unsigned Index) -> bool {
    if (!this->emitGetLocal(PT_Ptr, Offset, E))
      return false;
    return this->emitArrayElemPop(ElemT, Index, E);
  };

  switch (E->getOpcode()) {
  case UO_Minus:
    if (!prepareResult())
      return false;
    if (!createTemp())
      return false;
    for (unsigned I = 0; I != 2; ++I) {
      if (!getElem(SubExprOffset, I))
        return false;
      if (!this->emitNeg(ElemT, E))
        return false;
      if (!this->emitInitElem(ElemT, I, E))
        return false;
    }
    break;

  case UO_Plus:   // +x
  case UO_AddrOf: // &x
  case UO_Deref:  // *x
    return this->delegate(SubExpr);

  case UO_LNot:
    if (!this->visit(SubExpr))
      return false;
    if (!this->emitComplexBoolCast(SubExpr))
      return false;
    if (!this->emitInv(E))
      return false;
    if (PrimType ET = classifyPrim(E->getType()); ET != PT_Bool)
      return this->emitCast(PT_Bool, ET, E);
    return true;

  case UO_Real:
    return this->emitComplexReal(SubExpr);

  case UO_Imag:
    if (!this->visit(SubExpr))
      return false;

    if (SubExpr->isLValue()) {
      if (!this->emitConstUint8(1, E))
        return false;
      return this->emitArrayElemPtrPopUint8(E);
    }

    // Since our _Complex implementation does not map to a primitive type,
    // we sometimes have to do the lvalue-to-rvalue conversion here manually.
    return this->emitArrayElemPop(classifyPrim(E->getType()), 1, E);

  case UO_Not: // ~x
    if (!this->visit(SubExpr))
      return false;
    // Negate the imaginary component.
    if (!this->emitArrayElem(ElemT, 1, E))
      return false;
    if (!this->emitNeg(ElemT, E))
      return false;
    if (!this->emitInitElem(ElemT, 1, E))
      return false;
    return DiscardResult ? this->emitPopPtr(E) : true;

  case UO_Extension:
    return this->delegate(SubExpr);

  default:
    return this->emitInvalid(E);
  }

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitVectorUnaryOperator(const UnaryOperator *E) {
  const Expr *SubExpr = E->getSubExpr();
  assert(SubExpr->getType()->isVectorType());

  if (DiscardResult)
    return this->discard(SubExpr);

  auto UnaryOp = E->getOpcode();
  if (UnaryOp == UO_Extension)
    return this->delegate(SubExpr);

  if (UnaryOp != UO_Plus && UnaryOp != UO_Minus && UnaryOp != UO_LNot &&
      UnaryOp != UO_Not && UnaryOp != UO_AddrOf)
    return this->emitInvalid(E);

  // Nothing to do here.
  if (UnaryOp == UO_Plus || UnaryOp == UO_AddrOf)
    return this->delegate(SubExpr);

  if (!Initializing) {
    std::optional<unsigned> LocalIndex = allocateLocal(SubExpr);
    if (!LocalIndex)
      return false;
    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
  }

  // The offset of the temporary, if we created one.
  unsigned SubExprOffset =
      this->allocateLocalPrimitive(SubExpr, PT_Ptr, /*IsConst=*/true);
  if (!this->visit(SubExpr))
    return false;
  if (!this->emitSetLocal(PT_Ptr, SubExprOffset, E))
    return false;

  const auto *VecTy = SubExpr->getType()->getAs<VectorType>();
  PrimType ElemT = classifyVectorElementType(SubExpr->getType());
  auto getElem = [=](unsigned Offset, unsigned Index) -> bool {
    if (!this->emitGetLocal(PT_Ptr, Offset, E))
      return false;
    return this->emitArrayElemPop(ElemT, Index, E);
  };

  switch (UnaryOp) {
  case UO_Minus:
    for (unsigned I = 0; I != VecTy->getNumElements(); ++I) {
      if (!getElem(SubExprOffset, I))
        return false;
      if (!this->emitNeg(ElemT, E))
        return false;
      if (!this->emitInitElem(ElemT, I, E))
        return false;
    }
    break;
  case UO_LNot: { // !x
    // In C++, the logic operators !, &&, || are available for vectors. !v is
    // equivalent to v == 0.
    //
    // The result of the comparison is a vector of the same width and number of
    // elements as the comparison operands with a signed integral element type.
    //
    // https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html
    QualType ResultVecTy = E->getType();
    PrimType ResultVecElemT =
        classifyPrim(ResultVecTy->getAs<VectorType>()->getElementType());
    for (unsigned I = 0; I != VecTy->getNumElements(); ++I) {
      if (!getElem(SubExprOffset, I))
        return false;
      // operator ! on vectors returns -1 for 'truth', so negate it.
      if (!this->emitPrimCast(ElemT, PT_Bool, Ctx.getASTContext().BoolTy, E))
        return false;
      if (!this->emitInv(E))
        return false;
      if (!this->emitPrimCast(PT_Bool, ElemT, VecTy->getElementType(), E))
        return false;
      if (!this->emitNeg(ElemT, E))
        return false;
      if (ElemT != ResultVecElemT &&
          !this->emitPrimCast(ElemT, ResultVecElemT, ResultVecTy, E))
        return false;
      if (!this->emitInitElem(ResultVecElemT, I, E))
        return false;
    }
    break;
  }
  case UO_Not: // ~x
    for (unsigned I = 0; I != VecTy->getNumElements(); ++I) {
      if (!getElem(SubExprOffset, I))
        return false;
      if (ElemT == PT_Bool) {
        if (!this->emitInv(E))
          return false;
      } else {
        if (!this->emitComp(ElemT, E))
          return false;
      }
      if (!this->emitInitElem(ElemT, I, E))
        return false;
    }
    break;
  default:
    llvm_unreachable("Unsupported unary operators should be handled up front");
  }
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::visitDeclRef(const ValueDecl *D, const Expr *E) {
  if (DiscardResult)
    return true;

  if (const auto *ECD = dyn_cast<EnumConstantDecl>(D)) {
    return this->emitConst(ECD->getInitVal(), E);
  } else if (const auto *BD = dyn_cast<BindingDecl>(D)) {
    return this->visit(BD->getBinding());
  } else if (const auto *FuncDecl = dyn_cast<FunctionDecl>(D)) {
    const Function *F = getFunction(FuncDecl);
    return F && this->emitGetFnPtr(F, E);
  } else if (const auto *TPOD = dyn_cast<TemplateParamObjectDecl>(D)) {
    if (std::optional<unsigned> Index = P.getOrCreateGlobal(D)) {
      if (!this->emitGetPtrGlobal(*Index, E))
        return false;
      if (std::optional<PrimType> T = classify(E->getType())) {
        if (!this->visitAPValue(TPOD->getValue(), *T, E))
          return false;
        return this->emitInitGlobal(*T, *Index, E);
      }
      return this->visitAPValueInitializer(TPOD->getValue(), E,
                                           TPOD->getType());
    }
    return false;
  }

  // References are implemented via pointers, so when we see a DeclRefExpr
  // pointing to a reference, we need to get its value directly (i.e. the
  // pointer to the actual value) instead of a pointer to the pointer to the
  // value.
  bool IsReference = D->getType()->isReferenceType();

  // Check for local/global variables and parameters.
  if (auto It = Locals.find(D); It != Locals.end()) {
    const unsigned Offset = It->second.Offset;
    if (IsReference)
      return this->emitGetLocal(classifyPrim(E), Offset, E);
    return this->emitGetPtrLocal(Offset, E);
  } else if (auto GlobalIndex = P.getGlobal(D)) {
    if (IsReference) {
      if (!Ctx.getLangOpts().CPlusPlus11)
        return this->emitGetGlobal(classifyPrim(E), *GlobalIndex, E);
      return this->emitGetGlobalUnchecked(classifyPrim(E), *GlobalIndex, E);
    }

    return this->emitGetPtrGlobal(*GlobalIndex, E);
  } else if (const auto *PVD = dyn_cast<ParmVarDecl>(D)) {
    if (auto It = this->Params.find(PVD); It != this->Params.end()) {
      if (IsReference || !It->second.IsPtr)
        return this->emitGetParam(classifyPrim(E), It->second.Offset, E);

      return this->emitGetPtrParam(It->second.Offset, E);
    }
  }

  // In case we need to re-visit a declaration.
  auto revisit = [&](const VarDecl *VD) -> bool {
    if (!this->emitPushCC(VD->hasConstantInitialization(), E))
      return false;
    auto VarState = this->visitDecl(VD, /*IsConstexprUnknown=*/true);

    if (!this->emitPopCC(E))
      return false;

    if (VarState.notCreated())
      return true;
    if (!VarState)
      return false;
    // Retry.
    return this->visitDeclRef(D, E);
  };

  // Handle lambda captures.
  if (auto It = this->LambdaCaptures.find(D);
      It != this->LambdaCaptures.end()) {
    auto [Offset, IsPtr] = It->second;

    if (IsPtr)
      return this->emitGetThisFieldPtr(Offset, E);
    return this->emitGetPtrThisField(Offset, E);
  } else if (const auto *DRE = dyn_cast<DeclRefExpr>(E);
             DRE && DRE->refersToEnclosingVariableOrCapture()) {
    if (const auto *VD = dyn_cast<VarDecl>(D); VD && VD->isInitCapture())
      return revisit(VD);
  }

  // Avoid infinite recursion.
  if (D == InitializingDecl)
    return this->emitDummyPtr(D, E);

  // Try to lazily visit (or emit dummy pointers for) declarations
  // we haven't seen yet.
  // For C.
  if (!Ctx.getLangOpts().CPlusPlus) {
    if (const auto *VD = dyn_cast<VarDecl>(D);
        VD && VD->getAnyInitializer() &&
        VD->getType().isConstant(Ctx.getASTContext()) && !VD->isWeak())
      return revisit(VD);
    return this->emitDummyPtr(D, E);
  }

  // ... and C++.
  const auto *VD = dyn_cast<VarDecl>(D);
  if (!VD)
    return this->emitDummyPtr(D, E);

  const auto typeShouldBeVisited = [&](QualType T) -> bool {
    if (T.isConstant(Ctx.getASTContext()))
      return true;
    return T->isReferenceType();
  };

  if ((VD->hasGlobalStorage() || VD->isStaticDataMember()) &&
      typeShouldBeVisited(VD->getType())) {
    if (const Expr *Init = VD->getAnyInitializer();
        Init && !Init->isValueDependent()) {
      // Whether or not the evaluation is successul doesn't really matter
      // here -- we will create a global variable in any case, and that
      // will have the state of initializer evaluation attached.
      APValue V;
      SmallVector<PartialDiagnosticAt> Notes;
      (void)Init->EvaluateAsInitializer(V, Ctx.getASTContext(), VD, Notes,
                                        true);
      return this->visitDeclRef(D, E);
    }
    return revisit(VD);
  }

  // FIXME: The evaluateValue() check here is a little ridiculous, since
  // it will ultimately call into Context::evaluateAsInitializer(). In
  // other words, we're evaluating the initializer, just to know if we can
  // evaluate the initializer.
  if (VD->isLocalVarDecl() && typeShouldBeVisited(VD->getType()) &&
      VD->getInit() && !VD->getInit()->isValueDependent()) {

    if (VD->evaluateValue())
      return revisit(VD);

    if (!D->getType()->isReferenceType())
      return this->emitDummyPtr(D, E);

    return this->emitInvalidDeclRef(cast<DeclRefExpr>(E),
                                    /*InitializerFailed=*/true, E);
  }

  return this->emitDummyPtr(D, E);
}

template <class Emitter>
bool Compiler<Emitter>::VisitDeclRefExpr(const DeclRefExpr *E) {
  const auto *D = E->getDecl();
  return this->visitDeclRef(D, E);
}

template <class Emitter> void Compiler<Emitter>::emitCleanup() {
  for (VariableScope<Emitter> *C = VarScope; C; C = C->getParent())
    C->emitDestruction();
}

template <class Emitter>
unsigned Compiler<Emitter>::collectBaseOffset(const QualType BaseType,
                                              const QualType DerivedType) {
  const auto extractRecordDecl = [](QualType Ty) -> const CXXRecordDecl * {
    if (const auto *R = Ty->getPointeeCXXRecordDecl())
      return R;
    return Ty->getAsCXXRecordDecl();
  };
  const CXXRecordDecl *BaseDecl = extractRecordDecl(BaseType);
  const CXXRecordDecl *DerivedDecl = extractRecordDecl(DerivedType);

  return Ctx.collectBaseOffset(BaseDecl, DerivedDecl);
}

/// Emit casts from a PrimType to another PrimType.
template <class Emitter>
bool Compiler<Emitter>::emitPrimCast(PrimType FromT, PrimType ToT,
                                     QualType ToQT, const Expr *E) {

  if (FromT == PT_Float) {
    // Floating to floating.
    if (ToT == PT_Float) {
      const llvm::fltSemantics *ToSem = &Ctx.getFloatSemantics(ToQT);
      return this->emitCastFP(ToSem, getRoundingMode(E), E);
    }

    if (ToT == PT_IntAP)
      return this->emitCastFloatingIntegralAP(Ctx.getBitWidth(ToQT),
                                              getFPOptions(E), E);
    if (ToT == PT_IntAPS)
      return this->emitCastFloatingIntegralAPS(Ctx.getBitWidth(ToQT),
                                               getFPOptions(E), E);

    // Float to integral.
    if (isIntegralType(ToT) || ToT == PT_Bool)
      return this->emitCastFloatingIntegral(ToT, getFPOptions(E), E);
  }

  if (isIntegralType(FromT) || FromT == PT_Bool) {
    if (ToT == PT_IntAP)
      return this->emitCastAP(FromT, Ctx.getBitWidth(ToQT), E);
    if (ToT == PT_IntAPS)
      return this->emitCastAPS(FromT, Ctx.getBitWidth(ToQT), E);

    // Integral to integral.
    if (isIntegralType(ToT) || ToT == PT_Bool)
      return FromT != ToT ? this->emitCast(FromT, ToT, E) : true;

    if (ToT == PT_Float) {
      // Integral to floating.
      const llvm::fltSemantics *ToSem = &Ctx.getFloatSemantics(ToQT);
      return this->emitCastIntegralFloating(FromT, ToSem, getFPOptions(E), E);
    }
  }

  return false;
}

/// Emits __real(SubExpr)
template <class Emitter>
bool Compiler<Emitter>::emitComplexReal(const Expr *SubExpr) {
  assert(SubExpr->getType()->isAnyComplexType());

  if (DiscardResult)
    return this->discard(SubExpr);

  if (!this->visit(SubExpr))
    return false;
  if (SubExpr->isLValue()) {
    if (!this->emitConstUint8(0, SubExpr))
      return false;
    return this->emitArrayElemPtrPopUint8(SubExpr);
  }

  // Rvalue, load the actual element.
  return this->emitArrayElemPop(classifyComplexElementType(SubExpr->getType()),
                                0, SubExpr);
}

template <class Emitter>
bool Compiler<Emitter>::emitComplexBoolCast(const Expr *E) {
  assert(!DiscardResult);
  PrimType ElemT = classifyComplexElementType(E->getType());
  // We emit the expression (__real(E) != 0 || __imag(E) != 0)
  // for us, that means (bool)E[0] || (bool)E[1]
  if (!this->emitArrayElem(ElemT, 0, E))
    return false;
  if (ElemT == PT_Float) {
    if (!this->emitCastFloatingIntegral(PT_Bool, getFPOptions(E), E))
      return false;
  } else {
    if (!this->emitCast(ElemT, PT_Bool, E))
      return false;
  }

  // We now have the bool value of E[0] on the stack.
  LabelTy LabelTrue = this->getLabel();
  if (!this->jumpTrue(LabelTrue))
    return false;

  if (!this->emitArrayElemPop(ElemT, 1, E))
    return false;
  if (ElemT == PT_Float) {
    if (!this->emitCastFloatingIntegral(PT_Bool, getFPOptions(E), E))
      return false;
  } else {
    if (!this->emitCast(ElemT, PT_Bool, E))
      return false;
  }
  // Leave the boolean value of E[1] on the stack.
  LabelTy EndLabel = this->getLabel();
  this->jump(EndLabel);

  this->emitLabel(LabelTrue);
  if (!this->emitPopPtr(E))
    return false;
  if (!this->emitConstBool(true, E))
    return false;

  this->fallthrough(EndLabel);
  this->emitLabel(EndLabel);

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::emitComplexComparison(const Expr *LHS, const Expr *RHS,
                                              const BinaryOperator *E) {
  assert(E->isComparisonOp());
  assert(!Initializing);
  assert(!DiscardResult);

  PrimType ElemT;
  bool LHSIsComplex;
  unsigned LHSOffset;
  if (LHS->getType()->isAnyComplexType()) {
    LHSIsComplex = true;
    ElemT = classifyComplexElementType(LHS->getType());
    LHSOffset = allocateLocalPrimitive(LHS, PT_Ptr, /*IsConst=*/true);
    if (!this->visit(LHS))
      return false;
    if (!this->emitSetLocal(PT_Ptr, LHSOffset, E))
      return false;
  } else {
    LHSIsComplex = false;
    PrimType LHST = classifyPrim(LHS->getType());
    LHSOffset = this->allocateLocalPrimitive(LHS, LHST, /*IsConst=*/true);
    if (!this->visit(LHS))
      return false;
    if (!this->emitSetLocal(LHST, LHSOffset, E))
      return false;
  }

  bool RHSIsComplex;
  unsigned RHSOffset;
  if (RHS->getType()->isAnyComplexType()) {
    RHSIsComplex = true;
    ElemT = classifyComplexElementType(RHS->getType());
    RHSOffset = allocateLocalPrimitive(RHS, PT_Ptr, /*IsConst=*/true);
    if (!this->visit(RHS))
      return false;
    if (!this->emitSetLocal(PT_Ptr, RHSOffset, E))
      return false;
  } else {
    RHSIsComplex = false;
    PrimType RHST = classifyPrim(RHS->getType());
    RHSOffset = this->allocateLocalPrimitive(RHS, RHST, /*IsConst=*/true);
    if (!this->visit(RHS))
      return false;
    if (!this->emitSetLocal(RHST, RHSOffset, E))
      return false;
  }

  auto getElem = [&](unsigned LocalOffset, unsigned Index,
                     bool IsComplex) -> bool {
    if (IsComplex) {
      if (!this->emitGetLocal(PT_Ptr, LocalOffset, E))
        return false;
      return this->emitArrayElemPop(ElemT, Index, E);
    }
    return this->emitGetLocal(ElemT, LocalOffset, E);
  };

  for (unsigned I = 0; I != 2; ++I) {
    // Get both values.
    if (!getElem(LHSOffset, I, LHSIsComplex))
      return false;
    if (!getElem(RHSOffset, I, RHSIsComplex))
      return false;
    // And compare them.
    if (!this->emitEQ(ElemT, E))
      return false;

    if (!this->emitCastBoolUint8(E))
      return false;
  }

  // We now have two bool values on the stack. Compare those.
  if (!this->emitAddUint8(E))
    return false;
  if (!this->emitConstUint8(2, E))
    return false;

  if (E->getOpcode() == BO_EQ) {
    if (!this->emitEQUint8(E))
      return false;
  } else if (E->getOpcode() == BO_NE) {
    if (!this->emitNEUint8(E))
      return false;
  } else
    return false;

  // In C, this returns an int.
  if (PrimType ResT = classifyPrim(E->getType()); ResT != PT_Bool)
    return this->emitCast(PT_Bool, ResT, E);
  return true;
}

/// When calling this, we have a pointer of the local-to-destroy
/// on the stack.
/// Emit destruction of record types (or arrays of record types).
template <class Emitter>
bool Compiler<Emitter>::emitRecordDestruction(const Record *R, SourceInfo Loc) {
  assert(R);
  assert(!R->isAnonymousUnion());
  const CXXDestructorDecl *Dtor = R->getDestructor();
  if (!Dtor || Dtor->isTrivial())
    return true;

  assert(Dtor);
  const Function *DtorFunc = getFunction(Dtor);
  if (!DtorFunc)
    return false;
  assert(DtorFunc->hasThisPointer());
  assert(DtorFunc->getNumParams() == 1);
  if (!this->emitDupPtr(Loc))
    return false;
  return this->emitCall(DtorFunc, 0, Loc);
}
/// When calling this, we have a pointer of the local-to-destroy
/// on the stack.
/// Emit destruction of record types (or arrays of record types).
template <class Emitter>
bool Compiler<Emitter>::emitDestruction(const Descriptor *Desc,
                                        SourceInfo Loc) {
  assert(Desc);
  assert(!Desc->isPrimitive());
  assert(!Desc->isPrimitiveArray());

  // Can happen if the decl is invalid.
  if (Desc->isDummy())
    return true;

  // Arrays.
  if (Desc->isArray()) {
    const Descriptor *ElemDesc = Desc->ElemDesc;
    assert(ElemDesc);

    // Don't need to do anything for these.
    if (ElemDesc->isPrimitiveArray())
      return true;

    // If this is an array of record types, check if we need
    // to call the element destructors at all. If not, try
    // to save the work.
    if (const Record *ElemRecord = ElemDesc->ElemRecord) {
      if (const CXXDestructorDecl *Dtor = ElemRecord->getDestructor();
          !Dtor || Dtor->isTrivial())
        return true;
    }

    if (unsigned N = Desc->getNumElems()) {
      for (ssize_t I = N - 1; I >= 0; --I) {
        if (!this->emitConstUint64(I, Loc))
          return false;
        if (!this->emitArrayElemPtrUint64(Loc))
          return false;
        if (!this->emitDestruction(ElemDesc, Loc))
          return false;
        if (!this->emitPopPtr(Loc))
          return false;
      }
    }
    return true;
  }

  assert(Desc->ElemRecord);
  if (Desc->ElemRecord->isAnonymousUnion())
    return true;

  return this->emitRecordDestruction(Desc->ElemRecord, Loc);
}

/// Create a dummy pointer for the given decl (or expr) and
/// push a pointer to it on the stack.
template <class Emitter>
bool Compiler<Emitter>::emitDummyPtr(const DeclTy &D, const Expr *E) {
  assert(!DiscardResult && "Should've been checked before");

  unsigned DummyID = P.getOrCreateDummy(D);

  if (!this->emitGetPtrGlobal(DummyID, E))
    return false;
  if (E->getType()->isVoidType())
    return true;

  // Convert the dummy pointer to another pointer type if we have to.
  if (PrimType PT = classifyPrim(E); PT != PT_Ptr) {
    if (isPtrType(PT))
      return this->emitDecayPtr(PT_Ptr, PT, E);
    return false;
  }
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::emitFloat(const APFloat &F, const Expr *E) {
  assert(!DiscardResult && "Should've been checked before");

  if (Floating::singleWord(F.getSemantics()))
    return this->emitConstFloat(Floating(F), E);

  APInt I = F.bitcastToAPInt();
  return this->emitConstFloat(
      Floating(const_cast<uint64_t *>(I.getRawData()),
               llvm::APFloatBase::SemanticsToEnum(F.getSemantics())),
      E);
}

//  This function is constexpr if and only if To, From, and the types of
//  all subobjects of To and From are types T such that...
//  (3.1) - is_union_v<T> is false;
//  (3.2) - is_pointer_v<T> is false;
//  (3.3) - is_member_pointer_v<T> is false;
//  (3.4) - is_volatile_v<T> is false; and
//  (3.5) - T has no non-static data members of reference type
template <class Emitter>
bool Compiler<Emitter>::emitBuiltinBitCast(const CastExpr *E) {
  const Expr *SubExpr = E->getSubExpr();
  QualType FromType = SubExpr->getType();
  QualType ToType = E->getType();
  std::optional<PrimType> ToT = classify(ToType);

  assert(!ToType->isReferenceType());

  // Prepare storage for the result in case we discard.
  if (DiscardResult && !Initializing && !ToT) {
    std::optional<unsigned> LocalIndex = allocateLocal(E);
    if (!LocalIndex)
      return false;
    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
  }

  // Get a pointer to the value-to-cast on the stack.
  // For CK_LValueToRValueBitCast, this is always an lvalue and
  // we later assume it to be one (i.e. a PT_Ptr). However,
  // we call this function for other utility methods where
  // a bitcast might be useful, so convert it to a PT_Ptr in that case.
  if (SubExpr->isGLValue() || FromType->isVectorType()) {
    if (!this->visit(SubExpr))
      return false;
  } else if (std::optional<PrimType> FromT = classify(SubExpr)) {
    unsigned TempOffset =
        allocateLocalPrimitive(SubExpr, *FromT, /*IsConst=*/true);
    if (!this->visit(SubExpr))
      return false;
    if (!this->emitSetLocal(*FromT, TempOffset, E))
      return false;
    if (!this->emitGetPtrLocal(TempOffset, E))
      return false;
  } else {
    return false;
  }

  if (!ToT) {
    if (!this->emitBitCast(E))
      return false;
    return DiscardResult ? this->emitPopPtr(E) : true;
  }
  assert(ToT);

  const llvm::fltSemantics *TargetSemantics = nullptr;
  if (ToT == PT_Float)
    TargetSemantics = &Ctx.getFloatSemantics(ToType);

  // Conversion to a primitive type. FromType can be another
  // primitive type, or a record/array.
  bool ToTypeIsUChar = (ToType->isSpecificBuiltinType(BuiltinType::UChar) ||
                        ToType->isSpecificBuiltinType(BuiltinType::Char_U));
  uint32_t ResultBitWidth = std::max(Ctx.getBitWidth(ToType), 8u);

  if (!this->emitBitCastPrim(*ToT, ToTypeIsUChar || ToType->isStdByteType(),
                             ResultBitWidth, TargetSemantics, E))
    return false;

  if (DiscardResult)
    return this->emitPop(*ToT, E);

  return true;
}

namespace clang {
namespace interp {

template class Compiler<ByteCodeEmitter>;
template class Compiler<EvalEmitter>;

} // namespace interp
} // namespace clang
