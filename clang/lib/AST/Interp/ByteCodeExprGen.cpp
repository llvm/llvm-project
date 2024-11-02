//===--- ByteCodeExprGen.cpp - Code generator for expressions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ByteCodeExprGen.h"
#include "ByteCodeEmitter.h"
#include "ByteCodeGenError.h"
#include "ByteCodeStmtGen.h"
#include "Context.h"
#include "Floating.h"
#include "Function.h"
#include "PrimType.h"
#include "Program.h"

using namespace clang;
using namespace clang::interp;

using APSInt = llvm::APSInt;

namespace clang {
namespace interp {

/// Scope used to handle temporaries in toplevel variable declarations.
template <class Emitter> class DeclScope final : public VariableScope<Emitter> {
public:
  DeclScope(ByteCodeExprGen<Emitter> *Ctx, const ValueDecl *VD)
      : VariableScope<Emitter>(Ctx), Scope(Ctx->P, VD),
        OldGlobalDecl(Ctx->GlobalDecl) {
    Ctx->GlobalDecl = Context::shouldBeGloballyIndexed(VD);
  }

  void addExtended(const Scope::Local &Local) override {
    return this->addLocal(Local);
  }

  ~DeclScope() { this->Ctx->GlobalDecl = OldGlobalDecl; }

private:
  Program::DeclScope Scope;
  bool OldGlobalDecl;
};

/// Scope used to handle initialization methods.
template <class Emitter> class OptionScope final {
public:
  /// Root constructor, compiling or discarding primitives.
  OptionScope(ByteCodeExprGen<Emitter> *Ctx, bool NewDiscardResult,
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
  ByteCodeExprGen<Emitter> *Ctx;
  /// Old discard flag to restore.
  bool OldDiscardResult;
  bool OldInitializing;
};

} // namespace interp
} // namespace clang

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCastExpr(const CastExpr *CE) {
  const Expr *SubExpr = CE->getSubExpr();
  switch (CE->getCastKind()) {

  case CK_LValueToRValue: {
    return dereference(
        SubExpr, DerefKind::Read,
        [](PrimType) {
          // Value loaded - nothing to do here.
          return true;
        },
        [this, CE](PrimType T) {
          // Pointer on stack - dereference it.
          if (!this->emitLoadPop(T, CE))
            return false;
          return DiscardResult ? this->emitPop(T, CE) : true;
        });
  }

  case CK_UncheckedDerivedToBase:
  case CK_DerivedToBase: {
    if (!this->visit(SubExpr))
      return false;

    unsigned DerivedOffset = collectBaseOffset(getRecordTy(CE->getType()),
                                               getRecordTy(SubExpr->getType()));

    return this->emitGetPtrBasePop(DerivedOffset, CE);
  }

  case CK_BaseToDerived: {
    if (!this->visit(SubExpr))
      return false;

    unsigned DerivedOffset = collectBaseOffset(getRecordTy(SubExpr->getType()),
                                               getRecordTy(CE->getType()));

    return this->emitGetPtrDerivedPop(DerivedOffset, CE);
  }

  case CK_FloatingCast: {
    if (DiscardResult)
      return this->discard(SubExpr);
    if (!this->visit(SubExpr))
      return false;
    const auto *TargetSemantics = &Ctx.getFloatSemantics(CE->getType());
    return this->emitCastFP(TargetSemantics, getRoundingMode(CE), CE);
  }

  case CK_IntegralToFloating: {
    if (DiscardResult)
      return this->discard(SubExpr);
    std::optional<PrimType> FromT = classify(SubExpr->getType());
    if (!FromT)
      return false;

    if (!this->visit(SubExpr))
      return false;

    const auto *TargetSemantics = &Ctx.getFloatSemantics(CE->getType());
    llvm::RoundingMode RM = getRoundingMode(CE);
    return this->emitCastIntegralFloating(*FromT, TargetSemantics, RM, CE);
  }

  case CK_FloatingToBoolean:
  case CK_FloatingToIntegral: {
    if (DiscardResult)
      return this->discard(SubExpr);

    std::optional<PrimType> ToT = classify(CE->getType());

    if (!ToT)
      return false;

    if (!this->visit(SubExpr))
      return false;

    if (ToT == PT_IntAP)
      return this->emitCastFloatingIntegralAP(Ctx.getBitWidth(CE->getType()),
                                              CE);
    if (ToT == PT_IntAPS)
      return this->emitCastFloatingIntegralAPS(Ctx.getBitWidth(CE->getType()),
                                               CE);

    return this->emitCastFloatingIntegral(*ToT, CE);
  }

  case CK_NullToPointer:
    if (DiscardResult)
      return true;
    return this->emitNull(classifyPrim(CE->getType()), CE);

  case CK_PointerToIntegral: {
    // TODO: Discard handling.
    if (!this->visit(SubExpr))
      return false;

    PrimType T = classifyPrim(CE->getType());
    return this->emitCastPointerIntegral(T, CE);
  }

  case CK_ArrayToPointerDecay: {
    if (!this->visit(SubExpr))
      return false;
    if (!this->emitArrayDecay(CE))
      return false;
    if (DiscardResult)
      return this->emitPopPtr(CE);
    return true;
  }

  case CK_AtomicToNonAtomic:
  case CK_ConstructorConversion:
  case CK_FunctionToPointerDecay:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_BitCast:
    return this->delegate(SubExpr);

  case CK_IntegralToBoolean:
  case CK_IntegralCast: {
    if (DiscardResult)
      return this->discard(SubExpr);
    std::optional<PrimType> FromT = classify(SubExpr->getType());
    std::optional<PrimType> ToT = classify(CE->getType());

    if (!FromT || !ToT)
      return false;

    if (!this->visit(SubExpr))
      return false;

    if (ToT == PT_IntAP)
      return this->emitCastAP(*FromT, Ctx.getBitWidth(CE->getType()), CE);
    if (ToT == PT_IntAPS)
      return this->emitCastAPS(*FromT, Ctx.getBitWidth(CE->getType()), CE);

    if (FromT == ToT)
      return true;
    return this->emitCast(*FromT, *ToT, CE);
  }

  case CK_PointerToBoolean: {
    PrimType PtrT = classifyPrim(SubExpr->getType());

    // Just emit p != nullptr for this.
    if (!this->visit(SubExpr))
      return false;

    if (!this->emitNull(PtrT, CE))
      return false;

    return this->emitNE(PtrT, CE);
  }

  case CK_IntegralComplexToBoolean:
  case CK_FloatingComplexToBoolean: {
    std::optional<PrimType> ElemT =
        classifyComplexElementType(SubExpr->getType());
    if (!ElemT)
      return false;
    // We emit the expression (__real(E) != 0 || __imag(E) != 0)
    // for us, that means (bool)E[0] || (bool)E[1]
    if (!this->visit(SubExpr))
      return false;
    if (!this->emitConstUint8(0, CE))
      return false;
    if (!this->emitArrayElemPtrUint8(CE))
      return false;
    if (!this->emitLoadPop(*ElemT, CE))
      return false;
    if (*ElemT == PT_Float) {
      if (!this->emitCastFloatingIntegral(PT_Bool, CE))
        return false;
    } else {
      if (!this->emitCast(*ElemT, PT_Bool, CE))
        return false;
    }

    // We now have the bool value of E[0] on the stack.
    LabelTy LabelTrue = this->getLabel();
    if (!this->jumpTrue(LabelTrue))
      return false;

    if (!this->emitConstUint8(1, CE))
      return false;
    if (!this->emitArrayElemPtrPopUint8(CE))
      return false;
    if (!this->emitLoadPop(*ElemT, CE))
      return false;
    if (*ElemT == PT_Float) {
      if (!this->emitCastFloatingIntegral(PT_Bool, CE))
        return false;
    } else {
      if (!this->emitCast(*ElemT, PT_Bool, CE))
        return false;
    }
    // Leave the boolean value of E[1] on the stack.
    LabelTy EndLabel = this->getLabel();
    this->jump(EndLabel);

    this->emitLabel(LabelTrue);
    if (!this->emitPopPtr(CE))
      return false;
    if (!this->emitConstBool(true, CE))
      return false;

    this->fallthrough(EndLabel);
    this->emitLabel(EndLabel);

    return true;
  }

  case CK_IntegralComplexToReal:
  case CK_FloatingComplexToReal:
    return this->emitComplexReal(SubExpr);

  case CK_IntegralRealToComplex:
  case CK_FloatingRealToComplex: {
    // We're creating a complex value here, so we need to
    // allocate storage for it.
    if (!Initializing) {
      std::optional<unsigned> LocalIndex =
          allocateLocal(CE, /*IsExtended=*/true);
      if (!LocalIndex)
        return false;
      if (!this->emitGetPtrLocal(*LocalIndex, CE))
        return false;
    }

    // Init the complex value to {SubExpr, 0}.
    if (!this->visitArrayElemInit(0, SubExpr))
      return false;
    // Zero-init the second element.
    PrimType T = classifyPrim(SubExpr->getType());
    if (!this->visitZeroInitializer(T, SubExpr->getType(), SubExpr))
      return false;
    return this->emitInitElem(T, 1, SubExpr);
  }

  case CK_ToVoid:
    return discard(SubExpr);

  default:
    assert(false && "Cast not implemented");
  }
  llvm_unreachable("Unhandled clang::CastKind enum");
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitIntegerLiteral(const IntegerLiteral *LE) {
  if (DiscardResult)
    return true;

  return this->emitConst(LE->getValue(), LE);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitFloatingLiteral(const FloatingLiteral *E) {
  if (DiscardResult)
    return true;

  return this->emitConstFloat(E->getValue(), E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitParenExpr(const ParenExpr *E) {
  return this->delegate(E->getSubExpr());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitBinaryOperator(const BinaryOperator *BO) {
  // Need short-circuiting for these.
  if (BO->isLogicalOp())
    return this->VisitLogicalBinOp(BO);

  if (BO->getType()->isAnyComplexType())
    return this->VisitComplexBinOp(BO);

  const Expr *LHS = BO->getLHS();
  const Expr *RHS = BO->getRHS();

  if (BO->isPtrMemOp())
    return this->visit(RHS);

  // Typecheck the args.
  std::optional<PrimType> LT = classify(LHS->getType());
  std::optional<PrimType> RT = classify(RHS->getType());
  std::optional<PrimType> T = classify(BO->getType());

  // Deal with operations which have composite or void types.
  if (BO->isCommaOp()) {
    if (!this->discard(LHS))
      return false;
    if (RHS->getType()->isVoidType())
      return this->discard(RHS);

    return this->delegate(RHS);
  }

  // Special case for C++'s three-way/spaceship operator <=>, which
  // returns a std::{strong,weak,partial}_ordering (which is a class, so doesn't
  // have a PrimType).
  if (!T) {
    if (DiscardResult)
      return true;
    const ComparisonCategoryInfo *CmpInfo =
        Ctx.getASTContext().CompCategories.lookupInfoForType(BO->getType());
    assert(CmpInfo);

    // We need a temporary variable holding our return value.
    if (!Initializing) {
      std::optional<unsigned> ResultIndex = this->allocateLocal(BO, false);
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
    if (T == PT_Ptr || (LT == PT_Ptr && RT == PT_Ptr))
      return this->VisitPointerArithBinOp(BO);
  }

  if (!visit(LHS) || !visit(RHS))
    return false;

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
      return Discard(this->emitSubf(getRoundingMode(BO), BO));
    return Discard(this->emitSub(*T, BO));
  case BO_Add:
    if (BO->getType()->isFloatingType())
      return Discard(this->emitAddf(getRoundingMode(BO), BO));
    return Discard(this->emitAdd(*T, BO));
  case BO_Mul:
    if (BO->getType()->isFloatingType())
      return Discard(this->emitMulf(getRoundingMode(BO), BO));
    return Discard(this->emitMul(*T, BO));
  case BO_Rem:
    return Discard(this->emitRem(*T, BO));
  case BO_Div:
    if (BO->getType()->isFloatingType())
      return Discard(this->emitDivf(getRoundingMode(BO), BO));
    return Discard(this->emitDiv(*T, BO));
  case BO_Assign:
    if (DiscardResult)
      return LHS->refersToBitField() ? this->emitStoreBitFieldPop(*T, BO)
                                     : this->emitStorePop(*T, BO);
    return LHS->refersToBitField() ? this->emitStoreBitField(*T, BO)
                                   : this->emitStore(*T, BO);
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
bool ByteCodeExprGen<Emitter>::VisitPointerArithBinOp(const BinaryOperator *E) {
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

  if (LHS->getType()->isPointerType() && RHS->getType()->isPointerType()) {
    if (Op != BO_Sub)
      return false;

    assert(E->getType()->isIntegerType());
    if (!visit(RHS) || !visit(LHS))
      return false;

    return this->emitSubPtr(classifyPrim(E->getType()), E);
  }

  PrimType OffsetType;
  if (LHS->getType()->isIntegerType()) {
    if (!visit(RHS) || !visit(LHS))
      return false;
    OffsetType = *LT;
  } else if (RHS->getType()->isIntegerType()) {
    if (!visit(LHS) || !visit(RHS))
      return false;
    OffsetType = *RT;
  } else {
    return false;
  }

  if (Op == BO_Add)
    return this->emitAddOffset(OffsetType, E);
  else if (Op == BO_Sub)
    return this->emitSubOffset(OffsetType, E);

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitLogicalBinOp(const BinaryOperator *E) {
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
bool ByteCodeExprGen<Emitter>::VisitComplexBinOp(const BinaryOperator *E) {
  assert(Initializing);

  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  PrimType LHSElemT = *this->classifyComplexElementType(LHS->getType());
  PrimType RHSElemT = *this->classifyComplexElementType(RHS->getType());

  unsigned LHSOffset = this->allocateLocalPrimitive(LHS, PT_Ptr, true, false);
  unsigned RHSOffset = this->allocateLocalPrimitive(RHS, PT_Ptr, true, false);
  unsigned ResultOffset = ~0u;
  if (!this->DiscardResult)
    ResultOffset = this->allocateLocalPrimitive(E, PT_Ptr, true, false);

  assert(LHSElemT == RHSElemT);

  // Save result pointer in ResultOffset
  if (!this->DiscardResult) {
    if (!this->emitDupPtr(E))
      return false;
    if (!this->emitSetLocal(PT_Ptr, ResultOffset, E))
      return false;
  }

  // Evaluate LHS and save value to LHSOffset.
  if (!this->visit(LHS))
    return false;
  if (!this->emitSetLocal(PT_Ptr, LHSOffset, E))
    return false;

  // Same with RHS.
  if (!this->visit(RHS))
    return false;
  if (!this->emitSetLocal(PT_Ptr, RHSOffset, E))
    return false;

  // Now we can get pointers to the LHS and RHS from the offsets above.
  BinaryOperatorKind Op = E->getOpcode();
  for (unsigned ElemIndex = 0; ElemIndex != 2; ++ElemIndex) {
    // Result pointer for the store later.
    if (!this->DiscardResult) {
      if (!this->emitGetLocal(PT_Ptr, ResultOffset, E))
        return false;
    }

    if (!this->emitGetLocal(PT_Ptr, LHSOffset, E))
      return false;
    if (!this->emitConstUint8(ElemIndex, E))
      return false;
    if (!this->emitArrayElemPtrPopUint8(E))
      return false;
    if (!this->emitLoadPop(LHSElemT, E))
      return false;

    if (!this->emitGetLocal(PT_Ptr, RHSOffset, E))
      return false;
    if (!this->emitConstUint8(ElemIndex, E))
      return false;
    if (!this->emitArrayElemPtrPopUint8(E))
      return false;
    if (!this->emitLoadPop(RHSElemT, E))
      return false;

    // The actual operation.
    switch (Op) {
    case BO_Add:
      if (LHSElemT == PT_Float) {
        if (!this->emitAddf(getRoundingMode(E), E))
          return false;
      } else {
        if (!this->emitAdd(LHSElemT, E))
          return false;
      }
      break;
    case BO_Sub:
      if (LHSElemT == PT_Float) {
        if (!this->emitSubf(getRoundingMode(E), E))
          return false;
      } else {
        if (!this->emitSub(LHSElemT, E))
          return false;
      }
      break;

    default:
      return false;
    }

    if (!this->DiscardResult) {
      // Initialize array element with the value we just computed.
      if (!this->emitInitElemPop(LHSElemT, ElemIndex, E))
        return false;
    } else {
      if (!this->emitPop(LHSElemT, E))
        return false;
    }
  }
  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
  QualType QT = E->getType();

  if (std::optional<PrimType> T = classify(QT))
    return this->visitZeroInitializer(*T, QT, E);

  if (QT->isRecordType())
    return false;

  if (QT->isIncompleteArrayType())
    return true;

  if (QT->isArrayType()) {
    const ArrayType *AT = QT->getAsArrayTypeUnsafe();
    assert(AT);
    const auto *CAT = cast<ConstantArrayType>(AT);
    size_t NumElems = CAT->getSize().getZExtValue();
    PrimType ElemT = classifyPrim(CAT->getElementType());

    for (size_t I = 0; I != NumElems; ++I) {
      if (!this->visitZeroInitializer(ElemT, CAT->getElementType(), E))
        return false;
      if (!this->emitInitElem(ElemT, I, E))
        return false;
    }

    return true;
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitArraySubscriptExpr(
    const ArraySubscriptExpr *E) {
  const Expr *Base = E->getBase();
  const Expr *Index = E->getIdx();

  if (DiscardResult)
    return this->discard(Base) && this->discard(Index);

  // Take pointer of LHS, add offset from RHS.
  // What's left on the stack after this is a pointer.
  if (!this->visit(Base))
    return false;

  if (!this->visit(Index))
    return false;

  PrimType IndexT = classifyPrim(Index->getType());
  return this->emitArrayElemPtrPop(IndexT, E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitInitList(ArrayRef<const Expr *> Inits,
                                             const Expr *E) {
  assert(E->getType()->isRecordType());
  const Record *R = getRecord(E->getType());

  unsigned InitIndex = 0;
  for (const Expr *Init : Inits) {
    if (!this->emitDupPtr(E))
      return false;

    if (std::optional<PrimType> T = classify(Init)) {
      const Record::Field *FieldToInit = R->getField(InitIndex);
      if (!this->visit(Init))
        return false;

      if (FieldToInit->isBitField()) {
        if (!this->emitInitBitField(*T, FieldToInit, E))
          return false;
      } else {
        if (!this->emitInitField(*T, FieldToInit->Offset, E))
          return false;
      }

      if (!this->emitPopPtr(E))
        return false;
      ++InitIndex;
    } else {
      // Initializer for a direct base class.
      if (const Record::Base *B = R->getBase(Init->getType())) {
        if (!this->emitGetPtrBasePop(B->Offset, Init))
          return false;

        if (!this->visitInitializer(Init))
          return false;

        if (!this->emitInitPtrPop(E))
          return false;
        // Base initializers don't increase InitIndex, since they don't count
        // into the Record's fields.
      } else {
        const Record::Field *FieldToInit = R->getField(InitIndex);
        // Non-primitive case. Get a pointer to the field-to-initialize
        // on the stack and recurse into visitInitializer().
        if (!this->emitGetPtrField(FieldToInit->Offset, Init))
          return false;

        if (!this->visitInitializer(Init))
          return false;

        if (!this->emitPopPtr(E))
          return false;
        ++InitIndex;
      }
    }
  }
  return true;
}

/// Pointer to the array(not the element!) must be on the stack when calling
/// this.
template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitArrayElemInit(unsigned ElemIndex,
                                                  const Expr *Init) {
  if (std::optional<PrimType> T = classify(Init->getType())) {
    // Visit the primitive element like normal.
    if (!this->visit(Init))
      return false;
    return this->emitInitElem(*T, ElemIndex, Init);
  }

  // Advance the pointer currently on the stack to the given
  // dimension.
  if (!this->emitConstUint32(ElemIndex, Init))
    return false;
  if (!this->emitArrayElemPtrUint32(Init))
    return false;
  if (!this->visitInitializer(Init))
    return false;
  return this->emitPopPtr(Init);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitInitListExpr(const InitListExpr *E) {
  // Handle discarding first.
  if (DiscardResult) {
    for (const Expr *Init : E->inits()) {
      if (!this->discard(Init))
        return false;
    }
    return true;
  }

  // Primitive values.
  if (std::optional<PrimType> T = classify(E->getType())) {
    assert(!DiscardResult);
    if (E->getNumInits() == 0)
      return this->visitZeroInitializer(*T, E->getType(), E);
    assert(E->getNumInits() == 1);
    return this->delegate(E->inits()[0]);
  }

  QualType T = E->getType();
  if (T->isRecordType())
    return this->visitInitList(E->inits(), E);

  if (T->isArrayType()) {
    // FIXME: Array fillers.
    unsigned ElementIndex = 0;
    for (const Expr *Init : E->inits()) {
      if (!this->visitArrayElemInit(ElementIndex, Init))
        return false;
      ++ElementIndex;
    }
    return true;
  }

  if (T->isAnyComplexType()) {
    unsigned NumInits = E->getNumInits();

    if (NumInits == 1)
      return this->delegate(E->inits()[0]);

    QualType ElemQT = E->getType()->getAs<ComplexType>()->getElementType();
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
      for (const Expr *Init : E->inits()) {
        if (!this->visit(Init))
          return false;

        if (!this->emitInitElem(ElemT, InitIndex, E))
          return false;
        ++InitIndex;
      }
    }
    return true;
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXParenListInitExpr(
    const CXXParenListInitExpr *E) {
  if (DiscardResult) {
    for (const Expr *Init : E->getInitExprs()) {
      if (!this->discard(Init))
        return false;
    }
    return true;
  }

  assert(E->getType()->isRecordType());
  return this->visitInitList(E->getInitExprs(), E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitSubstNonTypeTemplateParmExpr(
    const SubstNonTypeTemplateParmExpr *E) {
  return this->delegate(E->getReplacement());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitConstantExpr(const ConstantExpr *E) {
  // Try to emit the APValue directly, without visiting the subexpr.
  // This will only fail if we can't emit the APValue, so won't emit any
  // diagnostics or any double values.
  std::optional<PrimType> T = classify(E->getType());
  if (T && E->hasAPValueResult() &&
      this->visitAPValue(E->getAPValueResult(), *T, E))
    return true;

  return this->delegate(E->getSubExpr());
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

  // __alignof is defined to return the preferred alignment.
  // Before 8, clang returned the preferred alignment for alignof and
  // _Alignof as well.
  if (Kind == UETT_PreferredAlignOf || AlignOfReturnsPreferred)
    return ASTCtx.toCharUnitsFromBits(ASTCtx.getPreferredTypeAlign(T));

  return ASTCtx.getTypeAlignInChars(T);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *E) {
  UnaryExprOrTypeTrait Kind = E->getKind();
  ASTContext &ASTCtx = Ctx.getASTContext();

  if (Kind == UETT_SizeOf) {
    QualType ArgType = E->getTypeOfArgument();
    CharUnits Size;
    if (ArgType->isVoidType() || ArgType->isFunctionType())
      Size = CharUnits::One();
    else {
      if (ArgType->isDependentType() || !ArgType->isConstantSizeType())
        return false;

      Size = ASTCtx.getTypeSizeInChars(ArgType);
    }

    if (DiscardResult)
      return true;

    return this->emitConst(Size.getQuantity(), E);
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

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitMemberExpr(const MemberExpr *E) {
  // 'Base.Member'
  const Expr *Base = E->getBase();

  if (DiscardResult)
    return this->discard(Base);

  if (!this->visit(Base))
    return false;

  // Base above gives us a pointer on the stack.
  // TODO: Implement non-FieldDecl members.
  const ValueDecl *Member = E->getMemberDecl();
  if (const auto *FD = dyn_cast<FieldDecl>(Member)) {
    const RecordDecl *RD = FD->getParent();
    const Record *R = getRecord(RD);
    const Record::Field *F = R->getField(FD);
    // Leave a pointer to the field on the stack.
    if (F->Decl->getType()->isReferenceType())
      return this->emitGetFieldPop(PT_Ptr, F->Offset, E);
    return this->emitGetPtrField(F->Offset, E);
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitArrayInitIndexExpr(
    const ArrayInitIndexExpr *E) {
  // ArrayIndex might not be set if a ArrayInitIndexExpr is being evaluated
  // stand-alone, e.g. via EvaluateAsInt().
  if (!ArrayIndex)
    return false;
  return this->emitConst(*ArrayIndex, E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitArrayInitLoopExpr(
    const ArrayInitLoopExpr *E) {
  assert(Initializing);
  assert(!DiscardResult);
  // TODO: This compiles to quite a lot of bytecode if the array is larger.
  //   Investigate compiling this to a loop.

  const Expr *SubExpr = E->getSubExpr();
  const Expr *CommonExpr = E->getCommonExpr();
  size_t Size = E->getArraySize().getZExtValue();

  // If the common expression is an opaque expression, we visit it
  // here once so we have its value cached.
  // FIXME: This might be necessary (or useful) for all expressions.
  if (isa<OpaqueValueExpr>(CommonExpr)) {
    if (!this->discard(CommonExpr))
      return false;
  }

  // So, every iteration, we execute an assignment here
  // where the LHS is on the stack (the target array)
  // and the RHS is our SubExpr.
  for (size_t I = 0; I != Size; ++I) {
    ArrayIndexScope<Emitter> IndexScope(this, I);
    BlockScope<Emitter> BS(this);

    if (!this->visitArrayElemInit(I, SubExpr))
      return false;
  }
  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitOpaqueValueExpr(const OpaqueValueExpr *E) {
  if (Initializing)
    return this->visitInitializer(E->getSourceExpr());

  PrimType SubExprT = classify(E->getSourceExpr()).value_or(PT_Ptr);
  if (auto It = OpaqueExprs.find(E); It != OpaqueExprs.end())
    return this->emitGetLocal(SubExprT, It->second, E);

  if (!this->visit(E->getSourceExpr()))
    return false;

  // At this point we either have the evaluated source expression or a pointer
  // to an object on the stack. We want to create a local variable that stores
  // this value.
  std::optional<unsigned> LocalIndex =
      allocateLocalPrimitive(E, SubExprT, /*IsConst=*/true);
  if (!LocalIndex)
    return false;
  if (!this->emitSetLocal(SubExprT, *LocalIndex, E))
    return false;

  // Here the local variable is created but the value is removed from the stack,
  // so we put it back, because the caller might need it.
  if (!DiscardResult) {
    if (!this->emitGetLocal(SubExprT, *LocalIndex, E))
      return false;
  }

  // FIXME: Ideally the cached value should be cleaned up later.
  OpaqueExprs.insert({E, *LocalIndex});

  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *E) {
  const Expr *Condition = E->getCond();
  const Expr *TrueExpr = E->getTrueExpr();
  const Expr *FalseExpr = E->getFalseExpr();

  LabelTy LabelEnd = this->getLabel();   // Label after the operator.
  LabelTy LabelFalse = this->getLabel(); // Label for the false expr.

  if (!this->visitBool(Condition))
    return false;

  if (!this->jumpFalse(LabelFalse))
    return false;

  if (!this->delegate(TrueExpr))
    return false;
  if (!this->jump(LabelEnd))
    return false;

  this->emitLabel(LabelFalse);

  if (!this->delegate(FalseExpr))
    return false;

  this->fallthrough(LabelEnd);
  this->emitLabel(LabelEnd);

  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitStringLiteral(const StringLiteral *E) {
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
  unsigned ArraySize = CAT->getSize().getZExtValue();
  unsigned N = std::min(ArraySize, E->getLength());
  size_t CharWidth = E->getCharByteWidth();

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
bool ByteCodeExprGen<Emitter>::VisitCharacterLiteral(
    const CharacterLiteral *E) {
  if (DiscardResult)
    return true;
  return this->emitConst(E->getValue(), E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitFloatCompoundAssignOperator(
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

  llvm::RoundingMode RM = getRoundingMode(E);
  switch (E->getOpcode()) {
  case BO_AddAssign:
    if (!this->emitAddf(RM, E))
      return false;
    break;
  case BO_SubAssign:
    if (!this->emitSubf(RM, E))
      return false;
    break;
  case BO_MulAssign:
    if (!this->emitMulf(RM, E))
      return false;
    break;
  case BO_DivAssign:
    if (!this->emitDivf(RM, E))
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
bool ByteCodeExprGen<Emitter>::VisitPointerCompoundAssignOperator(
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
  assert(*LT == PT_Ptr);

  if (!visit(LHS))
    return false;

  if (!this->emitLoadPtr(LHS))
    return false;

  if (!visit(RHS))
    return false;

  if (Op == BO_AddAssign)
    this->emitAddOffset(*RT, E);
  else
    this->emitSubOffset(*RT, E);

  if (DiscardResult)
    return this->emitStorePopPtr(E);
  return this->emitStorePtr(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCompoundAssignOperator(
    const CompoundAssignOperator *E) {

  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  std::optional<PrimType> LHSComputationT =
      classify(E->getComputationLHSType());
  std::optional<PrimType> LT = classify(LHS->getType());
  std::optional<PrimType> RT = classify(E->getComputationResultType());
  std::optional<PrimType> ResultT = classify(E->getType());

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
  if (*LT != *LHSComputationT) {
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
  if (*ResultT != *LHSComputationT) {
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
bool ByteCodeExprGen<Emitter>::VisitExprWithCleanups(
    const ExprWithCleanups *E) {
  const Expr *SubExpr = E->getSubExpr();

  assert(E->getNumObjects() == 0 && "TODO: Implement cleanups");

  return this->delegate(SubExpr);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *E) {
  const Expr *SubExpr = E->getSubExpr();

  if (Initializing) {
    // We already have a value, just initialize that.
    return this->visitInitializer(SubExpr);
  }
  // If we don't end up using the materialized temporary anyway, don't
  // bother creating it.
  if (DiscardResult)
    return this->discard(SubExpr);

  // When we're initializing a global variable *or* the storage duration of
  // the temporary is explicitly static, create a global variable.
  std::optional<PrimType> SubExprT = classify(SubExpr);
  bool IsStatic = E->getStorageDuration() == SD_Static;
  if (GlobalDecl || IsStatic) {
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
    if (std::optional<unsigned> LocalIndex = allocateLocalPrimitive(
            SubExpr, *SubExprT, /*IsConst=*/true, /*IsExtended=*/true)) {
      if (!this->visit(SubExpr))
        return false;
      this->emitSetLocal(*SubExprT, *LocalIndex, E);
      return this->emitGetPtrLocal(*LocalIndex, E);
    }
  } else {
    if (std::optional<unsigned> LocalIndex =
            allocateLocal(SubExpr, /*IsExtended=*/true)) {
      if (!this->emitGetPtrLocal(*LocalIndex, E))
        return false;
      return this->visitInitializer(SubExpr);
    }
  }
  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXBindTemporaryExpr(
    const CXXBindTemporaryExpr *E) {
  return this->delegate(E->getSubExpr());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCompoundLiteralExpr(
    const CompoundLiteralExpr *E) {
  const Expr *Init = E->getInitializer();
  if (Initializing) {
    // We already have a value, just initialize that.
    return this->visitInitializer(Init);
  }

  std::optional<PrimType> T = classify(E->getType());
  if (E->isFileScope()) {
    if (std::optional<unsigned> GlobalIndex = P.createGlobal(E)) {
      if (classify(E->getType()))
        return this->visit(Init);
      if (!this->emitGetPtrGlobal(*GlobalIndex, E))
        return false;
      return this->visitInitializer(Init);
    }
  }

  // Otherwise, use a local variable.
  if (T) {
    // For primitive types, we just visit the initializer.
    return this->delegate(Init);
  } else {
    if (std::optional<unsigned> LocalIndex = allocateLocal(Init)) {
      if (!this->emitGetPtrLocal(*LocalIndex, E))
        return false;
      if (!this->visitInitializer(Init))
        return false;
      if (DiscardResult)
        return this->emitPopPtr(E);
      return true;
    }
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitTypeTraitExpr(const TypeTraitExpr *E) {
  if (DiscardResult)
    return true;
  return this->emitConstBool(E->getValue(), E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitLambdaExpr(const LambdaExpr *E) {
  assert(Initializing);
  const Record *R = P.getOrCreateRecord(E->getLambdaClass());

  auto *CaptureInitIt = E->capture_init_begin();
  // Initialize all fields (which represent lambda captures) of the
  // record with their initializers.
  for (const Record::Field &F : R->fields()) {
    const Expr *Init = *CaptureInitIt;
    ++CaptureInitIt;

    if (std::optional<PrimType> T = classify(Init)) {
      if (!this->visit(Init))
        return false;

      if (!this->emitSetField(*T, F.Offset, E))
        return false;
    } else {
      if (!this->emitDupPtr(E))
        return false;

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
bool ByteCodeExprGen<Emitter>::VisitPredefinedExpr(const PredefinedExpr *E) {
  if (DiscardResult)
    return true;

  assert(!Initializing);
  return this->visit(E->getFunctionName());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXThrowExpr(const CXXThrowExpr *E) {
  if (E->getSubExpr() && !this->discard(E->getSubExpr()))
    return false;

  return this->emitInvalid(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXReinterpretCastExpr(
    const CXXReinterpretCastExpr *E) {
  if (!this->discard(E->getSubExpr()))
    return false;

  return this->emitInvalidCast(CastKind::Reinterpret, E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXNoexceptExpr(const CXXNoexceptExpr *E) {
  assert(E->getType()->isBooleanType());

  if (DiscardResult)
    return true;
  return this->emitConstBool(E->getValue(), E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXConstructExpr(
    const CXXConstructExpr *E) {
  QualType T = E->getType();
  assert(!classify(T));

  if (T->isRecordType()) {
    const CXXConstructorDecl *Ctor = E->getConstructor();

    // Trivial zero initialization.
    if (E->requiresZeroInitialization() && Ctor->isTrivial()) {
      const Record *R = getRecord(E->getType());
      return this->visitZeroRecordInitializer(R, E);
    }

    const Function *Func = getFunction(Ctor);

    if (!Func)
      return false;

    assert(Func->hasThisPointer());
    assert(!Func->hasRVO());

    // If we're discarding a construct expression, we still need
    // to allocate a variable and call the constructor and destructor.
    if (DiscardResult) {
      assert(!Initializing);
      std::optional<unsigned> LocalIndex =
          allocateLocal(E, /*IsExtended=*/true);

      if (!LocalIndex)
        return false;

      if (!this->emitGetPtrLocal(*LocalIndex, E))
        return false;
    }

    //  The This pointer is already on the stack because this is an initializer,
    //  but we need to dup() so the call() below has its own copy.
    if (!this->emitDupPtr(E))
      return false;

    // Constructor arguments.
    for (const auto *Arg : E->arguments()) {
      if (!this->visit(Arg))
        return false;
    }

    if (!this->emitCall(Func, E))
      return false;

    // Immediately call the destructor if we have to.
    if (DiscardResult) {
      if (!this->emitPopPtr(E))
        return false;
    }
    return true;
  }

  if (T->isArrayType()) {
    const ConstantArrayType *CAT =
        Ctx.getASTContext().getAsConstantArrayType(E->getType());
    assert(CAT);
    size_t NumElems = CAT->getSize().getZExtValue();
    const Function *Func = getFunction(E->getConstructor());
    if (!Func || !Func->isConstexpr())
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

      if (!this->emitCall(Func, E))
        return false;
    }
    return true;
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitSourceLocExpr(const SourceLocExpr *E) {
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
bool ByteCodeExprGen<Emitter>::VisitOffsetOfExpr(const OffsetOfExpr *E) {
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
bool ByteCodeExprGen<Emitter>::VisitCXXScalarValueInitExpr(
    const CXXScalarValueInitExpr *E) {
  QualType Ty = E->getType();

  if (Ty->isVoidType())
    return true;

  return this->visitZeroInitializer(classifyPrim(Ty), Ty, E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitSizeOfPackExpr(const SizeOfPackExpr *E) {
  return this->emitConst(E->getPackLength(), E);
}

template <class Emitter> bool ByteCodeExprGen<Emitter>::discard(const Expr *E) {
  if (E->containsErrors())
    return false;

  OptionScope<Emitter> Scope(this, /*NewDiscardResult=*/true,
                             /*NewInitializing=*/false);
  return this->Visit(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::delegate(const Expr *E) {
  if (E->containsErrors())
    return false;

  // We're basically doing:
  // OptionScope<Emitter> Scope(this, DicardResult, Initializing);
  // but that's unnecessary of course.
  return this->Visit(E);
}

template <class Emitter> bool ByteCodeExprGen<Emitter>::visit(const Expr *E) {
  if (E->containsErrors())
    return false;

  if (E->getType()->isVoidType())
    return this->discard(E);

  // Create local variable to hold the return value.
  if (!E->isGLValue() && !E->getType()->isAnyComplexType() &&
      !classify(E->getType())) {
    std::optional<unsigned> LocalIndex = allocateLocal(E, /*IsExtended=*/true);
    if (!LocalIndex)
      return false;

    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
    return this->visitInitializer(E);
  }

  //  Otherwise,we have a primitive return value, produce the value directly
  //  and push it on the stack.
  OptionScope<Emitter> Scope(this, /*NewDiscardResult=*/false,
                             /*NewInitializing=*/false);
  return this->Visit(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitInitializer(const Expr *E) {
  assert(!classify(E->getType()));

  if (E->containsErrors())
    return false;

  OptionScope<Emitter> Scope(this, /*NewDiscardResult=*/false,
                             /*NewInitializing=*/true);
  return this->Visit(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitBool(const Expr *E) {
  std::optional<PrimType> T = classify(E->getType());
  if (!T)
    return false;

  if (!this->visit(E))
    return false;

  if (T == PT_Bool)
    return true;

  // Convert pointers to bool.
  if (T == PT_Ptr || T == PT_FnPtr) {
    if (!this->emitNull(*T, E))
      return false;
    return this->emitNE(*T, E);
  }

  // Or Floats.
  if (T == PT_Float)
    return this->emitCastFloatingIntegralBool(E);

  // Or anything else we can.
  return this->emitCast(*T, PT_Bool, E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitZeroInitializer(PrimType T, QualType QT,
                                                    const Expr *E) {
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
    return this->emitNullPtr(E);
  case PT_FnPtr:
    return this->emitNullFnPtr(E);
  case PT_Float: {
    return this->emitConstFloat(APFloat::getZero(Ctx.getFloatSemantics(QT)), E);
  }
  }
  llvm_unreachable("unknown primitive type");
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitZeroRecordInitializer(const Record *R,
                                                          const Expr *E) {
  assert(E);
  assert(R);
  // Fields
  for (const Record::Field &Field : R->fields()) {
    const Descriptor *D = Field.Desc;
    if (D->isPrimitive()) {
      QualType QT = D->getType();
      PrimType T = classifyPrim(D->getType());
      if (!this->visitZeroInitializer(T, QT, E))
        return false;
      if (!this->emitInitField(T, Field.Offset, E))
        return false;
      continue;
    }

    // TODO: Add GetPtrFieldPop and get rid of this dup.
    if (!this->emitDupPtr(E))
      return false;
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
      const Record *ElemRecord = D->ElemDesc->ElemRecord;
      assert(D->ElemDesc->ElemRecord);
      for (uint32_t I = 0, N = D->getNumElems(); I != N; ++I) {
        if (!this->emitConstUint32(I, E))
          return false;
        if (!this->emitArrayElemPtr(PT_Uint32, E))
          return false;
        if (!this->visitZeroRecordInitializer(ElemRecord, E))
          return false;
        if (!this->emitPopPtr(E))
          return false;
      }
    } else if (D->isRecord()) {
      if (!this->visitZeroRecordInitializer(D->ElemRecord, E))
        return false;
    } else {
      assert(false);
    }

    if (!this->emitPopPtr(E))
      return false;
  }

  for (const Record::Base &B : R->bases()) {
    if (!this->emitGetPtrBase(B.Offset, E))
      return false;
    if (!this->visitZeroRecordInitializer(B.R, E))
      return false;
    if (!this->emitInitPtrPop(E))
      return false;
  }

  // FIXME: Virtual bases.

  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::dereference(
    const Expr *LV, DerefKind AK, llvm::function_ref<bool(PrimType)> Direct,
    llvm::function_ref<bool(PrimType)> Indirect) {
  if (std::optional<PrimType> T = classify(LV->getType())) {
    if (!LV->refersToBitField()) {
      // Only primitive, non bit-field types can be dereferenced directly.
      if (const auto *DE = dyn_cast<DeclRefExpr>(LV)) {
        if (!DE->getDecl()->getType()->isReferenceType()) {
          if (const auto *PD = dyn_cast<ParmVarDecl>(DE->getDecl()))
            return dereferenceParam(LV, *T, PD, AK, Direct, Indirect);
          if (const auto *VD = dyn_cast<VarDecl>(DE->getDecl()))
            return dereferenceVar(LV, *T, VD, AK, Direct, Indirect);
        }
      }
    }

    if (!visit(LV))
      return false;
    return Indirect(*T);
  }

  if (LV->getType()->isAnyComplexType())
    return this->delegate(LV);

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::dereferenceParam(
    const Expr *LV, PrimType T, const ParmVarDecl *PD, DerefKind AK,
    llvm::function_ref<bool(PrimType)> Direct,
    llvm::function_ref<bool(PrimType)> Indirect) {
  auto It = this->Params.find(PD);
  if (It != this->Params.end()) {
    unsigned Idx = It->second.Offset;
    switch (AK) {
    case DerefKind::Read:
      return DiscardResult ? true : this->emitGetParam(T, Idx, LV);

    case DerefKind::Write:
      if (!Direct(T))
        return false;
      if (!this->emitSetParam(T, Idx, LV))
        return false;
      return DiscardResult ? true : this->emitGetPtrParam(Idx, LV);

    case DerefKind::ReadWrite:
      if (!this->emitGetParam(T, Idx, LV))
        return false;
      if (!Direct(T))
        return false;
      if (!this->emitSetParam(T, Idx, LV))
        return false;
      return DiscardResult ? true : this->emitGetPtrParam(Idx, LV);
    }
    return true;
  }

  // If the param is a pointer, we can dereference a dummy value.
  if (!DiscardResult && T == PT_Ptr && AK == DerefKind::Read) {
    if (auto Idx = P.getOrCreateDummy(PD))
      return this->emitGetPtrGlobal(*Idx, PD);
    return false;
  }

  // Value cannot be produced - try to emit pointer and do stuff with it.
  return visit(LV) && Indirect(T);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::dereferenceVar(
    const Expr *LV, PrimType T, const VarDecl *VD, DerefKind AK,
    llvm::function_ref<bool(PrimType)> Direct,
    llvm::function_ref<bool(PrimType)> Indirect) {
  auto It = Locals.find(VD);
  if (It != Locals.end()) {
    const auto &L = It->second;
    switch (AK) {
    case DerefKind::Read:
      if (!this->emitGetLocal(T, L.Offset, LV))
        return false;
      return DiscardResult ? this->emitPop(T, LV) : true;

    case DerefKind::Write:
      if (!Direct(T))
        return false;
      if (!this->emitSetLocal(T, L.Offset, LV))
        return false;
      return DiscardResult ? true : this->emitGetPtrLocal(L.Offset, LV);

    case DerefKind::ReadWrite:
      if (!this->emitGetLocal(T, L.Offset, LV))
        return false;
      if (!Direct(T))
        return false;
      if (!this->emitSetLocal(T, L.Offset, LV))
        return false;
      return DiscardResult ? true : this->emitGetPtrLocal(L.Offset, LV);
    }
  } else if (auto Idx = P.getGlobal(VD)) {
    switch (AK) {
    case DerefKind::Read:
      if (!this->emitGetGlobal(T, *Idx, LV))
        return false;
      return DiscardResult ? this->emitPop(T, LV) : true;

    case DerefKind::Write:
      if (!Direct(T))
        return false;
      if (!this->emitSetGlobal(T, *Idx, LV))
        return false;
      return DiscardResult ? true : this->emitGetPtrGlobal(*Idx, LV);

    case DerefKind::ReadWrite:
      if (!this->emitGetGlobal(T, *Idx, LV))
        return false;
      if (!Direct(T))
        return false;
      if (!this->emitSetGlobal(T, *Idx, LV))
        return false;
      return DiscardResult ? true : this->emitGetPtrGlobal(*Idx, LV);
    }
  }

  // If the declaration is a constant value, emit it here even
  // though the declaration was not evaluated in the current scope.
  // The access mode can only be read in this case.
  if (!DiscardResult && AK == DerefKind::Read) {
    if (VD->hasLocalStorage() && VD->hasInit() && !VD->isConstexpr()) {
      QualType VT = VD->getType();
      if (VT.isConstQualified() && VT->isFundamentalType())
        return this->visit(VD->getInit());
    }
  }

  // Value cannot be produced - try to emit pointer.
  return visit(LV) && Indirect(T);
}

template <class Emitter>
template <typename T>
bool ByteCodeExprGen<Emitter>::emitConst(T Value, PrimType Ty, const Expr *E) {
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
  case PT_IntAP:
  case PT_IntAPS:
    assert(false);
    return false;
  case PT_Bool:
    return this->emitConstBool(Value, E);
  case PT_Ptr:
  case PT_FnPtr:
  case PT_Float:
    llvm_unreachable("Invalid integral type");
    break;
  }
  llvm_unreachable("unknown primitive type");
}

template <class Emitter>
template <typename T>
bool ByteCodeExprGen<Emitter>::emitConst(T Value, const Expr *E) {
  return this->emitConst(Value, classifyPrim(E->getType()), E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::emitConst(const APSInt &Value, PrimType Ty,
                                         const Expr *E) {
  if (Value.isSigned())
    return this->emitConst(Value.getSExtValue(), Ty, E);
  return this->emitConst(Value.getZExtValue(), Ty, E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::emitConst(const APSInt &Value, const Expr *E) {
  return this->emitConst(Value, classifyPrim(E->getType()), E);
}

template <class Emitter>
unsigned ByteCodeExprGen<Emitter>::allocateLocalPrimitive(DeclTy &&Src,
                                                          PrimType Ty,
                                                          bool IsConst,
                                                          bool IsExtended) {
  // Make sure we don't accidentally register the same decl twice.
  if (const auto *VD =
          dyn_cast_if_present<ValueDecl>(Src.dyn_cast<const Decl *>())) {
    assert(!P.getGlobal(VD));
    assert(!Locals.contains(VD));
  }

  // FIXME: There are cases where Src.is<Expr*>() is wrong, e.g.
  //   (int){12} in C. Consider using Expr::isTemporaryObject() instead
  //   or isa<MaterializeTemporaryExpr>().
  Descriptor *D = P.createDescriptor(Src, Ty, Descriptor::InlineDescMD, IsConst,
                                     Src.is<const Expr *>());
  Scope::Local Local = this->createLocal(D);
  if (auto *VD = dyn_cast_if_present<ValueDecl>(Src.dyn_cast<const Decl *>()))
    Locals.insert({VD, Local});
  VarScope->add(Local, IsExtended);
  return Local.Offset;
}

template <class Emitter>
std::optional<unsigned>
ByteCodeExprGen<Emitter>::allocateLocal(DeclTy &&Src, bool IsExtended) {
  // Make sure we don't accidentally register the same decl twice.
  if ([[maybe_unused]]  const auto *VD =
          dyn_cast_if_present<ValueDecl>(Src.dyn_cast<const Decl *>())) {
    assert(!P.getGlobal(VD));
    assert(!Locals.contains(VD));
  }

  QualType Ty;
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
    Ty = E->getType();
  }

  Descriptor *D = P.createDescriptor(
      Src, Ty.getTypePtr(), Descriptor::InlineDescMD, Ty.isConstQualified(),
      IsTemporary, /*IsMutable=*/false, Init);
  if (!D)
    return {};

  Scope::Local Local = this->createLocal(D);
  if (Key)
    Locals.insert({Key, Local});
  VarScope->add(Local, IsExtended);
  return Local.Offset;
}

template <class Emitter>
const RecordType *ByteCodeExprGen<Emitter>::getRecordTy(QualType Ty) {
  if (const PointerType *PT = dyn_cast<PointerType>(Ty))
    return PT->getPointeeType()->getAs<RecordType>();
  return Ty->getAs<RecordType>();
}

template <class Emitter>
Record *ByteCodeExprGen<Emitter>::getRecord(QualType Ty) {
  if (const auto *RecordTy = getRecordTy(Ty))
    return getRecord(RecordTy->getDecl());
  return nullptr;
}

template <class Emitter>
Record *ByteCodeExprGen<Emitter>::getRecord(const RecordDecl *RD) {
  return P.getOrCreateRecord(RD);
}

template <class Emitter>
const Function *ByteCodeExprGen<Emitter>::getFunction(const FunctionDecl *FD) {
  return Ctx.getOrCreateFunction(FD);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitExpr(const Expr *E) {
  ExprScope<Emitter> RootScope(this);
  // Void expressions.
  if (E->getType()->isVoidType()) {
    if (!visit(E))
      return false;
    return this->emitRetVoid(E);
  }

  // Expressions with a primitive return type.
  if (std::optional<PrimType> T = classify(E)) {
    if (!visit(E))
      return false;
    return this->emitRet(*T, E);
  }

  // Expressions with a composite return type.
  // For us, that means everything we don't
  // have a PrimType for.
  if (std::optional<unsigned> LocalOffset = this->allocateLocal(E)) {
    if (!this->visitLocalInitializer(E, *LocalOffset))
      return false;

    if (!this->emitGetPtrLocal(*LocalOffset, E))
      return false;
    return this->emitRetValue(E);
  }

  return false;
}

/// Toplevel visitDecl().
/// We get here from evaluateAsInitializer().
/// We need to evaluate the initializer and return its value.
template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitDecl(const VarDecl *VD) {
  assert(!VD->isInvalidDecl() && "Trying to constant evaluate an invalid decl");

  // Create and initialize the variable.
  if (!this->visitVarDecl(VD))
    return false;

  std::optional<PrimType> VarT = classify(VD->getType());
  // Get a pointer to the variable
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

  // Return the value
  if (VarT)
    return this->emitRet(*VarT, VD);

  // Return non-primitive values as pointers here.
  return this->emitRet(PT_Ptr, VD);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitVarDecl(const VarDecl *VD) {
  // We don't know what to do with these, so just return false.
  if (VD->getType().isNull())
    return false;

  const Expr *Init = VD->getInit();
  std::optional<PrimType> VarT = classify(VD->getType());

  if (Context::shouldBeGloballyIndexed(VD)) {
    // We've already seen and initialized this global.
    if (P.getGlobal(VD))
      return true;

    std::optional<unsigned> GlobalIndex = P.createGlobal(VD, Init);

    if (!GlobalIndex)
      return false;

    assert(Init);
    {
      DeclScope<Emitter> LocalScope(this, VD);

      if (VarT) {
        if (!this->visit(Init))
          return false;
        return this->emitInitGlobal(*VarT, *GlobalIndex, VD);
      }
      return this->visitGlobalInitializer(Init, *GlobalIndex);
    }
  } else {
    VariableScope<Emitter> LocalScope(this);
    if (VarT) {
      unsigned Offset = this->allocateLocalPrimitive(
          VD, *VarT, VD->getType().isConstQualified());
      if (Init) {
        // Compile the initializer in its own scope.
        ExprScope<Emitter> Scope(this);
        if (!this->visit(Init))
          return false;

        return this->emitSetLocal(*VarT, Offset, VD);
      }
    } else {
      if (std::optional<unsigned> Offset = this->allocateLocal(VD)) {
        if (Init)
          return this->visitLocalInitializer(Init, *Offset);
      }
    }
    return true;
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitAPValue(const APValue &Val,
                                            PrimType ValType, const Expr *E) {
  assert(!DiscardResult);
  if (Val.isInt())
    return this->emitConst(Val.getInt(), ValType, E);

  if (Val.isLValue()) {
    APValue::LValueBase Base = Val.getLValueBase();
    if (const Expr *BaseExpr = Base.dyn_cast<const Expr *>())
      return this->visit(BaseExpr);
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitBuiltinCallExpr(const CallExpr *E) {
  const Function *Func = getFunction(E->getDirectCallee());
  if (!Func)
    return false;

  if (!Func->isUnevaluatedBuiltin()) {
    // Put arguments on the stack.
    for (const auto *Arg : E->arguments()) {
      if (!this->visit(Arg))
        return false;
    }
  }

  if (!this->emitCallBI(Func, E, E))
    return false;

  QualType ReturnType = E->getCallReturnType(Ctx.getASTContext());
  if (DiscardResult && !ReturnType->isVoidType()) {
    PrimType T = classifyPrim(ReturnType);
    return this->emitPop(T, E);
  }

  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCallExpr(const CallExpr *E) {
  if (E->getBuiltinCallee())
    return VisitBuiltinCallExpr(E);

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
      assert(Initializing);
      if (!this->emitDupPtr(E))
        return false;
    }
  }

  // Add the (optional, implicit) This pointer.
  if (const auto *MC = dyn_cast<CXXMemberCallExpr>(E)) {
    if (!this->visit(MC->getImplicitObjectArgument()))
      return false;
  }

  // Put arguments on the stack.
  for (const auto *Arg : E->arguments()) {
    if (!this->visit(Arg))
      return false;
  }

  if (const FunctionDecl *FuncDecl = E->getDirectCallee()) {
    const Function *Func = getFunction(FuncDecl);
    if (!Func)
      return false;
    // If the function is being compiled right now, this is a recursive call.
    // In that case, the function can't be valid yet, even though it will be
    // later.
    // If the function is already fully compiled but not constexpr, it was
    // found to be faulty earlier on, so bail out.
    if (Func->isFullyCompiled() && !Func->isConstexpr())
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
      if (!this->emitCallVirt(Func, E))
        return false;
    } else {
      if (!this->emitCall(Func, E))
        return false;
    }
  } else {
    // Indirect call. Visit the callee, which will leave a FunctionPointer on
    // the stack. Cleanup of the returned value if necessary will be done after
    // the function call completed.
    if (!this->visit(E->getCallee()))
      return false;

    if (!this->emitCallPtr(E))
      return false;
  }

  // Cleanup for discarded return values.
  if (DiscardResult && !ReturnType->isVoidType() && T)
    return this->emitPop(*T, E);

  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXDefaultInitExpr(
    const CXXDefaultInitExpr *E) {
  SourceLocScope<Emitter> SLS(this, E);
  if (Initializing)
    return this->visitInitializer(E->getExpr());

  assert(classify(E->getType()));
  return this->visit(E->getExpr());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXDefaultArgExpr(
    const CXXDefaultArgExpr *E) {
  SourceLocScope<Emitter> SLS(this, E);

  const Expr *SubExpr = E->getExpr();
  if (std::optional<PrimType> T = classify(E->getExpr()))
    return this->visit(SubExpr);

  assert(Initializing);
  return this->visitInitializer(SubExpr);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXBoolLiteralExpr(
    const CXXBoolLiteralExpr *E) {
  if (DiscardResult)
    return true;

  return this->emitConstBool(E->getValue(), E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXNullPtrLiteralExpr(
    const CXXNullPtrLiteralExpr *E) {
  if (DiscardResult)
    return true;

  return this->emitNullPtr(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitGNUNullExpr(const GNUNullExpr *E) {
  if (DiscardResult)
    return true;

  assert(E->getType()->isIntegerType());

  PrimType T = classifyPrim(E->getType());
  return this->emitZero(T, E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXThisExpr(const CXXThisExpr *E) {
  if (DiscardResult)
    return true;

  if (this->LambdaThisCapture > 0)
    return this->emitGetThisFieldPtr(this->LambdaThisCapture, E);

  return this->emitThis(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitUnaryOperator(const UnaryOperator *E) {
  const Expr *SubExpr = E->getSubExpr();
  std::optional<PrimType> T = classify(SubExpr->getType());

  switch (E->getOpcode()) {
  case UO_PostInc: { // x++
    if (!this->visit(SubExpr))
      return false;

    if (T == PT_Ptr) {
      if (!this->emitIncPtr(E))
        return false;

      return DiscardResult ? this->emitPopPtr(E) : true;
    }

    if (T == PT_Float) {
      return DiscardResult ? this->emitIncfPop(getRoundingMode(E), E)
                           : this->emitIncf(getRoundingMode(E), E);
    }

    return DiscardResult ? this->emitIncPop(*T, E) : this->emitInc(*T, E);
  }
  case UO_PostDec: { // x--
    if (!this->visit(SubExpr))
      return false;

    if (T == PT_Ptr) {
      if (!this->emitDecPtr(E))
        return false;

      return DiscardResult ? this->emitPopPtr(E) : true;
    }

    if (T == PT_Float) {
      return DiscardResult ? this->emitDecfPop(getRoundingMode(E), E)
                           : this->emitDecf(getRoundingMode(E), E);
    }

    return DiscardResult ? this->emitDecPop(*T, E) : this->emitDec(*T, E);
  }
  case UO_PreInc: { // ++x
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
        return this->emitIncfPop(getRoundingMode(E), E);
      return this->emitIncPop(*T, E);
    }

    if (T == PT_Float) {
      const auto &TargetSemantics = Ctx.getFloatSemantics(E->getType());
      if (!this->emitLoadFloat(E))
        return false;
      if (!this->emitConstFloat(llvm::APFloat(TargetSemantics, 1), E))
        return false;
      if (!this->emitAddf(getRoundingMode(E), E))
        return false;
      return this->emitStoreFloat(E);
    }
    if (!this->emitLoad(*T, E))
      return false;
    if (!this->emitConst(1, E))
      return false;
    if (!this->emitAdd(*T, E))
      return false;
    return this->emitStore(*T, E);
  }
  case UO_PreDec: { // --x
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
        return this->emitDecfPop(getRoundingMode(E), E);
      return this->emitDecPop(*T, E);
    }

    if (T == PT_Float) {
      const auto &TargetSemantics = Ctx.getFloatSemantics(E->getType());
      if (!this->emitLoadFloat(E))
        return false;
      if (!this->emitConstFloat(llvm::APFloat(TargetSemantics, 1), E))
        return false;
      if (!this->emitSubf(getRoundingMode(E), E))
        return false;
      return this->emitStoreFloat(E);
    }
    if (!this->emitLoad(*T, E))
      return false;
    if (!this->emitConst(1, E))
      return false;
    if (!this->emitSub(*T, E))
      return false;
    return this->emitStore(*T, E);
  }
  case UO_LNot: // !x
    if (DiscardResult)
      return this->discard(SubExpr);

    if (!this->visitBool(SubExpr))
      return false;

    if (!this->emitInvBool(E))
      return false;

    if (PrimType ET = classifyPrim(E->getType()); ET != PT_Bool)
      return this->emitCast(PT_Bool, ET, E);
    return true;
  case UO_Minus: // -x
    if (!this->visit(SubExpr))
      return false;
    return DiscardResult ? this->emitPop(*T, E) : this->emitNeg(*T, E);
  case UO_Plus:  // +x
    if (!this->visit(SubExpr)) // noop
      return false;
    return DiscardResult ? this->emitPop(*T, E) : true;
  case UO_AddrOf: // &x
    // We should already have a pointer when we get here.
    return this->delegate(SubExpr);
  case UO_Deref:  // *x
    return dereference(
        SubExpr, DerefKind::Read,
        [](PrimType) {
          llvm_unreachable("Dereferencing requires a pointer");
          return false;
        },
        [this, E](PrimType T) {
          return DiscardResult ? this->emitPop(T, E) : true;
        });
  case UO_Not:    // ~x
    if (!this->visit(SubExpr))
      return false;
    return DiscardResult ? this->emitPop(*T, E) : this->emitComp(*T, E);
  case UO_Real: // __real x
    if (T)
      return this->delegate(SubExpr);
    return this->emitComplexReal(SubExpr);
  case UO_Imag: { // __imag x
    if (T) {
      if (!this->discard(SubExpr))
        return false;
      return this->visitZeroInitializer(*T, SubExpr->getType(), SubExpr);
    }
    if (!this->visit(SubExpr))
      return false;
    if (!this->emitConstUint8(1, E))
      return false;
    if (!this->emitArrayElemPtrPopUint8(E))
      return false;

    // Since our _Complex implementation does not map to a primitive type,
    // we sometimes have to do the lvalue-to-rvalue conversion here manually.
    if (!SubExpr->isLValue())
      return this->emitLoadPop(classifyPrim(E->getType()), E);
    return true;
  }
  case UO_Extension:
    return this->delegate(SubExpr);
  case UO_Coawait:
    assert(false && "Unhandled opcode");
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitDeclRefExpr(const DeclRefExpr *E) {
  if (DiscardResult)
    return true;

  const auto *D = E->getDecl();

  if (const auto *ECD = dyn_cast<EnumConstantDecl>(D)) {
    return this->emitConst(ECD->getInitVal(), E);
  } else if (const auto *BD = dyn_cast<BindingDecl>(D)) {
    return this->visit(BD->getBinding());
  } else if (const auto *FuncDecl = dyn_cast<FunctionDecl>(D)) {
    const Function *F = getFunction(FuncDecl);
    return F && this->emitGetFnPtr(F, E);
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
      return this->emitGetLocal(PT_Ptr, Offset, E);
    return this->emitGetPtrLocal(Offset, E);
  } else if (auto GlobalIndex = P.getGlobal(D)) {
    if (IsReference)
      return this->emitGetGlobalPtr(*GlobalIndex, E);

    return this->emitGetPtrGlobal(*GlobalIndex, E);
  } else if (const auto *PVD = dyn_cast<ParmVarDecl>(D)) {
    if (auto It = this->Params.find(PVD); It != this->Params.end()) {
      if (IsReference || !It->second.IsPtr)
        return this->emitGetParamPtr(It->second.Offset, E);

      return this->emitGetPtrParam(It->second.Offset, E);
    }
  }

  // Handle lambda captures.
  if (auto It = this->LambdaCaptures.find(D);
      It != this->LambdaCaptures.end()) {
    auto [Offset, IsPtr] = It->second;

    if (IsPtr)
      return this->emitGetThisFieldPtr(Offset, E);
    return this->emitGetPtrThisField(Offset, E);
  }

  // Lazily visit global declarations we haven't seen yet.
  // This happens in C.
  if (!Ctx.getLangOpts().CPlusPlus) {
    if (const auto *VD = dyn_cast<VarDecl>(D);
        VD && VD->hasGlobalStorage() && VD->getAnyInitializer() &&
        VD->getType().isConstQualified()) {
      if (!this->visitVarDecl(VD))
        return false;
      // Retry.
      return this->VisitDeclRefExpr(E);
    }

    if (std::optional<unsigned> I = P.getOrCreateDummy(D))
      return this->emitGetPtrGlobal(*I, E);
  }

  return this->emitInvalidDeclRef(E, E);
}

template <class Emitter>
void ByteCodeExprGen<Emitter>::emitCleanup() {
  for (VariableScope<Emitter> *C = VarScope; C; C = C->getParent())
    C->emitDestruction();
}

template <class Emitter>
unsigned
ByteCodeExprGen<Emitter>::collectBaseOffset(const RecordType *BaseType,
                                            const RecordType *DerivedType) {
  const auto *FinalDecl = cast<CXXRecordDecl>(BaseType->getDecl());
  const RecordDecl *CurDecl = DerivedType->getDecl();
  const Record *CurRecord = getRecord(CurDecl);
  assert(CurDecl && FinalDecl);

  unsigned OffsetSum = 0;
  for (;;) {
    assert(CurRecord->getNumBases() > 0);
    // One level up
    for (const Record::Base &B : CurRecord->bases()) {
      const auto *BaseDecl = cast<CXXRecordDecl>(B.Decl);

      if (BaseDecl == FinalDecl || BaseDecl->isDerivedFrom(FinalDecl)) {
        OffsetSum += B.Offset;
        CurRecord = B.R;
        CurDecl = BaseDecl;
        break;
      }
    }
    if (CurDecl == FinalDecl)
      break;
  }

  assert(OffsetSum > 0);
  return OffsetSum;
}

/// Emit casts from a PrimType to another PrimType.
template <class Emitter>
bool ByteCodeExprGen<Emitter>::emitPrimCast(PrimType FromT, PrimType ToT,
                                            QualType ToQT, const Expr *E) {

  if (FromT == PT_Float) {
    // Floating to floating.
    if (ToT == PT_Float) {
      const llvm::fltSemantics *ToSem = &Ctx.getFloatSemantics(ToQT);
      return this->emitCastFP(ToSem, getRoundingMode(E), E);
    }

    // Float to integral.
    if (isIntegralType(ToT) || ToT == PT_Bool)
      return this->emitCastFloatingIntegral(ToT, E);
  }

  if (isIntegralType(FromT) || FromT == PT_Bool) {
    // Integral to integral.
    if (isIntegralType(ToT) || ToT == PT_Bool)
      return FromT != ToT ? this->emitCast(FromT, ToT, E) : true;

    if (ToT == PT_Float) {
      // Integral to floating.
      const llvm::fltSemantics *ToSem = &Ctx.getFloatSemantics(ToQT);
      return this->emitCastIntegralFloating(FromT, ToSem, getRoundingMode(E),
                                            E);
    }
  }

  return false;
}

/// Emits __real(SubExpr)
template <class Emitter>
bool ByteCodeExprGen<Emitter>::emitComplexReal(const Expr *SubExpr) {
  assert(SubExpr->getType()->isAnyComplexType());

  if (DiscardResult)
    return this->discard(SubExpr);

  if (!this->visit(SubExpr))
    return false;
  if (!this->emitConstUint8(0, SubExpr))
    return false;
  if (!this->emitArrayElemPtrPopUint8(SubExpr))
    return false;

  // Since our _Complex implementation does not map to a primitive type,
  // we sometimes have to do the lvalue-to-rvalue conversion here manually.
  if (!SubExpr->isLValue())
    return this->emitLoadPop(*classifyComplexElementType(SubExpr->getType()),
                             SubExpr);
  return true;
}

/// When calling this, we have a pointer of the local-to-destroy
/// on the stack.
/// Emit destruction of record types (or arrays of record types).
template <class Emitter>
bool ByteCodeExprGen<Emitter>::emitRecordDestruction(const Descriptor *Desc) {
  assert(Desc);
  assert(!Desc->isPrimitive());
  assert(!Desc->isPrimitiveArray());

  // Arrays.
  if (Desc->isArray()) {
    const Descriptor *ElemDesc = Desc->ElemDesc;
    assert(ElemDesc);

    // Don't need to do anything for these.
    if (ElemDesc->isPrimitiveArray())
      return this->emitPopPtr(SourceInfo{});

    // If this is an array of record types, check if we need
    // to call the element destructors at all. If not, try
    // to save the work.
    if (const Record *ElemRecord = ElemDesc->ElemRecord) {
      if (const CXXDestructorDecl *Dtor = ElemRecord->getDestructor();
          !Dtor || Dtor->isTrivial())
        return this->emitPopPtr(SourceInfo{});
    }

    for (ssize_t I = Desc->getNumElems() - 1; I >= 0; --I) {
      if (!this->emitConstUint64(I, SourceInfo{}))
        return false;
      if (!this->emitArrayElemPtrUint64(SourceInfo{}))
        return false;
      if (!this->emitRecordDestruction(ElemDesc))
        return false;
    }
    return this->emitPopPtr(SourceInfo{});
  }

  const Record *R = Desc->ElemRecord;
  assert(R);
  // First, destroy all fields.
  for (const Record::Field &Field : llvm::reverse(R->fields())) {
    const Descriptor *D = Field.Desc;
    if (!D->isPrimitive() && !D->isPrimitiveArray()) {
      if (!this->emitDupPtr(SourceInfo{}))
        return false;
      if (!this->emitGetPtrField(Field.Offset, SourceInfo{}))
        return false;
      if (!this->emitRecordDestruction(D))
        return false;
    }
  }

  // FIXME: Unions need to be handled differently here. We don't want to
  //   call the destructor of its members.

  // Now emit the destructor and recurse into base classes.
  if (const CXXDestructorDecl *Dtor = R->getDestructor();
      Dtor && !Dtor->isTrivial()) {
    if (const Function *DtorFunc = getFunction(Dtor)) {
      assert(DtorFunc->hasThisPointer());
      assert(DtorFunc->getNumParams() == 1);
      if (!this->emitDupPtr(SourceInfo{}))
        return false;
      if (!this->emitCall(DtorFunc, SourceInfo{}))
        return false;
    }
  }

  for (const Record::Base &Base : llvm::reverse(R->bases())) {
    if (!this->emitGetPtrBase(Base.Offset, SourceInfo{}))
      return false;
    if (!this->emitRecordDestruction(Base.Desc))
      return false;
  }
  // FIXME: Virtual bases.

  // Remove the instance pointer.
  return this->emitPopPtr(SourceInfo{});
}

namespace clang {
namespace interp {

template class ByteCodeExprGen<ByteCodeEmitter>;
template class ByteCodeExprGen<EvalEmitter>;

} // namespace interp
} // namespace clang
