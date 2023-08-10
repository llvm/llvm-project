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
#include "State.h"

using namespace clang;
using namespace clang::interp;

using APSInt = llvm::APSInt;

namespace clang {
namespace interp {

/// Scope used to handle temporaries in toplevel variable declarations.
template <class Emitter> class DeclScope final : public VariableScope<Emitter> {
public:
  DeclScope(ByteCodeExprGen<Emitter> *Ctx, const ValueDecl *VD)
      : VariableScope<Emitter>(Ctx), Scope(Ctx->P, VD) {}

  void addExtended(const Scope::Local &Local) override {
    return this->addLocal(Local);
  }

private:
  Program::DeclScope Scope;
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
  auto *SubExpr = CE->getSubExpr();
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
    if (!this->visit(SubExpr))
      return false;
    const auto *TargetSemantics = &Ctx.getFloatSemantics(CE->getType());
    return this->emitCastFP(TargetSemantics, getRoundingMode(CE), CE);
  }

  case CK_IntegralToFloating: {
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
    std::optional<PrimType> ToT = classify(CE->getType());

    if (!ToT)
      return false;

    if (!this->visit(SubExpr))
      return false;

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

  case CK_ArrayToPointerDecay:
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
bool ByteCodeExprGen<Emitter>::VisitParenExpr(const ParenExpr *PE) {
  const Expr *SubExpr = PE->getSubExpr();

  if (DiscardResult)
    return this->discard(SubExpr);

  return this->visit(SubExpr);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitBinaryOperator(const BinaryOperator *BO) {
  // Need short-circuiting for these.
  if (BO->isLogicalOp())
    return this->VisitLogicalBinOp(BO);

  const Expr *LHS = BO->getLHS();
  const Expr *RHS = BO->getRHS();

  if (BO->isPtrMemOp())
    return this->visit(RHS);

  // Typecheck the args.
  std::optional<PrimType> LT = classify(LHS->getType());
  std::optional<PrimType> RT = classify(RHS->getType());
  std::optional<PrimType> T = classify(BO->getType());

  auto Discard = [this, T, BO](bool Result) {
    if (!Result)
      return false;
    return DiscardResult ? this->emitPop(*T, BO) : true;
  };

  // Deal with operations which have composite or void types.
  if (BO->isCommaOp()) {
    if (!this->discard(LHS))
      return false;
    if (RHS->getType()->isVoidType())
      return this->discard(RHS);

    // Otherwise, visit RHS and optionally discard its value.
    return Discard(Initializing ? this->visitInitializer(RHS)
                                : this->visit(RHS));
  }

  if (!LT || !RT || !T)
    return this->bail(BO);

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
      return this->emitStorePop(*T, BO);
    return this->emitStore(*T, BO);
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
    return this->bail(BO);
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

  return this->bail(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitLogicalBinOp(const BinaryOperator *E) {
  assert(E->isLogicalOp());
  BinaryOperatorKind Op = E->getOpcode();
  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();

  if (Op == BO_LOr) {
    // Logical OR. Visit LHS and only evaluate RHS if LHS was FALSE.
    LabelTy LabelTrue = this->getLabel();
    LabelTy LabelEnd = this->getLabel();

    if (!this->visit(LHS))
      return false;
    if (!this->jumpTrue(LabelTrue))
      return false;

    if (!this->visit(RHS))
      return false;
    if (!this->jump(LabelEnd))
      return false;

    this->emitLabel(LabelTrue);
    this->emitConstBool(true, E);
    this->fallthrough(LabelEnd);
    this->emitLabel(LabelEnd);

    if (DiscardResult)
      return this->emitPopBool(E);

    return true;
  }

  // Logical AND.
  // Visit LHS. Only visit RHS if LHS was TRUE.
  LabelTy LabelFalse = this->getLabel();
  LabelTy LabelEnd = this->getLabel();

  if (!this->visit(LHS))
    return false;
  if (!this->jumpFalse(LabelFalse))
    return false;

  if (!this->visit(RHS))
    return false;
  if (!this->jump(LabelEnd))
    return false;

  this->emitLabel(LabelFalse);
  this->emitConstBool(false, E);
  this->fallthrough(LabelEnd);
  this->emitLabel(LabelEnd);

  if (DiscardResult)
    return this->emitPopBool(E);

  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
  QualType QT = E->getType();

  if (classify(QT))
    return this->visitZeroInitializer(QT, E);

  if (QT->isRecordType())
    return false;

  if (QT->isArrayType()) {
    const ArrayType *AT = QT->getAsArrayTypeUnsafe();
    assert(AT);
    const auto *CAT = cast<ConstantArrayType>(AT);
    size_t NumElems = CAT->getSize().getZExtValue();

    if (std::optional<PrimType> ElemT = classify(CAT->getElementType())) {
      // TODO(perf): For int and bool types, we can probably just skip this
      //   since we memset our Block*s to 0 and so we have the desired value
      //   without this.
      for (size_t I = 0; I != NumElems; ++I) {
        if (!this->visitZeroInitializer(CAT->getElementType(), E))
          return false;
        if (!this->emitInitElem(*ElemT, I, E))
          return false;
      }
    } else {
      assert(false && "default initializer for non-primitive type");
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
    assert(E->getNumInits() == 1);
    assert(!DiscardResult);
    return this->delegate(E->inits()[0]);
  }

  QualType T = E->getType();
  if (T->isRecordType()) {
    const Record *R = getRecord(T);

    unsigned InitIndex = 0;
    for (const Expr *Init : E->inits()) {
      if (!this->emitDupPtr(E))
        return false;

      if (std::optional<PrimType> T = classify(Init)) {
        const Record::Field *FieldToInit = R->getField(InitIndex);
        if (!this->visit(Init))
          return false;

        if (!this->emitInitField(*T, FieldToInit->Offset, E))
          return false;

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

          if (!this->emitPopPtr(E))
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

  if (T->isArrayType()) {
    // FIXME: Array fillers.
    unsigned ElementIndex = 0;
    for (const Expr *Init : E->inits()) {
      if (std::optional<PrimType> T = classify(Init->getType())) {
        // Visit the primitive element like normal.
        if (!this->visit(Init))
          return false;
        if (!this->emitInitElem(*T, ElementIndex, Init))
          return false;
      } else {
        // Advance the pointer currently on the stack to the given
        // dimension.
        if (!this->emitConstUint32(ElementIndex, Init))
          return false;
        if (!this->emitArrayElemPtrUint32(Init))
          return false;
        if (!this->visitInitializer(Init))
          return false;
        if (!this->emitPopPtr(Init))
          return false;
      }

      ++ElementIndex;
    }
    return true;
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitSubstNonTypeTemplateParmExpr(
    const SubstNonTypeTemplateParmExpr *E) {
  return this->delegate(E->getReplacement());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitConstantExpr(const ConstantExpr *E) {
  // TODO: Check if the ConstantExpr already has a value set and if so,
  //   use that instead of evaluating it again.
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
  //   Investigate compiling this to a loop, or at least try to use
  //   the AILE's Common expr.
  const Expr *SubExpr = E->getSubExpr();
  size_t Size = E->getArraySize().getZExtValue();
  std::optional<PrimType> ElemT = classify(SubExpr->getType());

  // So, every iteration, we execute an assignment here
  // where the LHS is on the stack (the target array)
  // and the RHS is our SubExpr.
  for (size_t I = 0; I != Size; ++I) {
    ArrayIndexScope<Emitter> IndexScope(this, I);

    if (ElemT) {
      if (!this->visit(SubExpr))
        return false;
      if (!this->emitInitElem(*ElemT, I, E))
        return false;
    } else {
      // Get to our array element and recurse into visitInitializer()
      if (!this->emitConstUint64(I, SubExpr))
        return false;
      if (!this->emitArrayElemPtrUint64(SubExpr))
        return false;
      if (!visitInitializer(SubExpr))
        return false;
      if (!this->emitPopPtr(E))
        return false;
    }
  }
  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitOpaqueValueExpr(const OpaqueValueExpr *E) {
  if (Initializing)
    return this->visitInitializer(E->getSourceExpr());
  return this->visit(E->getSourceExpr());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *E) {
  const Expr *Condition = E->getCond();
  const Expr *TrueExpr = E->getTrueExpr();
  const Expr *FalseExpr = E->getFalseExpr();

  LabelTy LabelEnd = this->getLabel();   // Label after the operator.
  LabelTy LabelFalse = this->getLabel(); // Label for the false expr.

  if (!this->visit(Condition))
    return false;

  // C special case: Convert to bool because our jump ops need that.
  // TODO: We probably want this to be done in visitBool().
  if (std::optional<PrimType> CondT = classify(Condition->getType());
      CondT && CondT != PT_Bool) {
    if (!this->emitCast(*CondT, PT_Bool, E))
      return false;
  }

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
  unsigned N =
      std::min(unsigned(CAT->getSize().getZExtValue()), E->getLength());
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
  if (DiscardResult)
    return this->emitStorePop(*ResultT, E);
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

  std::optional<PrimType> SubExprT = classify(SubExpr);
  if (E->getStorageDuration() == SD_Static) {
    std::optional<unsigned> GlobalIndex = P.createGlobal(E);
    if (!GlobalIndex)
      return false;

    const LifetimeExtendedTemporaryDecl *TempDecl =
        E->getLifetimeExtendedTemporaryDecl();
    assert(TempDecl);

    if (SubExprT) {
      if (!this->visit(SubExpr))
        return false;
      if (!this->emitInitGlobalTemp(*SubExprT, *GlobalIndex, TempDecl, E))
        return false;
      return this->emitGetPtrGlobal(*GlobalIndex, E);
    }

    // Non-primitive values.
    if (!this->emitGetPtrGlobal(*GlobalIndex, E))
      return false;
    if (!this->visitInitializer(SubExpr))
      return false;
    return this->emitInitGlobalTempComp(TempDecl, E);
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
  if (Initializing)
    return this->visitInitializer(E->getSubExpr());
  return this->visit(E->getSubExpr());
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
  if (!E->isGLValue() && !classify(E->getType())) {
    std::optional<unsigned> LocalIndex = allocateLocal(E, /*IsExtended=*/true);
    if (!LocalIndex)
      return false;

    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
    return this->visitInitializer(E);
  }

  //  Otherwise,we have a primitive return value, produce the value directly
  //  and puish it on the stack.
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
  if (std::optional<PrimType> T = classify(E->getType()))
    return visit(E);
  return this->bail(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitZeroInitializer(QualType QT,
                                                    const Expr *E) {
  // FIXME: We need the QualType to get the float semantics, but that means we
  //   classify it over and over again in array situations.
  PrimType T = classifyPrim(QT);

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
      if (!this->visitZeroInitializer(QT, E))
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
        if (!this->visitZeroInitializer(ET, E))
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
    if (!this->emitPopPtr(E))
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
bool ByteCodeExprGen<Emitter>::emitConst(T Value, const Expr *E) {
  switch (classifyPrim(E->getType())) {
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
  case PT_FnPtr:
  case PT_Float:
    llvm_unreachable("Invalid integral type");
    break;
  }
  llvm_unreachable("unknown primitive type");
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::emitConst(const APSInt &Value, const Expr *E) {
  if (Value.isSigned())
    return this->emitConst(Value.getSExtValue(), E);
  return this->emitConst(Value.getZExtValue(), E);
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
  else
    return Ty->getAs<RecordType>();
}

template <class Emitter>
Record *ByteCodeExprGen<Emitter>::getRecord(QualType Ty) {
  if (auto *RecordTy = getRecordTy(Ty)) {
    return getRecord(RecordTy->getDecl());
  }
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
  if (!visit(E))
    return false;

  if (E->getType()->isVoidType())
    return this->emitRetVoid(E);

  if (std::optional<PrimType> T = classify(E))
    return this->emitRet(*T, E);
  return this->emitRetValue(E);
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

  // Get a pointer to the variable
  if (Context::shouldBeGloballyIndexed(VD)) {
    auto GlobalIndex = P.getGlobal(VD);
    assert(GlobalIndex); // visitVarDecl() didn't return false.
    if (!this->emitGetPtrGlobal(*GlobalIndex, VD))
      return false;
  } else {
    auto Local = Locals.find(VD);
    assert(Local != Locals.end()); // Same here.
    if (!this->emitGetPtrLocal(Local->second.Offset, VD))
      return false;
  }

  // Return the value
  if (std::optional<PrimType> VarT = classify(VD->getType())) {
    if (!this->emitLoadPop(*VarT, VD))
      return false;

    return this->emitRet(*VarT, VD);
  }

  return this->emitRetValue(VD);
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
      return this->bail(VD);

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
bool ByteCodeExprGen<Emitter>::VisitBuiltinCallExpr(const CallExpr *E) {
  const Function *Func = getFunction(E->getDirectCallee());
  if (!Func)
    return false;

  // Put arguments on the stack.
  for (const auto *Arg : E->arguments()) {
    if (!this->visit(Arg))
      return false;
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
      if (!isa<CXXMemberCallExpr>(E)) {
        if (!this->emitDupPtr(E))
          return false;
      }
    }
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
bool ByteCodeExprGen<Emitter>::VisitCXXMemberCallExpr(
    const CXXMemberCallExpr *E) {
  if (Initializing) {
    // If we're initializing, the current stack top is the pointer to
    // initialize, so dup that so this call has its own version.
    if (!this->emitDupPtr(E))
      return false;
  }

  if (!this->visit(E->getImplicitObjectArgument()))
    return false;

  return VisitCallExpr(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXDefaultInitExpr(
    const CXXDefaultInitExpr *E) {

  if (Initializing)
    return this->visitInitializer(E->getExpr());

  assert(classify(E->getType()));
  return this->visit(E->getExpr());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXDefaultArgExpr(
    const CXXDefaultArgExpr *E) {
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
    if (!this->visit(SubExpr))
      return false;
    // The Inv doesn't change anything, so skip it if we don't need the result.
    return DiscardResult ? this->emitPop(*T, E) : this->emitInvBool(E);
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
    if (!this->visit(SubExpr))
      return false;
    return DiscardResult ? this->emitPop(*T, E) : true;
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
  case UO_Real:   // __real x
  case UO_Imag:   // __imag x
  case UO_Extension:
    if (DiscardResult)
      return this->discard(SubExpr);
    return this->visit(SubExpr);
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

  return false;
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

/// When calling this, we have a pointer of the local-to-destroy
/// on the stack.
/// Emit destruction of record types (or arrays of record types).
/// FIXME: Handle virtual destructors.
template <class Emitter>
bool ByteCodeExprGen<Emitter>::emitRecordDestruction(const Descriptor *Desc) {
  assert(Desc);
  assert(!Desc->isPrimitive());
  assert(!Desc->isPrimitiveArray());

  // Arrays.
  if (Desc->isArray()) {
    const Descriptor *ElemDesc = Desc->ElemDesc;
    const Record *ElemRecord = ElemDesc->ElemRecord;
    assert(ElemRecord); // This is not a primitive array.

    if (const CXXDestructorDecl *Dtor = ElemRecord->getDestructor();
        Dtor && !Dtor->isTrivial()) {
      for (ssize_t I = Desc->getNumElems() - 1; I >= 0; --I) {
        if (!this->emitConstUint64(I, SourceInfo{}))
          return false;
        if (!this->emitArrayElemPtrUint64(SourceInfo{}))
          return false;
        if (!this->emitRecordDestruction(Desc->ElemDesc))
          return false;
      }
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
