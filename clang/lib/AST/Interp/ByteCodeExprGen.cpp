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
#include "Function.h"
#include "PrimType.h"
#include "Program.h"
#include "State.h"

using namespace clang;
using namespace clang::interp;

using APSInt = llvm::APSInt;
template <typename T> using Expected = llvm::Expected<T>;
template <typename T> using Optional = llvm::Optional<T>;

namespace clang {
namespace interp {

/// Scope used to handle temporaries in toplevel variable declarations.
template <class Emitter> class DeclScope final : public LocalScope<Emitter> {
public:
  DeclScope(ByteCodeExprGen<Emitter> *Ctx, const VarDecl *VD)
      : LocalScope<Emitter>(Ctx), Scope(Ctx->P, VD) {}

  void addExtended(const Scope::Local &Local) override {
    return this->addLocal(Local);
  }

private:
  Program::DeclScope Scope;
};

/// Scope used to handle initialization methods.
template <class Emitter> class OptionScope {
public:
  using InitFnRef = typename ByteCodeExprGen<Emitter>::InitFnRef;
  using ChainedInitFnRef = std::function<bool(InitFnRef)>;

  /// Root constructor, compiling or discarding primitives.
  OptionScope(ByteCodeExprGen<Emitter> *Ctx, bool NewDiscardResult)
      : Ctx(Ctx), OldDiscardResult(Ctx->DiscardResult),
        OldInitFn(std::move(Ctx->InitFn)) {
    Ctx->DiscardResult = NewDiscardResult;
    Ctx->InitFn = llvm::Optional<InitFnRef>{};
  }

  /// Root constructor, setting up compilation state.
  OptionScope(ByteCodeExprGen<Emitter> *Ctx, InitFnRef NewInitFn)
      : Ctx(Ctx), OldDiscardResult(Ctx->DiscardResult),
        OldInitFn(std::move(Ctx->InitFn)) {
    Ctx->DiscardResult = true;
    Ctx->InitFn = NewInitFn;
  }

  /// Extends the chain of initialisation pointers.
  OptionScope(ByteCodeExprGen<Emitter> *Ctx, ChainedInitFnRef NewInitFn)
      : Ctx(Ctx), OldDiscardResult(Ctx->DiscardResult),
        OldInitFn(std::move(Ctx->InitFn)) {
    assert(OldInitFn && "missing initializer");
    Ctx->InitFn = [this, NewInitFn] { return NewInitFn(*OldInitFn); };
  }

  ~OptionScope() {
    Ctx->DiscardResult = OldDiscardResult;
    Ctx->InitFn = std::move(OldInitFn);
  }

private:
  /// Parent context.
  ByteCodeExprGen<Emitter> *Ctx;
  /// Old discard flag to restore.
  bool OldDiscardResult;
  /// Old pointer emitter to restore.
  llvm::Optional<InitFnRef> OldInitFn;
};

} // namespace interp
} // namespace clang

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCastExpr(const CastExpr *CE) {
  auto *SubExpr = CE->getSubExpr();
  switch (CE->getCastKind()) {

  case CK_LValueToRValue: {
    return dereference(
        CE->getSubExpr(), DerefKind::Read,
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

    const CXXRecordDecl *FromDecl = getRecordDecl(SubExpr);
    assert(FromDecl);
    const CXXRecordDecl *ToDecl = getRecordDecl(CE);
    assert(ToDecl);
    const Record *R = getRecord(FromDecl);
    const Record::Base *ToBase = R->getBase(ToDecl);
    assert(ToBase);

    return this->emitGetPtrBase(ToBase->Offset, CE);
  }

  case CK_ArrayToPointerDecay:
  case CK_AtomicToNonAtomic:
  case CK_ConstructorConversion:
  case CK_FunctionToPointerDecay:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_NullToPointer:
    return this->visit(SubExpr);

  case CK_IntegralToBoolean:
  case CK_IntegralCast: {
    Optional<PrimType> FromT = classify(SubExpr->getType());
    Optional<PrimType> ToT = classify(CE->getType());
    if (!FromT || !ToT)
      return false;

    if (!this->visit(SubExpr))
      return false;

    // TODO: Emit this only if FromT != ToT.
    return this->emitCast(*FromT, *ToT, CE);
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
bool ByteCodeExprGen<Emitter>::VisitParenExpr(const ParenExpr *PE) {
  return this->visit(PE->getSubExpr());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitBinaryOperator(const BinaryOperator *BO) {
  const Expr *LHS = BO->getLHS();
  const Expr *RHS = BO->getRHS();

  // Deal with operations which have composite or void types.
  switch (BO->getOpcode()) {
  case BO_Comma:
    if (!discard(LHS))
      return false;
    if (!this->visit(RHS))
      return false;
    return true;
  default:
    break;
  }

  // Typecheck the args.
  Optional<PrimType> LT = classify(LHS->getType());
  Optional<PrimType> RT = classify(RHS->getType());
  Optional<PrimType> T = classify(BO->getType());
  if (!LT || !RT || !T) {
    return this->bail(BO);
  }

  auto Discard = [this, T, BO](bool Result) {
    if (!Result)
      return false;
    return DiscardResult ? this->emitPop(*T, BO) : true;
  };

  // Pointer arithmetic special case.
  if (BO->getOpcode() == BO_Add || BO->getOpcode() == BO_Sub) {
    if (*T == PT_Ptr || (*LT == PT_Ptr && *RT == PT_Ptr))
      return this->VisitPointerArithBinOp(BO);
  }

  if (!visit(LHS) || !visit(RHS))
    return false;

  switch (BO->getOpcode()) {
  case BO_EQ:
    return Discard(this->emitEQ(*LT, BO));
  case BO_NE:
    return Discard(this->emitNE(*LT, BO));
  case BO_LT:
    return Discard(this->emitLT(*LT, BO));
  case BO_LE:
    return Discard(this->emitLE(*LT, BO));
  case BO_GT:
    return Discard(this->emitGT(*LT, BO));
  case BO_GE:
    return Discard(this->emitGE(*LT, BO));
  case BO_Sub:
    return Discard(this->emitSub(*T, BO));
  case BO_Add:
    return Discard(this->emitAdd(*T, BO));
  case BO_Mul:
    return Discard(this->emitMul(*T, BO));
  case BO_Rem:
    return Discard(this->emitRem(*T, BO));
  case BO_Div:
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
  case BO_LAnd:
  case BO_LOr:
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

  Optional<PrimType> LT = classify(LHS);
  Optional<PrimType> RT = classify(RHS);

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
bool ByteCodeExprGen<Emitter>::VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
  if (Optional<PrimType> T = classify(E))
    return this->emitZero(*T, E);

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitArraySubscriptExpr(
    const ArraySubscriptExpr *E) {
  const Expr *Base = E->getBase();
  const Expr *Index = E->getIdx();
  PrimType IndexT = classifyPrim(Index->getType());

  // Take pointer of LHS, add offset from RHS, narrow result.
  // What's left on the stack after this is a pointer.
  if (!this->visit(Base))
    return false;

  if (!this->visit(Index))
    return false;

  if (!this->emitAddOffset(IndexT, E))
    return false;

  if (!this->emitNarrowPtr(E))
    return false;

  if (DiscardResult)
    return this->emitPopPtr(E);

  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitInitListExpr(const InitListExpr *E) {
  for (const Expr *Init : E->inits()) {
    if (!this->visit(Init))
      return false;
  }
  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitSubstNonTypeTemplateParmExpr(
    const SubstNonTypeTemplateParmExpr *E) {
  return this->visit(E->getReplacement());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitConstantExpr(const ConstantExpr *E) {
  // TODO: Check if the ConstantExpr already has a value set and if so,
  //   use that instead of evaluating it again.
  return this->visit(E->getSubExpr());
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

    return this->emitConst(Size.getQuantity(), E);
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitMemberExpr(const MemberExpr *E) {
  if (DiscardResult)
    return true;

  // 'Base.Member'
  const Expr *Base = E->getBase();
  const ValueDecl *Member = E->getMemberDecl();

  if (!this->visit(Base))
    return false;

  // Base above gives us a pointer on the stack.
  // TODO: Implement non-FieldDecl members.
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
bool ByteCodeExprGen<Emitter>::VisitOpaqueValueExpr(const OpaqueValueExpr *E) {
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
  if (!this->jumpFalse(LabelFalse))
    return false;

  if (!this->visit(TrueExpr))
    return false;
  if (!this->jump(LabelEnd))
    return false;

  this->emitLabel(LabelFalse);

  if (!this->visit(FalseExpr))
    return false;

  this->fallthrough(LabelEnd);
  this->emitLabel(LabelEnd);

  return true;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitStringLiteral(const StringLiteral *E) {
  unsigned StringIndex = P.createGlobalString(E);
  return this->emitGetPtrGlobal(StringIndex, E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCharacterLiteral(
    const CharacterLiteral *E) {
  return this->emitConst(E->getValue(), E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCompoundAssignOperator(
    const CompoundAssignOperator *E) {
  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();
  Optional<PrimType> LT = classify(E->getLHS()->getType());
  Optional<PrimType> RT = classify(E->getRHS()->getType());

  if (!LT || !RT)
    return false;

  assert(!E->getType()->isPointerType() &&
         "Support pointer arithmethic in compound assignment operators");

  // Get LHS pointer, load its value and get RHS value.
  if (!visit(LHS))
    return false;
  if (!this->emitLoad(*LT, E))
    return false;
  if (!visit(RHS))
    return false;

  // Perform operation.
  switch (E->getOpcode()) {
  case BO_AddAssign:
    if (!this->emitAdd(*LT, E))
      return false;
    break;
  case BO_SubAssign:
    if (!this->emitSub(*LT, E))
      return false;
    break;

  case BO_MulAssign:
  case BO_DivAssign:
  case BO_RemAssign:
  case BO_ShlAssign:
    if (!this->emitShl(*LT, *RT, E))
      return false;
    break;
  case BO_ShrAssign:
    if (!this->emitShr(*LT, *RT, E))
      return false;
    break;
  case BO_AndAssign:
  case BO_XorAssign:
  case BO_OrAssign:
  default:
    llvm_unreachable("Unimplemented compound assign operator");
  }

  // And store the result in LHS.
  if (DiscardResult)
    return this->emitStorePop(*LT, E);
  return this->emitStore(*LT, E);
}

template <class Emitter> bool ByteCodeExprGen<Emitter>::discard(const Expr *E) {
  OptionScope<Emitter> Scope(this, /*NewDiscardResult=*/true);
  return this->Visit(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visit(const Expr *E) {
  OptionScope<Emitter> Scope(this, /*NewDiscardResult=*/false);
  return this->Visit(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitBool(const Expr *E) {
  if (Optional<PrimType> T = classify(E->getType())) {
    return visit(E);
  } else {
    return this->bail(E);
  }
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitZeroInitializer(PrimType T, const Expr *E) {
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
  }
  llvm_unreachable("unknown primitive type");
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::dereference(
    const Expr *LV, DerefKind AK, llvm::function_ref<bool(PrimType)> Direct,
    llvm::function_ref<bool(PrimType)> Indirect) {
  if (Optional<PrimType> T = classify(LV->getType())) {
    if (!LV->refersToBitField()) {
      // Only primitive, non bit-field types can be dereferenced directly.
      if (auto *DE = dyn_cast<DeclRefExpr>(LV)) {
        if (!DE->getDecl()->getType()->isReferenceType()) {
          if (auto *PD = dyn_cast<ParmVarDecl>(DE->getDecl()))
            return dereferenceParam(LV, *T, PD, AK, Direct, Indirect);
          if (auto *VD = dyn_cast<VarDecl>(DE->getDecl()))
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
    unsigned Idx = It->second;
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
  Descriptor *D = P.createDescriptor(Src, Ty, IsConst, Src.is<const Expr *>());
  Scope::Local Local = this->createLocal(D);
  if (auto *VD = dyn_cast_or_null<ValueDecl>(Src.dyn_cast<const Decl *>()))
    Locals.insert({VD, Local});
  VarScope->add(Local, IsExtended);
  return Local.Offset;
}

template <class Emitter>
llvm::Optional<unsigned>
ByteCodeExprGen<Emitter>::allocateLocal(DeclTy &&Src, bool IsExtended) {
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
      Src, Ty.getTypePtr(), Ty.isConstQualified(), IsTemporary, false, Init);
  if (!D)
    return {};

  Scope::Local Local = this->createLocal(D);
  if (Key)
    Locals.insert({Key, Local});
  VarScope->add(Local, IsExtended);
  return Local.Offset;
}

// NB: When calling this function, we have a pointer to the
//   array-to-initialize on the stack.
template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitArrayInitializer(const Expr *Initializer) {
  assert(Initializer->getType()->isArrayType());

  // TODO: Fillers?
  if (const auto *InitList = dyn_cast<InitListExpr>(Initializer)) {
    unsigned ElementIndex = 0;
    for (const Expr *Init : InitList->inits()) {
      if (Optional<PrimType> T = classify(Init->getType())) {
        // Visit the primitive element like normal.
        if (!this->emitDupPtr(Init))
          return false;
        if (!this->visit(Init))
          return false;
        if (!this->emitInitElem(*T, ElementIndex, Init))
          return false;
      } else {
        // Advance the pointer currently on the stack to the given
        // dimension and narrow().
        if (!this->emitDupPtr(Init))
          return false;
        if (!this->emitConstUint32(ElementIndex, Init))
          return false;
        if (!this->emitAddOffsetUint32(Init))
          return false;
        if (!this->emitNarrowPtr(Init))
          return false;

        if (!visitInitializer(Init))
          return false;
      }
        if (!this->emitPopPtr(Init))
          return false;

      ++ElementIndex;
    }
    return true;
  } else if (const auto *DIE = dyn_cast<CXXDefaultInitExpr>(Initializer)) {
    return this->visitInitializer(DIE->getExpr());
  } else if (const auto *AILE = dyn_cast<ArrayInitLoopExpr>(Initializer)) {
    // TODO: This compiles to quite a lot of bytecode if the array is larger.
    //   Investigate compiling this to a loop, or at least try to use
    //   the AILE's Common expr.
    const Expr *SubExpr = AILE->getSubExpr();
    size_t Size = AILE->getArraySize().getZExtValue();
    Optional<PrimType> ElemT = classify(SubExpr->getType());

    // So, every iteration, we execute an assignment here
    // where the LHS is on the stack (the target array)
    // and the RHS is our SubExpr.
    for (size_t I = 0; I != Size; ++I) {
      ArrayIndexScope<Emitter> IndexScope(this, I);

      if (!this->emitDupPtr(SubExpr)) // LHS
        return false;

      if (ElemT) {
        if (!this->visit(SubExpr))
          return false;
        if (!this->emitInitElem(*ElemT, I, Initializer))
          return false;
      } else {
        // Narrow to our array element and recurse into visitInitializer()
        if (!this->emitConstUint64(I, SubExpr))
          return false;

        if (!this->emitAddOffsetUint64(SubExpr))
          return false;

        if (!this->emitNarrowPtr(SubExpr))
          return false;

        if (!visitInitializer(SubExpr))
          return false;
      }

      if (!this->emitPopPtr(Initializer))
        return false;
    }
    return true;
  } else if (const auto *IVIE = dyn_cast<ImplicitValueInitExpr>(Initializer)) {
    const ArrayType *AT = IVIE->getType()->getAsArrayTypeUnsafe();
    assert(AT);
    const auto *CAT = cast<ConstantArrayType>(AT);
    size_t NumElems = CAT->getSize().getZExtValue();

    if (Optional<PrimType> ElemT = classify(CAT->getElementType())) {
      // TODO(perf): For int and bool types, we can probably just skip this
      //   since we memset our Block*s to 0 and so we have the desired value
      //   without this.
      for (size_t I = 0; I != NumElems; ++I) {
        if (!this->emitZero(*ElemT, Initializer))
          return false;
        if (!this->emitInitElem(*ElemT, I, Initializer))
          return false;
      }
    } else {
      assert(false && "default initializer for non-primitive type");
    }

    return true;
  } else if (const auto *Ctor = dyn_cast<CXXConstructExpr>(Initializer)) {
    const ConstantArrayType *CAT =
        Ctx.getASTContext().getAsConstantArrayType(Ctor->getType());
    assert(CAT);
    size_t NumElems = CAT->getSize().getZExtValue();
    const Function *Func = getFunction(Ctor->getConstructor());
    if (!Func || !Func->isConstexpr())
      return false;

    // FIXME(perf): We're calling the constructor once per array element here,
    //   in the old intepreter we had a special-case for trivial constructors.
    for (size_t I = 0; I != NumElems; ++I) {
      if (!this->emitDupPtr(Initializer))
        return false;
      if (!this->emitConstUint64(I, Initializer))
        return false;
      if (!this->emitAddOffsetUint64(Initializer))
        return false;
      if (!this->emitNarrowPtr(Initializer))
        return false;

      // Constructor arguments.
      for (const auto *Arg : Ctor->arguments()) {
        if (!this->visit(Arg))
          return false;
      }

      if (!this->emitCall(Func, Initializer))
        return false;
    }
    return true;
  }

  assert(false && "Unknown expression for array initialization");
  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitRecordInitializer(const Expr *Initializer) {
  Initializer = Initializer->IgnoreParenImpCasts();
  assert(Initializer->getType()->isRecordType());

  if (const auto CtorExpr = dyn_cast<CXXConstructExpr>(Initializer)) {
    const Function *Func = getFunction(CtorExpr->getConstructor());

    if (!Func || !Func->isConstexpr())
      return false;

    // The This pointer is already on the stack because this is an initializer,
    // but we need to dup() so the call() below has its own copy.
    if (!this->emitDupPtr(Initializer))
      return false;

    // Constructor arguments.
    for (const auto *Arg : CtorExpr->arguments()) {
      if (!this->visit(Arg))
        return false;
    }

    return this->emitCall(Func, Initializer);
  } else if (const auto *InitList = dyn_cast<InitListExpr>(Initializer)) {
    const Record *R = getRecord(InitList->getType());

    unsigned InitIndex = 0;
    for (const Expr *Init : InitList->inits()) {
      const Record::Field *FieldToInit = R->getField(InitIndex);

      if (!this->emitDupPtr(Initializer))
        return false;

      if (Optional<PrimType> T = classify(Init)) {
        if (!this->visit(Init))
          return false;

        if (!this->emitInitField(*T, FieldToInit->Offset, Initializer))
          return false;

        if (!this->emitPopPtr(Initializer))
          return false;
      } else {
        // Non-primitive case. Get a pointer to the field-to-initialize
        // on the stack and recurse into visitInitializer().
        if (!this->emitGetPtrField(FieldToInit->Offset, Init))
          return false;

        if (!this->visitInitializer(Init))
          return false;

        if (!this->emitPopPtr(Initializer))
          return false;
      }
      ++InitIndex;
    }

    return true;
  } else if (const CallExpr *CE = dyn_cast<CallExpr>(Initializer)) {
    const Decl *Callee = CE->getCalleeDecl();
    const Function *Func = getFunction(dyn_cast<FunctionDecl>(Callee));

    if (!Func)
      return false;

    if (Func->hasRVO()) {
      // RVO functions expect a pointer to initialize on the stack.
      // Dup our existing pointer so it has its own copy to use.
      if (!this->emitDupPtr(Initializer))
        return false;

      return this->visit(CE);
    }
  } else if (const auto *DIE = dyn_cast<CXXDefaultInitExpr>(Initializer)) {
    return this->visitInitializer(DIE->getExpr());
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitInitializer(const Expr *Initializer) {
  QualType InitializerType = Initializer->getType();

  if (InitializerType->isArrayType())
    return visitArrayInitializer(Initializer);

  if (InitializerType->isRecordType())
    return visitRecordInitializer(Initializer);

  // Otherwise, visit the expression like normal.
  return this->visit(Initializer);
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
  assert(FD);
  const Function *Func = P.getFunction(FD);
  bool IsBeingCompiled = Func && !Func->isFullyCompiled();
  bool WasNotDefined = Func && !Func->hasBody();

  if (IsBeingCompiled)
    return Func;

  if (!Func || WasNotDefined) {
    if (auto R = ByteCodeStmtGen<ByteCodeEmitter>(Ctx, P).compileFunc(FD))
      Func = *R;
    else {
      llvm::consumeError(R.takeError());
      return nullptr;
    }
  }

  return Func;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitExpr(const Expr *Exp) {
  ExprScope<Emitter> RootScope(this);
  if (!visit(Exp))
    return false;

  if (Optional<PrimType> T = classify(Exp))
    return this->emitRet(*T, Exp);
  else
    return this->emitRetValue(Exp);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitDecl(const VarDecl *VD) {
  const Expr *Init = VD->getInit();

  if (Optional<unsigned> I = P.createGlobal(VD, Init)) {
    if (Optional<PrimType> T = classify(VD->getType())) {
      {
        // Primitive declarations - compute the value and set it.
        DeclScope<Emitter> LocalScope(this, VD);
        if (!visit(Init))
          return false;
      }

      // If the declaration is global, save the value for later use.
      if (!this->emitDup(*T, VD))
        return false;
      if (!this->emitInitGlobal(*T, *I, VD))
        return false;
      return this->emitRet(*T, VD);
    } else {
      {
        // Composite declarations - allocate storage and initialize it.
        DeclScope<Emitter> LocalScope(this, VD);
        if (!visitGlobalInitializer(Init, *I))
          return false;
      }

      // Return a pointer to the global.
      if (!this->emitGetPtrGlobal(*I, VD))
        return false;
      return this->emitRetValue(VD);
    }
  }

  return this->bail(VD);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCallExpr(const CallExpr *E) {
  assert(!E->getBuiltinCallee() && "Builtin functions aren't supported yet");

  const Decl *Callee = E->getCalleeDecl();
  if (const auto *FuncDecl = dyn_cast_or_null<FunctionDecl>(Callee)) {
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

    QualType ReturnType = E->getCallReturnType(Ctx.getASTContext());
    Optional<PrimType> T = classify(ReturnType);

    if (Func->hasRVO() && DiscardResult) {
      // If we need to discard the return value but the function returns its
      // value via an RVO pointer, we need to create one such pointer just
      // for this call.
      if (Optional<unsigned> LocalIndex = allocateLocal(E)) {
        if (!this->emitGetPtrLocal(*LocalIndex, E))
          return false;
      }
    }

    // Put arguments on the stack.
    for (const auto *Arg : E->arguments()) {
      if (!this->visit(Arg))
        return false;
    }

    // In any case call the function. The return value will end up on the stack and
    // if the function has RVO, we already have the pointer on the stack to write
    // the result into.
    if (!this->emitCall(Func, E))
      return false;

    if (DiscardResult && !ReturnType->isVoidType() && T)
      return this->emitPop(*T, E);

    return true;
  } else {
    assert(false && "We don't support non-FunctionDecl callees right now.");
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXMemberCallExpr(
    const CXXMemberCallExpr *E) {
  // Get a This pointer on the stack.
  if (!this->visit(E->getImplicitObjectArgument()))
    return false;

  return VisitCallExpr(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXDefaultInitExpr(
    const CXXDefaultInitExpr *E) {
  return this->visit(E->getExpr());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitCXXDefaultArgExpr(
    const CXXDefaultArgExpr *E) {
  return this->visit(E->getExpr());
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
bool ByteCodeExprGen<Emitter>::VisitCXXThisExpr(const CXXThisExpr *E) {
  if (DiscardResult)
    return true;
  return this->emitThis(E);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitUnaryOperator(const UnaryOperator *E) {
  const Expr *SubExpr = E->getSubExpr();
  Optional<PrimType> T = classify(SubExpr->getType());

  // TODO: Support pointers for inc/dec operators.
  switch (E->getOpcode()) {
  case UO_PostInc: { // x++
    if (!this->visit(SubExpr))
      return false;

    return DiscardResult ? this->emitIncPop(*T, E) : this->emitInc(*T, E);
  }
  case UO_PostDec: { // x--
    if (!this->visit(SubExpr))
      return false;

    return DiscardResult ? this->emitDecPop(*T, E) : this->emitDec(*T, E);
  }
  case UO_PreInc: { // ++x
    if (!this->visit(SubExpr))
      return false;

    // Post-inc and pre-inc are the same if the value is to be discarded.
    if (DiscardResult)
      return this->emitIncPop(*T, E);

    this->emitLoad(*T, E);
    this->emitConst(1, E);
    this->emitAdd(*T, E);
    return this->emitStore(*T, E);
  }
  case UO_PreDec: { // --x
    if (!this->visit(SubExpr))
      return false;

    // Post-dec and pre-dec are the same if the value is to be discarded.
    if (DiscardResult)
      return this->emitDecPop(*T, E);

    this->emitLoad(*T, E);
    this->emitConst(1, E);
    this->emitSub(*T, E);
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
  case UO_Coawait:
    assert(false && "Unhandled opcode");
  }

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitDeclRefExpr(const DeclRefExpr *E) {
  const auto *Decl = E->getDecl();
  // References are implemented via pointers, so when we see a DeclRefExpr
  // pointing to a reference, we need to get its value directly (i.e. the
  // pointer to the actual value) instead of a pointer to the pointer to the
  // value.
  bool IsReference = Decl->getType()->isReferenceType();

  if (auto It = Locals.find(Decl); It != Locals.end()) {
    const unsigned Offset = It->second.Offset;

    if (IsReference)
      return this->emitGetLocal(PT_Ptr, Offset, E);
    return this->emitGetPtrLocal(Offset, E);
  } else if (auto GlobalIndex = P.getGlobal(Decl)) {
    if (IsReference)
      return this->emitGetGlobal(PT_Ptr, *GlobalIndex, E);

    return this->emitGetPtrGlobal(*GlobalIndex, E);
  } else if (const auto *PVD = dyn_cast<ParmVarDecl>(Decl)) {
    if (auto It = this->Params.find(PVD); It != this->Params.end()) {
      if (IsReference)
        return this->emitGetParam(PT_Ptr, It->second, E);
      return this->emitGetPtrParam(It->second, E);
    }
  } else if (const auto *ECD = dyn_cast<EnumConstantDecl>(Decl)) {
    return this->emitConst(ECD->getInitVal(), E);
  }

  return false;
}

template <class Emitter>
void ByteCodeExprGen<Emitter>::emitCleanup() {
  for (VariableScope<Emitter> *C = VarScope; C; C = C->getParent())
    C->emitDestruction();
}

namespace clang {
namespace interp {

template class ByteCodeExprGen<ByteCodeEmitter>;
template class ByteCodeExprGen<EvalEmitter>;

} // namespace interp
} // namespace clang
