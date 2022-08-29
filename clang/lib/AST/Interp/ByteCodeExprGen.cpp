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

  case CK_ArrayToPointerDecay:
  case CK_AtomicToNonAtomic:
  case CK_ConstructorConversion:
  case CK_FunctionToPointerDecay:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_NullToPointer:
    return this->Visit(SubExpr);

  case CK_IntegralToBoolean:
  case CK_IntegralCast: {
    Optional<PrimType> FromT = classify(SubExpr->getType());
    Optional<PrimType> ToT = classify(CE->getType());
    if (!FromT || !ToT)
      return false;

    if (!this->Visit(SubExpr))
      return false;

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

  auto Val = LE->getValue();
  QualType LitTy = LE->getType();
  if (Optional<PrimType> T = classify(LitTy))
    return emitConst(*T, getIntWidth(LitTy), LE->getValue(), LE);
  return this->bail(LE);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitParenExpr(const ParenExpr *PE) {
  return this->Visit(PE->getSubExpr());
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
    if (!this->Visit(RHS))
      return false;
    return true;
  default:
    break;
  }

  // Typecheck the args.
  Optional<PrimType> LT = classify(LHS->getType());
  Optional<PrimType> RT = classify(RHS->getType());
  if (!LT || !RT) {
    return this->bail(BO);
  }

  if (Optional<PrimType> T = classify(BO->getType())) {
    if (!visit(LHS))
      return false;
    if (!visit(RHS))
      return false;

    auto Discard = [this, T, BO](bool Result) {
      if (!Result)
        return false;
      return DiscardResult ? this->emitPop(*T, BO) : true;
    };

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
    case BO_Assign:
      if (!this->emitStore(*T, BO))
        return false;
      return DiscardResult ? this->emitPopPtr(BO) : true;
    default:
      return this->bail(BO);
    }
  }

  return this->bail(BO);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
  if (Optional<PrimType> T = classify(E))
    return this->emitZero(*T, E);

  return false;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::VisitSubstNonTypeTemplateParmExpr(
    const SubstNonTypeTemplateParmExpr *E) {
  return this->visit(E->getReplacement());
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::discard(const Expr *E) {
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
  } else if (auto Idx = getGlobalIdx(VD)) {
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
        return this->Visit(VD->getInit());
    }
  }

  // Value cannot be produced - try to emit pointer.
  return visit(LV) && Indirect(T);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::emitConst(PrimType T, unsigned NumBits,
                                         const APInt &Value, const Expr *E) {
  switch (T) {
  case PT_Sint8:
    return this->emitConstSint8(Value.getSExtValue(), E);
  case PT_Uint8:
    return this->emitConstUint8(Value.getZExtValue(), E);
  case PT_Sint16:
    return this->emitConstSint16(Value.getSExtValue(), E);
  case PT_Uint16:
    return this->emitConstUint16(Value.getZExtValue(), E);
  case PT_Sint32:
    return this->emitConstSint32(Value.getSExtValue(), E);
  case PT_Uint32:
    return this->emitConstUint32(Value.getZExtValue(), E);
  case PT_Sint64:
    return this->emitConstSint64(Value.getSExtValue(), E);
  case PT_Uint64:
    return this->emitConstUint64(Value.getZExtValue(), E);
  case PT_Bool:
    return this->emitConstBool(Value.getBoolValue(), E);
  case PT_Ptr:
    llvm_unreachable("Invalid integral type");
    break;
  }
  llvm_unreachable("unknown primitive type");
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
  bool IsTemporary = false;
  if (auto *VD = dyn_cast_or_null<ValueDecl>(Src.dyn_cast<const Decl *>())) {
    Key = VD;
    Ty = VD->getType();
  }
  if (auto *E = Src.dyn_cast<const Expr *>()) {
    IsTemporary = true;
    Ty = E->getType();
  }

  Descriptor *D = P.createDescriptor(Src, Ty.getTypePtr(),
                                     Ty.isConstQualified(), IsTemporary);
  if (!D)
    return {};

  Scope::Local Local = this->createLocal(D);
  if (Key)
    Locals.insert({Key, Local});
  VarScope->add(Local, IsExtended);
  return Local.Offset;
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::visitInitializer(
    const Expr *Init, InitFnRef InitFn) {
  OptionScope<Emitter> Scope(this, InitFn);
  return this->Visit(Init);
}

template <class Emitter>
bool ByteCodeExprGen<Emitter>::getPtrVarDecl(const VarDecl *VD, const Expr *E) {
  // Generate a pointer to the local, loading refs.
  if (Optional<unsigned> Idx = getGlobalIdx(VD)) {
    if (VD->getType()->isReferenceType())
      return this->emitGetGlobalPtr(*Idx, E);
    else
      return this->emitGetPtrGlobal(*Idx, E);
  }
  return this->bail(VD);
}

template <class Emitter>
llvm::Optional<unsigned>
ByteCodeExprGen<Emitter>::getGlobalIdx(const VarDecl *VD) {
  if (VD->isConstexpr()) {
    // Constexpr decl - it must have already been defined.
    return P.getGlobal(VD);
  }
  if (!VD->hasLocalStorage()) {
    // Not constexpr, but a global var - can have pointer taken.
    Program::DeclScope Scope(P, VD);
    return P.getOrCreateGlobal(VD);
  }
  return {};
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

  if (Optional<unsigned> I = P.createGlobal(VD)) {
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
    const Function *Func = P.getFunction(FuncDecl);

    // Templated functions might not have been compiled yet, so do it now.
    if (!Func) {
      if (auto R =
              ByteCodeStmtGen<ByteCodeEmitter>(Ctx, P).compileFunc(FuncDecl))
        Func = *R;
    }
    assert(Func);

    // If the function is being compiled right now, this is a recursive call.
    // In that case, the function can't be valid yet, even though it will be
    // later.
    // If the function is already fully compiled but not constexpr, it was
    // found to be faulty earlier on, so bail out.
    if (Func->isFullyCompiled() && !Func->isConstexpr())
      return false;

    QualType ReturnType = E->getCallReturnType(Ctx.getASTContext());
    Optional<PrimType> T = classify(ReturnType);

    if (T || ReturnType->isVoidType()) {
      // Put arguments on the stack.
      for (const auto *Arg : E->arguments()) {
        if (!this->visit(Arg))
          return false;
      }

      if (T)
        return this->emitCall(*T, Func, E);
      return this->emitCallVoid(Func, E);
    } else {
      assert(false && "Can't classify function return type");
    }

  } else {
    assert(false && "We don't support non-FunctionDecl callees right now.");
  }

  return false;
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
bool ByteCodeExprGen<Emitter>::VisitUnaryOperator(const UnaryOperator *E) {
  const Expr *SubExpr = E->getSubExpr();

  switch (E->getOpcode()) {
  case UO_PostInc: // x++
  case UO_PostDec: // x--
  case UO_PreInc:  // --x
  case UO_PreDec:  // ++x
    return false;

  case UO_LNot: // !x
    if (!this->Visit(SubExpr))
      return false;
    return this->emitInvBool(E);
  case UO_Minus: // -x
    if (!this->Visit(SubExpr))
      return false;
    if (Optional<PrimType> T = classify(E->getType()))
      return this->emitNeg(*T, E);
    return false;
  case UO_Plus:  // +x
    return this->Visit(SubExpr); // noop

  case UO_AddrOf: // &x
    // We should already have a pointer when we get here.
    return this->Visit(SubExpr);

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

  if (auto It = Locals.find(Decl); It != Locals.end()) {
    const unsigned Offset = It->second.Offset;
    return this->emitGetPtrLocal(Offset, E);
  } else if (auto GlobalIndex = P.getGlobal(Decl)) {
    return this->emitGetPtrGlobal(*GlobalIndex, E);
  } else if (const auto *PVD = dyn_cast<ParmVarDecl>(Decl)) {
    if (auto It = this->Params.find(PVD); It != this->Params.end())
      return this->emitGetPtrParam(It->second, E);
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
