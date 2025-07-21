//===--- Context.cpp - Context for the constexpr VM -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "ByteCodeEmitter.h"
#include "Compiler.h"
#include "EvalEmitter.h"
#include "Interp.h"
#include "InterpFrame.h"
#include "InterpStack.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang;
using namespace clang::interp;

Context::Context(ASTContext &Ctx) : Ctx(Ctx), P(new Program(*this)) {
  this->ShortWidth = Ctx.getTargetInfo().getShortWidth();
  this->IntWidth = Ctx.getTargetInfo().getIntWidth();
  this->LongWidth = Ctx.getTargetInfo().getLongWidth();
  this->LongLongWidth = Ctx.getTargetInfo().getLongLongWidth();
  assert(Ctx.getTargetInfo().getCharWidth() == 8 &&
         "We're assuming 8 bit chars");
}

Context::~Context() {}

bool Context::isPotentialConstantExpr(State &Parent, const FunctionDecl *FD) {
  assert(Stk.empty());

  // Get a function handle.
  const Function *Func = getOrCreateFunction(FD);
  if (!Func)
    return false;

  // Compile the function.
  Compiler<ByteCodeEmitter>(*this, *P).compileFunc(
      FD, const_cast<Function *>(Func));

  ++EvalID;
  // And run it.
  if (!Run(Parent, Func))
    return false;

  return Func->isValid();
}

void Context::isPotentialConstantExprUnevaluated(State &Parent, const Expr *E,
                                                 const FunctionDecl *FD) {
  assert(Stk.empty());
  ++EvalID;
  size_t StackSizeBefore = Stk.size();
  Compiler<EvalEmitter> C(*this, *P, Parent, Stk);

  if (!C.interpretCall(FD, E)) {
    C.cleanup();
    Stk.clearTo(StackSizeBefore);
  }
}

bool Context::evaluateAsRValue(State &Parent, const Expr *E, APValue &Result) {
  ++EvalID;
  bool Recursing = !Stk.empty();
  size_t StackSizeBefore = Stk.size();
  Compiler<EvalEmitter> C(*this, *P, Parent, Stk);

  auto Res = C.interpretExpr(E, /*ConvertResultToRValue=*/E->isGLValue());

  if (Res.isInvalid()) {
    C.cleanup();
    Stk.clearTo(StackSizeBefore);
    return false;
  }

  if (!Recursing) {
    // We *can* actually get here with a non-empty stack, since
    // things like InterpState::noteSideEffect() exist.
    C.cleanup();
#ifndef NDEBUG
    // Make sure we don't rely on some value being still alive in
    // InterpStack memory.
    Stk.clearTo(StackSizeBefore);
#endif
  }

  Result = Res.toAPValue();

  return true;
}

bool Context::evaluate(State &Parent, const Expr *E, APValue &Result,
                       ConstantExprKind Kind) {
  ++EvalID;
  bool Recursing = !Stk.empty();
  size_t StackSizeBefore = Stk.size();
  Compiler<EvalEmitter> C(*this, *P, Parent, Stk);

  auto Res = C.interpretExpr(E, /*ConvertResultToRValue=*/false,
                             /*DestroyToplevelScope=*/true);
  if (Res.isInvalid()) {
    C.cleanup();
    Stk.clearTo(StackSizeBefore);
    return false;
  }

  if (!Recursing) {
    assert(Stk.empty());
    C.cleanup();
#ifndef NDEBUG
    // Make sure we don't rely on some value being still alive in
    // InterpStack memory.
    Stk.clearTo(StackSizeBefore);
#endif
  }

  Result = Res.toAPValue();
  return true;
}

bool Context::evaluateAsInitializer(State &Parent, const VarDecl *VD,
                                    APValue &Result) {
  ++EvalID;
  bool Recursing = !Stk.empty();
  size_t StackSizeBefore = Stk.size();
  Compiler<EvalEmitter> C(*this, *P, Parent, Stk);

  bool CheckGlobalInitialized =
      shouldBeGloballyIndexed(VD) &&
      (VD->getType()->isRecordType() || VD->getType()->isArrayType());
  auto Res = C.interpretDecl(VD, CheckGlobalInitialized);
  if (Res.isInvalid()) {
    C.cleanup();
    Stk.clearTo(StackSizeBefore);

    return false;
  }

  if (!Recursing) {
    assert(Stk.empty());
    C.cleanup();
#ifndef NDEBUG
    // Make sure we don't rely on some value being still alive in
    // InterpStack memory.
    Stk.clearTo(StackSizeBefore);
#endif
  }

  Result = Res.toAPValue();
  return true;
}

template <typename ResultT>
bool Context::evaluateStringRepr(State &Parent, const Expr *SizeExpr,
                                 const Expr *PtrExpr, ResultT &Result) {
  assert(Stk.empty());
  Compiler<EvalEmitter> C(*this, *P, Parent, Stk);

  // Evaluate size value.
  APValue SizeValue;
  if (!evaluateAsRValue(Parent, SizeExpr, SizeValue))
    return false;

  if (!SizeValue.isInt())
    return false;
  uint64_t Size = SizeValue.getInt().getZExtValue();

  auto PtrRes = C.interpretAsPointer(PtrExpr, [&](const Pointer &Ptr) {
    if (Size == 0) {
      if constexpr (std::is_same_v<ResultT, APValue>)
        Result = APValue(APValue::UninitArray{}, 0, 0);
      return true;
    }

    if (!Ptr.isLive() || !Ptr.getFieldDesc()->isPrimitiveArray())
      return false;

    // Must be char.
    if (Ptr.getFieldDesc()->getElemSize() != 1 /*bytes*/)
      return false;

    if (Size > Ptr.getNumElems()) {
      Parent.FFDiag(SizeExpr, diag::note_constexpr_access_past_end) << AK_Read;
      Size = Ptr.getNumElems();
    }

    if constexpr (std::is_same_v<ResultT, APValue>) {
      QualType CharTy = PtrExpr->getType()->getPointeeType();
      Result = APValue(APValue::UninitArray{}, Size, Size);
      for (uint64_t I = 0; I != Size; ++I) {
        if (std::optional<APValue> ElemVal =
                Ptr.atIndex(I).toRValue(*this, CharTy))
          Result.getArrayInitializedElt(I) = *ElemVal;
        else
          return false;
      }
    } else {
      assert((std::is_same_v<ResultT, std::string>));
      if (Size < Result.max_size())
        Result.resize(Size);
      Result.assign(reinterpret_cast<const char *>(Ptr.getRawAddress()), Size);
    }

    return true;
  });

  if (PtrRes.isInvalid()) {
    C.cleanup();
    Stk.clear();
    return false;
  }

  return true;
}

bool Context::evaluateCharRange(State &Parent, const Expr *SizeExpr,
                                const Expr *PtrExpr, APValue &Result) {
  assert(SizeExpr);
  assert(PtrExpr);

  return evaluateStringRepr(Parent, SizeExpr, PtrExpr, Result);
}

bool Context::evaluateCharRange(State &Parent, const Expr *SizeExpr,
                                const Expr *PtrExpr, std::string &Result) {
  assert(SizeExpr);
  assert(PtrExpr);

  return evaluateStringRepr(Parent, SizeExpr, PtrExpr, Result);
}

bool Context::evaluateStrlen(State &Parent, const Expr *E, uint64_t &Result) {
  assert(Stk.empty());
  Compiler<EvalEmitter> C(*this, *P, Parent, Stk);

  auto PtrRes = C.interpretAsPointer(E, [&](const Pointer &Ptr) {
    const Descriptor *FieldDesc = Ptr.getFieldDesc();
    if (!FieldDesc->isPrimitiveArray())
      return false;

    unsigned N = Ptr.getNumElems();
    if (Ptr.elemSize() == 1) {
      Result = strnlen(reinterpret_cast<const char *>(Ptr.getRawAddress()), N);
      return Result != N;
    }

    PrimType ElemT = FieldDesc->getPrimType();
    Result = 0;
    for (unsigned I = Ptr.getIndex(); I != N; ++I) {
      INT_TYPE_SWITCH(ElemT, {
        auto Elem = Ptr.elem<T>(I);
        if (Elem.isZero())
          return true;
        ++Result;
      });
    }
    // We didn't find a 0 byte.
    return false;
  });

  if (PtrRes.isInvalid()) {
    C.cleanup();
    Stk.clear();
    return false;
  }
  return true;
}

const LangOptions &Context::getLangOpts() const { return Ctx.getLangOpts(); }

static PrimType integralTypeToPrimTypeS(unsigned BitWidth) {
  switch (BitWidth) {
  case 64:
    return PT_Sint64;
  case 32:
    return PT_Sint32;
  case 16:
    return PT_Sint16;
  case 8:
    return PT_Sint8;
  default:
    return PT_IntAPS;
  }
  llvm_unreachable("Unhandled BitWidth");
}

static PrimType integralTypeToPrimTypeU(unsigned BitWidth) {
  switch (BitWidth) {
  case 64:
    return PT_Uint64;
  case 32:
    return PT_Uint32;
  case 16:
    return PT_Uint16;
  case 8:
    return PT_Uint8;
  default:
    return PT_IntAP;
  }
  llvm_unreachable("Unhandled BitWidth");
}

OptPrimType Context::classify(QualType T) const {

  if (const auto *BT = dyn_cast<BuiltinType>(T.getCanonicalType())) {
    auto Kind = BT->getKind();
    if (Kind == BuiltinType::Bool)
      return PT_Bool;
    if (Kind == BuiltinType::NullPtr)
      return PT_Ptr;
    if (Kind == BuiltinType::BoundMember)
      return PT_MemberPtr;

    // Just trying to avoid the ASTContext::getIntWidth call below.
    if (Kind == BuiltinType::Short)
      return integralTypeToPrimTypeS(this->ShortWidth);
    if (Kind == BuiltinType::UShort)
      return integralTypeToPrimTypeU(this->ShortWidth);

    if (Kind == BuiltinType::Int)
      return integralTypeToPrimTypeS(this->IntWidth);
    if (Kind == BuiltinType::UInt)
      return integralTypeToPrimTypeU(this->IntWidth);
    if (Kind == BuiltinType::Long)
      return integralTypeToPrimTypeS(this->LongWidth);
    if (Kind == BuiltinType::ULong)
      return integralTypeToPrimTypeU(this->LongWidth);
    if (Kind == BuiltinType::LongLong)
      return integralTypeToPrimTypeS(this->LongLongWidth);
    if (Kind == BuiltinType::ULongLong)
      return integralTypeToPrimTypeU(this->LongLongWidth);

    if (Kind == BuiltinType::SChar || Kind == BuiltinType::Char_S)
      return integralTypeToPrimTypeS(8);
    if (Kind == BuiltinType::UChar || Kind == BuiltinType::Char_U ||
        Kind == BuiltinType::Char8)
      return integralTypeToPrimTypeU(8);

    if (BT->isSignedInteger())
      return integralTypeToPrimTypeS(Ctx.getIntWidth(T));
    if (BT->isUnsignedInteger())
      return integralTypeToPrimTypeU(Ctx.getIntWidth(T));

    if (BT->isFloatingPoint())
      return PT_Float;
  }

  if (T->isPointerOrReferenceType())
    return PT_Ptr;

  if (T->isMemberPointerType())
    return PT_MemberPtr;

  if (const auto *BT = T->getAs<BitIntType>()) {
    if (BT->isSigned())
      return integralTypeToPrimTypeS(BT->getNumBits());
    return integralTypeToPrimTypeU(BT->getNumBits());
  }

  if (const auto *ET = T->getAs<EnumType>()) {
    const auto *D = ET->getDecl();
    if (!D->isComplete())
      return std::nullopt;
    return classify(D->getIntegerType());
  }

  if (const auto *AT = T->getAs<AtomicType>())
    return classify(AT->getValueType());

  if (const auto *DT = dyn_cast<DecltypeType>(T))
    return classify(DT->getUnderlyingType());

  if (T->isObjCObjectPointerType() || T->isBlockPointerType())
    return PT_Ptr;

  if (T->isFixedPointType())
    return PT_FixedPoint;

  // Vector and complex types get here.
  return std::nullopt;
}

unsigned Context::getCharBit() const {
  return Ctx.getTargetInfo().getCharWidth();
}

/// Simple wrapper around getFloatTypeSemantics() to make code a
/// little shorter.
const llvm::fltSemantics &Context::getFloatSemantics(QualType T) const {
  return Ctx.getFloatTypeSemantics(T);
}

bool Context::Run(State &Parent, const Function *Func) {

  {
    InterpState State(Parent, *P, Stk, *this, Func);
    if (Interpret(State)) {
      assert(Stk.empty());
      return true;
    }
    // State gets destroyed here, so the Stk.clear() below doesn't accidentally
    // remove values the State's destructor might access.
  }

  Stk.clear();
  return false;
}

// TODO: Virtual bases?
const CXXMethodDecl *
Context::getOverridingFunction(const CXXRecordDecl *DynamicDecl,
                               const CXXRecordDecl *StaticDecl,
                               const CXXMethodDecl *InitialFunction) const {
  assert(DynamicDecl);
  assert(StaticDecl);
  assert(InitialFunction);

  const CXXRecordDecl *CurRecord = DynamicDecl;
  const CXXMethodDecl *FoundFunction = InitialFunction;
  for (;;) {
    const CXXMethodDecl *Overrider =
        FoundFunction->getCorrespondingMethodDeclaredInClass(CurRecord, false);
    if (Overrider)
      return Overrider;

    // Common case of only one base class.
    if (CurRecord->getNumBases() == 1) {
      CurRecord = CurRecord->bases_begin()->getType()->getAsCXXRecordDecl();
      continue;
    }

    // Otherwise, go to the base class that will lead to the StaticDecl.
    for (const CXXBaseSpecifier &Spec : CurRecord->bases()) {
      const CXXRecordDecl *Base = Spec.getType()->getAsCXXRecordDecl();
      if (Base == StaticDecl || Base->isDerivedFrom(StaticDecl)) {
        CurRecord = Base;
        break;
      }
    }
  }

  llvm_unreachable(
      "Couldn't find an overriding function in the class hierarchy?");
  return nullptr;
}

const Function *Context::getOrCreateFunction(const FunctionDecl *FuncDecl) {
  assert(FuncDecl);
  FuncDecl = FuncDecl->getMostRecentDecl();

  if (const Function *Func = P->getFunction(FuncDecl))
    return Func;

  // Manually created functions that haven't been assigned proper
  // parameters yet.
  if (!FuncDecl->param_empty() && !FuncDecl->param_begin())
    return nullptr;

  bool IsLambdaStaticInvoker = false;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FuncDecl);
      MD && MD->isLambdaStaticInvoker()) {
    // For a lambda static invoker, we might have to pick a specialized
    // version if the lambda is generic. In that case, the picked function
    // will *NOT* be a static invoker anymore. However, it will still
    // be a non-static member function, this (usually) requiring an
    // instance pointer. We suppress that later in this function.
    IsLambdaStaticInvoker = true;

    const CXXRecordDecl *ClosureClass = MD->getParent();
    assert(ClosureClass->captures_begin() == ClosureClass->captures_end());
    if (ClosureClass->isGenericLambda()) {
      const CXXMethodDecl *LambdaCallOp = ClosureClass->getLambdaCallOperator();
      assert(MD->isFunctionTemplateSpecialization() &&
             "A generic lambda's static-invoker function must be a "
             "template specialization");
      const TemplateArgumentList *TAL = MD->getTemplateSpecializationArgs();
      FunctionTemplateDecl *CallOpTemplate =
          LambdaCallOp->getDescribedFunctionTemplate();
      void *InsertPos = nullptr;
      const FunctionDecl *CorrespondingCallOpSpecialization =
          CallOpTemplate->findSpecialization(TAL->asArray(), InsertPos);
      assert(CorrespondingCallOpSpecialization);
      FuncDecl = CorrespondingCallOpSpecialization;
    }
  }
  // Set up argument indices.
  unsigned ParamOffset = 0;
  SmallVector<PrimType, 8> ParamTypes;
  SmallVector<unsigned, 8> ParamOffsets;
  llvm::DenseMap<unsigned, Function::ParamDescriptor> ParamDescriptors;

  // If the return is not a primitive, a pointer to the storage where the
  // value is initialized in is passed as the first argument. See 'RVO'
  // elsewhere in the code.
  QualType Ty = FuncDecl->getReturnType();
  bool HasRVO = false;
  if (!Ty->isVoidType() && !classify(Ty)) {
    HasRVO = true;
    ParamTypes.push_back(PT_Ptr);
    ParamOffsets.push_back(ParamOffset);
    ParamOffset += align(primSize(PT_Ptr));
  }

  // If the function decl is a member decl, the next parameter is
  // the 'this' pointer. This parameter is pop()ed from the
  // InterpStack when calling the function.
  bool HasThisPointer = false;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FuncDecl)) {
    if (!IsLambdaStaticInvoker) {
      HasThisPointer = MD->isInstance();
      if (MD->isImplicitObjectMemberFunction()) {
        ParamTypes.push_back(PT_Ptr);
        ParamOffsets.push_back(ParamOffset);
        ParamOffset += align(primSize(PT_Ptr));
      }
    }

    if (isLambdaCallOperator(MD)) {
      // The parent record needs to be complete, we need to know about all
      // the lambda captures.
      if (!MD->getParent()->isCompleteDefinition())
        return nullptr;
      llvm::DenseMap<const ValueDecl *, FieldDecl *> LC;
      FieldDecl *LTC;

      MD->getParent()->getCaptureFields(LC, LTC);

      if (MD->isStatic() && !LC.empty()) {
        // Static lambdas cannot have any captures. If this one does,
        // it has already been diagnosed and we can only ignore it.
        return nullptr;
      }
    }
  }

  // Assign descriptors to all parameters.
  // Composite objects are lowered to pointers.
  for (const ParmVarDecl *PD : FuncDecl->parameters()) {
    OptPrimType T = classify(PD->getType());
    PrimType PT = T.value_or(PT_Ptr);
    Descriptor *Desc = P->createDescriptor(PD, PT);
    ParamDescriptors.insert({ParamOffset, {PT, Desc}});
    ParamOffsets.push_back(ParamOffset);
    ParamOffset += align(primSize(PT));
    ParamTypes.push_back(PT);
  }

  // Create a handle over the emitted code.
  assert(!P->getFunction(FuncDecl));
  const Function *Func = P->createFunction(
      FuncDecl, ParamOffset, std::move(ParamTypes), std::move(ParamDescriptors),
      std::move(ParamOffsets), HasThisPointer, HasRVO, IsLambdaStaticInvoker);
  return Func;
}

const Function *Context::getOrCreateObjCBlock(const BlockExpr *E) {
  const BlockDecl *BD = E->getBlockDecl();
  // Set up argument indices.
  unsigned ParamOffset = 0;
  SmallVector<PrimType, 8> ParamTypes;
  SmallVector<unsigned, 8> ParamOffsets;
  llvm::DenseMap<unsigned, Function::ParamDescriptor> ParamDescriptors;

  // Assign descriptors to all parameters.
  // Composite objects are lowered to pointers.
  for (const ParmVarDecl *PD : BD->parameters()) {
    OptPrimType T = classify(PD->getType());
    PrimType PT = T.value_or(PT_Ptr);
    Descriptor *Desc = P->createDescriptor(PD, PT);
    ParamDescriptors.insert({ParamOffset, {PT, Desc}});
    ParamOffsets.push_back(ParamOffset);
    ParamOffset += align(primSize(PT));
    ParamTypes.push_back(PT);
  }

  if (BD->hasCaptures())
    return nullptr;

  // Create a handle over the emitted code.
  Function *Func =
      P->createFunction(E, ParamOffset, std::move(ParamTypes),
                        std::move(ParamDescriptors), std::move(ParamOffsets),
                        /*HasThisPointer=*/false, /*HasRVO=*/false,
                        /*IsLambdaStaticInvoker=*/false);

  assert(Func);
  Func->setDefined(true);
  // We don't compile the BlockDecl code at all right now.
  Func->setIsFullyCompiled(true);
  return Func;
}

unsigned Context::collectBaseOffset(const RecordDecl *BaseDecl,
                                    const RecordDecl *DerivedDecl) const {
  assert(BaseDecl);
  assert(DerivedDecl);
  const auto *FinalDecl = cast<CXXRecordDecl>(BaseDecl);
  const RecordDecl *CurDecl = DerivedDecl;
  const Record *CurRecord = P->getOrCreateRecord(CurDecl);
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

const Record *Context::getRecord(const RecordDecl *D) const {
  return P->getOrCreateRecord(D);
}

bool Context::isUnevaluatedBuiltin(unsigned ID) {
  return ID == Builtin::BI__builtin_classify_type ||
         ID == Builtin::BI__builtin_os_log_format_buffer_size ||
         ID == Builtin::BI__builtin_constant_p || ID == Builtin::BI__noop;
}
