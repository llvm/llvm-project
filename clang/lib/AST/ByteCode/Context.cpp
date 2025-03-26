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

Context::Context(ASTContext &Ctx) : Ctx(Ctx), P(new Program(*this)) {}

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

  // And run it.
  if (!Run(Parent, Func))
    return false;

  return Func->isConstexpr();
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

const LangOptions &Context::getLangOpts() const { return Ctx.getLangOpts(); }

std::optional<PrimType> Context::classify(QualType T) const {
  if (T->isBooleanType())
    return PT_Bool;

  // We map these to primitive arrays.
  if (T->isAnyComplexType() || T->isVectorType())
    return std::nullopt;

  if (T->isSignedIntegerOrEnumerationType()) {
    switch (Ctx.getIntWidth(T)) {
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
  }

  if (T->isUnsignedIntegerOrEnumerationType()) {
    switch (Ctx.getIntWidth(T)) {
    case 64:
      return PT_Uint64;
    case 32:
      return PT_Uint32;
    case 16:
      return PT_Uint16;
    case 8:
      return PT_Uint8;
    case 1:
      // Might happen for enum types.
      return PT_Bool;
    default:
      return PT_IntAP;
    }
  }

  if (T->isNullPtrType())
    return PT_Ptr;

  if (T->isFloatingType())
    return PT_Float;

  if (T->isSpecificBuiltinType(BuiltinType::BoundMember) ||
      T->isMemberPointerType())
    return PT_MemberPtr;

  if (T->isFunctionPointerType() || T->isFunctionReferenceType() ||
      T->isFunctionType() || T->isBlockPointerType())
    return PT_FnPtr;

  if (T->isPointerOrReferenceType() || T->isObjCObjectPointerType())
    return PT_Ptr;

  if (const auto *AT = T->getAs<AtomicType>())
    return classify(AT->getValueType());

  if (const auto *DT = dyn_cast<DecltypeType>(T))
    return classify(DT->getUnderlyingType());

  if (T->isFixedPointType())
    return PT_FixedPoint;

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
    std::optional<PrimType> T = classify(PD->getType());
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
    std::optional<PrimType> T = classify(PD->getType());
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
