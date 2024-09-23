//===--- InterpFrame.cpp - Call Frame implementation for the VM -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InterpFrame.h"
#include "Boolean.h"
#include "Floating.h"
#include "Function.h"
#include "InterpStack.h"
#include "InterpState.h"
#include "MemberPointer.h"
#include "Pointer.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"

using namespace clang;
using namespace clang::interp;

InterpFrame::InterpFrame(InterpState &S, const Function *Func,
                         InterpFrame *Caller, CodePtr RetPC, unsigned ArgSize)
    : Caller(Caller), S(S), Depth(Caller ? Caller->Depth + 1 : 0), Func(Func),
      RetPC(RetPC), ArgSize(ArgSize), Args(static_cast<char *>(S.Stk.top())),
      FrameOffset(S.Stk.size()) {
  if (!Func)
    return;

  unsigned FrameSize = Func->getFrameSize();
  if (FrameSize == 0)
    return;

  Locals = std::make_unique<char[]>(FrameSize);
  for (auto &Scope : Func->scopes()) {
    for (auto &Local : Scope.locals()) {
      new (localBlock(Local.Offset)) Block(S.Ctx.getEvalID(), Local.Desc);
      // Note that we are NOT calling invokeCtor() here, since that is done
      // via the InitScope op.
      new (localInlineDesc(Local.Offset)) InlineDescriptor(Local.Desc);
    }
  }
}

InterpFrame::InterpFrame(InterpState &S, const Function *Func, CodePtr RetPC,
                         unsigned VarArgSize)
    : InterpFrame(S, Func, S.Current, RetPC, Func->getArgSize() + VarArgSize) {
  // As per our calling convention, the this pointer is
  // part of the ArgSize.
  // If the function has RVO, the RVO pointer is first.
  // If the fuction has a This pointer, that one is next.
  // Then follow the actual arguments (but those are handled
  // in getParamPointer()).
  if (Func->hasRVO())
    RVOPtr = stackRef<Pointer>(0);

  if (Func->hasThisPointer()) {
    if (Func->hasRVO())
      This = stackRef<Pointer>(sizeof(Pointer));
    else
      This = stackRef<Pointer>(0);
  }
}

InterpFrame::~InterpFrame() {
  for (auto &Param : Params)
    S.deallocate(reinterpret_cast<Block *>(Param.second.get()));

  // When destroying the InterpFrame, call the Dtor for all block
  // that haven't been destroyed via a destroy() op yet.
  // This happens when the execution is interruped midway-through.
  if (Func) {
    for (auto &Scope : Func->scopes()) {
      for (auto &Local : Scope.locals()) {
        S.deallocate(localBlock(Local.Offset));
      }
    }
  }
}

void InterpFrame::initScope(unsigned Idx) {
  if (!Func)
    return;
  for (auto &Local : Func->getScope(Idx).locals()) {
    localBlock(Local.Offset)->invokeCtor();
  }
}

void InterpFrame::destroy(unsigned Idx) {
  for (auto &Local : Func->getScope(Idx).locals()) {
    S.deallocate(localBlock(Local.Offset));
  }
}

template <typename T>
static void print(llvm::raw_ostream &OS, const T &V, ASTContext &ASTCtx,
                  QualType Ty) {
  V.toAPValue(ASTCtx).printPretty(OS, ASTCtx, Ty);
}

static bool shouldSkipInBacktrace(const Function *F) {
  if (F->isBuiltin())
    return true;
  if (F->isLambdaStaticInvoker())
    return true;

  const FunctionDecl *FD = F->getDecl();
  if (FD->getDeclName().getCXXOverloadedOperator() == OO_New ||
      FD->getDeclName().getCXXOverloadedOperator() == OO_Array_New)
    return true;
  return false;
}

void InterpFrame::describe(llvm::raw_ostream &OS) const {
  // We create frames for builtin functions as well, but we can't reliably
  // diagnose them. The 'in call to' diagnostics for them add no value to the
  // user _and_ it doesn't generally work since the argument types don't always
  // match the function prototype. Just ignore them.
  // Similarly, for lambda static invokers, we would just print __invoke().
  if (const auto *F = getFunction(); F && shouldSkipInBacktrace(F))
    return;

  const Expr *CallExpr = Caller->getExpr(getRetPC());
  const FunctionDecl *F = getCallee();
  bool IsMemberCall = isa<CXXMethodDecl>(F) && !isa<CXXConstructorDecl>(F) &&
                      cast<CXXMethodDecl>(F)->isImplicitObjectMemberFunction();
  if (Func->hasThisPointer() && IsMemberCall) {
    if (const auto *MCE = dyn_cast_if_present<CXXMemberCallExpr>(CallExpr)) {
      const Expr *Object = MCE->getImplicitObjectArgument();
      Object->printPretty(OS, /*Helper=*/nullptr,
                          S.getASTContext().getPrintingPolicy(),
                          /*Indentation=*/0);
      if (Object->getType()->isPointerType())
        OS << "->";
      else
        OS << ".";
    } else if (const auto *OCE =
                   dyn_cast_if_present<CXXOperatorCallExpr>(CallExpr)) {
      OCE->getArg(0)->printPretty(OS, /*Helper=*/nullptr,
                                  S.getASTContext().getPrintingPolicy(),
                                  /*Indentation=*/0);
      OS << ".";
    } else if (const auto *M = dyn_cast<CXXMethodDecl>(F)) {
      print(OS, This, S.getASTContext(),
            S.getASTContext().getLValueReferenceType(
                S.getASTContext().getRecordType(M->getParent())));
      OS << ".";
    }
  }

  F->getNameForDiagnostic(OS, S.getASTContext().getPrintingPolicy(),
                          /*Qualified=*/false);
  OS << '(';
  unsigned Off = 0;

  Off += Func->hasRVO() ? primSize(PT_Ptr) : 0;
  Off += Func->hasThisPointer() ? primSize(PT_Ptr) : 0;

  for (unsigned I = 0, N = F->getNumParams(); I < N; ++I) {
    QualType Ty = F->getParamDecl(I)->getType();

    PrimType PrimTy = S.Ctx.classify(Ty).value_or(PT_Ptr);

    TYPE_SWITCH(PrimTy, print(OS, stackRef<T>(Off), S.getASTContext(), Ty));
    Off += align(primSize(PrimTy));
    if (I + 1 != N)
      OS << ", ";
  }
  OS << ")";
}

Frame *InterpFrame::getCaller() const {
  if (Caller->Caller)
    return Caller;
  return S.getSplitFrame();
}

SourceRange InterpFrame::getCallRange() const {
  if (!Caller->Func) {
    if (SourceRange NullRange = S.getRange(nullptr, {}); NullRange.isValid())
      return NullRange;
    return S.EvalLocation;
  }
  return S.getRange(Caller->Func, RetPC - sizeof(uintptr_t));
}

const FunctionDecl *InterpFrame::getCallee() const {
  if (!Func)
    return nullptr;
  return Func->getDecl();
}

Pointer InterpFrame::getLocalPointer(unsigned Offset) const {
  assert(Offset < Func->getFrameSize() && "Invalid local offset.");
  return Pointer(localBlock(Offset));
}

Pointer InterpFrame::getParamPointer(unsigned Off) {
  // Return the block if it was created previously.
  if (auto Pt = Params.find(Off); Pt != Params.end())
    return Pointer(reinterpret_cast<Block *>(Pt->second.get()));

  // Allocate memory to store the parameter and the block metadata.
  const auto &Desc = Func->getParamDescriptor(Off);
  size_t BlockSize = sizeof(Block) + Desc.second->getAllocSize();
  auto Memory = std::make_unique<char[]>(BlockSize);
  auto *B = new (Memory.get()) Block(S.Ctx.getEvalID(), Desc.second);
  B->invokeCtor();

  // Copy the initial value.
  TYPE_SWITCH(Desc.first, new (B->data()) T(stackRef<T>(Off)));

  // Record the param.
  Params.insert({Off, std::move(Memory)});
  return Pointer(B);
}

static bool funcHasUsableBody(const Function *F) {
  assert(F);

  if (F->isConstructor() || F->isDestructor())
    return true;

  return !F->getDecl()->isImplicit();
}

SourceInfo InterpFrame::getSource(CodePtr PC) const {
  // Implicitly created functions don't have any code we could point at,
  // so return the call site.
  if (Func && !funcHasUsableBody(Func) && Caller)
    return Caller->getSource(RetPC);

  return S.getSource(Func, PC);
}

const Expr *InterpFrame::getExpr(CodePtr PC) const {
  if (Func && !funcHasUsableBody(Func) && Caller)
    return Caller->getExpr(PC);

  return S.getExpr(Func, PC);
}

SourceLocation InterpFrame::getLocation(CodePtr PC) const {
  if (Func && !funcHasUsableBody(Func) && Caller)
    return Caller->getLocation(RetPC);

  return S.getLocation(Func, PC);
}

SourceRange InterpFrame::getRange(CodePtr PC) const {
  if (Func && !funcHasUsableBody(Func) && Caller)
    return Caller->getRange(RetPC);

  return S.getRange(Func, PC);
}
