//===--- InterpFrame.cpp - Call Frame implementation for the VM -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InterpFrame.h"
#include "Function.h"
#include "Interp.h"
#include "InterpStack.h"
#include "Pointer.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace clang::interp;

InterpFrame::InterpFrame(InterpState &S, const Function *Func,
                         InterpFrame *Caller, CodePtr RetPC, Pointer &&This)
    : Caller(Caller), S(S), Func(Func), This(std::move(This)), RetPC(RetPC),
      ArgSize(Func ? Func->getArgSize() : 0),
      Args(static_cast<char *>(S.Stk.top())), FrameOffset(S.Stk.size()) {
  if (Func) {
    if (unsigned FrameSize = Func->getFrameSize()) {
      Locals = std::make_unique<char[]>(FrameSize);
      for (auto &Scope : Func->scopes()) {
        for (auto &Local : Scope.locals()) {
          Block *B = new (localBlock(Local.Offset)) Block(Local.Desc);
          B->invokeCtor();
        }
      }
    }
  }
}

InterpFrame::InterpFrame(InterpState &S, const Function *Func, CodePtr RetPC)
    : Caller(S.Current), S(S), Func(Func), RetPC(RetPC),
      ArgSize(Func ? Func->getArgSize() : 0),
      Args(static_cast<char *>(S.Stk.top())), FrameOffset(S.Stk.size()) {
  assert(Func);

  // As per our calling convention, the this pointer is
  // part of the ArgSize.
  // If the function has RVO, the RVO pointer is first.
  // If the fuction has a This pointer, that one is next.
  // Then follow the actual arguments (but those are handled
  // in getParamPointer()).
  if (Func->hasThisPointer()) {
    if (Func->hasRVO())
      This = stackRef<Pointer>(sizeof(Pointer));
    else
      This = stackRef<Pointer>(0);
  }

  if (unsigned FrameSize = Func->getFrameSize()) {
    Locals = std::make_unique<char[]>(FrameSize);
    for (auto &Scope : Func->scopes()) {
      for (auto &Local : Scope.locals()) {
        Block *B = new (localBlock(Local.Offset)) Block(Local.Desc);
        B->invokeCtor();
      }
    }
  }
}

InterpFrame::~InterpFrame() {
  if (Func && Func->isConstructor() && This.isBaseClass())
    This.initialize();
  for (auto &Param : Params)
    S.deallocate(reinterpret_cast<Block *>(Param.second.get()));
}

void InterpFrame::destroy(unsigned Idx) {
  for (auto &Local : Func->getScope(Idx).locals()) {
    S.deallocate(reinterpret_cast<Block *>(localBlock(Local.Offset)));
  }
}

void InterpFrame::popArgs() {
  for (PrimType Ty : Func->args_reverse())
    TYPE_SWITCH(Ty, S.Stk.discard<T>());
}

template <typename T>
static void print(llvm::raw_ostream &OS, const T &V, ASTContext &, QualType) {
  OS << V;
}

template <>
void print(llvm::raw_ostream &OS, const Pointer &P, ASTContext &Ctx,
           QualType Ty) {
  if (P.isZero()) {
    OS << "nullptr";
    return;
  }

  auto printDesc = [&OS, &Ctx](Descriptor *Desc) {
    if (auto *D = Desc->asDecl()) {
      // Subfields or named values.
      if (auto *VD = dyn_cast<ValueDecl>(D)) {
        OS << *VD;
        return;
      }
      // Base classes.
      if (isa<RecordDecl>(D)) {
        return;
      }
    }
    // Temporary expression.
    if (auto *E = Desc->asExpr()) {
      E->printPretty(OS, nullptr, Ctx.getPrintingPolicy());
      return;
    }
    llvm_unreachable("Invalid descriptor type");
  };

  if (!Ty->isReferenceType())
    OS << "&";
  llvm::SmallVector<Pointer, 2> Levels;
  for (Pointer F = P; !F.isRoot(); ) {
    Levels.push_back(F);
    F = F.isArrayElement() ? F.getArray().expand() : F.getBase();
  }

  printDesc(P.getDeclDesc());
  for (auto It = Levels.rbegin(); It != Levels.rend(); ++It) {
    if (It->inArray()) {
      OS << "[" << It->expand().getIndex() << "]";
      continue;
    }
    if (auto Index = It->getIndex()) {
      OS << " + " << Index;
      continue;
    }
    OS << ".";
    printDesc(It->getFieldDesc());
  }
}

void InterpFrame::describe(llvm::raw_ostream &OS) {
  const FunctionDecl *F = getCallee();
  auto *M = dyn_cast<CXXMethodDecl>(F);
  if (M && M->isInstance() && !isa<CXXConstructorDecl>(F)) {
    print(OS, This, S.getCtx(), S.getCtx().getRecordType(M->getParent()));
    OS << "->";
  }
  OS << *F << "(";
  unsigned Off = Func->hasRVO() ? primSize(PT_Ptr) : 0;
  for (unsigned I = 0, N = F->getNumParams(); I < N; ++I) {
    QualType Ty = F->getParamDecl(I)->getType();

    PrimType PrimTy;
    if (llvm::Optional<PrimType> T = S.Ctx.classify(Ty)) {
      PrimTy = *T;
    } else {
      PrimTy = PT_Ptr;
    }

    TYPE_SWITCH(PrimTy, print(OS, stackRef<T>(Off), S.getCtx(), Ty));
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

SourceLocation InterpFrame::getCallLocation() const {
  if (!Caller->Func)
    return S.getLocation(nullptr, {});
  return S.getLocation(Caller->Func, RetPC - sizeof(uintptr_t));
}

const FunctionDecl *InterpFrame::getCallee() const {
  return Func->getDecl();
}

Pointer InterpFrame::getLocalPointer(unsigned Offset) {
  assert(Offset < Func->getFrameSize() && "Invalid local offset.");
  return Pointer(
      reinterpret_cast<Block *>(Locals.get() + Offset - sizeof(Block)));
}

Pointer InterpFrame::getParamPointer(unsigned Off) {
  // Return the block if it was created previously.
  auto Pt = Params.find(Off);
  if (Pt != Params.end()) {
    return Pointer(reinterpret_cast<Block *>(Pt->second.get()));
  }

  // Allocate memory to store the parameter and the block metadata.
  const auto &Desc = Func->getParamDescriptor(Off);
  size_t BlockSize = sizeof(Block) + Desc.second->getAllocSize();
  auto Memory = std::make_unique<char[]>(BlockSize);
  auto *B = new (Memory.get()) Block(Desc.second);

  // Copy the initial value.
  TYPE_SWITCH(Desc.first, new (B->data()) T(stackRef<T>(Off)));

  // Record the param.
  Params.insert({Off, std::move(Memory)});
  return Pointer(B);
}

SourceInfo InterpFrame::getSource(CodePtr PC) const {
  return S.getSource(Func, PC);
}

const Expr *InterpFrame::getExpr(CodePtr PC) const {
  return S.getExpr(Func, PC);
}

SourceLocation InterpFrame::getLocation(CodePtr PC) const {
  return S.getLocation(Func, PC);
}

