//===--- InterpState.cpp - Interpreter for the constexpr VM -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InterpState.h"
#include <limits>
#include "Function.h"
#include "InterpFrame.h"
#include "InterpStack.h"
#include "Opcode.h"
#include "PrimType.h"
#include "Program.h"
#include "State.h"

using namespace clang;
using namespace clang::interp;

using APSInt = llvm::APSInt;

InterpState::InterpState(State &Parent, Program &P, InterpStack &Stk,
                         Context &Ctx, SourceMapper *M)
    : Parent(Parent), M(M), P(P), Stk(Stk), Ctx(Ctx), Current(nullptr),
      CallStackDepth(Parent.getCallStackDepth() + 1) {}

InterpState::~InterpState() {
  while (Current) {
    InterpFrame *Next = Current->Caller;
    delete Current;
    Current = Next;
  }

  while (DeadBlocks) {
    DeadBlock *Next = DeadBlocks->Next;
    free(DeadBlocks);
    DeadBlocks = Next;
  }
}

Frame *InterpState::getCurrentFrame() {
  if (Current && Current->Caller) {
    return Current;
  } else {
    return Parent.getCurrentFrame();
  }
}

bool InterpState::reportOverflow(const Expr *E, const llvm::APSInt &Value) {
  QualType Type = E->getType();
  CCEDiag(E, diag::note_constexpr_overflow) << Value << Type;
  return noteUndefinedBehavior();
}

void InterpState::deallocate(Block *B) {
  Descriptor *Desc = B->getDescriptor();
  if (B->hasPointers()) {
    size_t Size = B->getSize();

    // Allocate a new block, transferring over pointers.
    char *Memory = reinterpret_cast<char *>(malloc(sizeof(DeadBlock) + Size));
    auto *D = new (Memory) DeadBlock(DeadBlocks, B);

    // Move data from one block to another.
    if (Desc->MoveFn)
      Desc->MoveFn(B, B->data(), D->data(), Desc);
  } else {
    // Free storage, if necessary.
    if (Desc->DtorFn)
      Desc->DtorFn(B, B->data(), Desc);
  }
}
