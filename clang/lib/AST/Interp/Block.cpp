//===--- Block.cpp - Allocated blocks for the interpreter -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the classes describing allocated blocks.
//
//===----------------------------------------------------------------------===//

#include "Block.h"
#include "Pointer.h"

using namespace clang;
using namespace clang::interp;



void Block::addPointer(Pointer *P) {
  if (IsStatic)
    return;
  if (Pointers)
    Pointers->Prev = P;
  P->Next = Pointers;
  P->Prev = nullptr;
  Pointers = P;
}

void Block::removePointer(Pointer *P) {
  if (IsStatic)
    return;
  if (Pointers == P)
    Pointers = P->Next;
  if (P->Prev)
    P->Prev->Next = P->Next;
  if (P->Next)
    P->Next->Prev = P->Prev;
}

void Block::cleanup() {
  if (Pointers == nullptr && IsDead)
    (reinterpret_cast<DeadBlock *>(this + 1) - 1)->free();
}

void Block::movePointer(Pointer *From, Pointer *To) {
  if (IsStatic)
    return;
  To->Prev = From->Prev;
  if (To->Prev)
    To->Prev->Next = To;
  To->Next = From->Next;
  if (To->Next)
    To->Next->Prev = To;
  if (Pointers == From)
    Pointers = To;

  From->Prev = nullptr;
  From->Next = nullptr;
}

DeadBlock::DeadBlock(DeadBlock *&Root, Block *Blk)
    : Root(Root), B(Blk->Desc, Blk->IsStatic, Blk->IsExtern, /*isDead=*/true) {
  // Add the block to the chain of dead blocks.
  if (Root)
    Root->Prev = this;

  Next = Root;
  Prev = nullptr;
  Root = this;

  // Transfer pointers.
  B.Pointers = Blk->Pointers;
  for (Pointer *P = Blk->Pointers; P; P = P->Next)
    P->Pointee = &B;
}

void DeadBlock::free() {
  if (Prev)
    Prev->Next = Next;
  if (Next)
    Next->Prev = Prev;
  if (Root == this)
    Root = Next;
  ::free(this);
}
