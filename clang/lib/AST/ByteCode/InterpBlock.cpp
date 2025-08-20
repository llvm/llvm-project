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

#include "InterpBlock.h"
#include "Pointer.h"

using namespace clang;
using namespace clang::interp;

void Block::addPointer(Pointer *P) {
  assert(P);

#ifndef NDEBUG
  assert(!hasPointer(P));
#endif
  if (Pointers)
    Pointers->PointeeStorage.BS.Prev = P;
  P->PointeeStorage.BS.Next = Pointers;
  P->PointeeStorage.BS.Prev = nullptr;
  Pointers = P;
#ifndef NDEBUG
  assert(hasPointer(P));
#endif
}

void Block::removePointer(Pointer *P) {
  assert(P->isBlockPointer());
  assert(P);

#ifndef NDEBUG
  assert(hasPointer(P));
#endif

  BlockPointer &BP = P->PointeeStorage.BS;

  if (Pointers == P)
    Pointers = BP.Next;

  if (BP.Prev)
    BP.Prev->PointeeStorage.BS.Next = BP.Next;
  if (BP.Next)
    BP.Next->PointeeStorage.BS.Prev = BP.Prev;
  P->PointeeStorage.BS.Pointee = nullptr;
#ifndef NDEBUG
  assert(!hasPointer(P));
#endif
}

void Block::cleanup() {
  if (Pointers == nullptr && !isDynamic() && isDead())
    (reinterpret_cast<DeadBlock *>(this + 1) - 1)->free();
}

void Block::replacePointer(Pointer *Old, Pointer *New) {
  assert(Old);
  assert(Old->isBlockPointer());
  assert(New);
  assert(New->isBlockPointer());
  assert(Old != New);
#ifndef NDEBUG
  assert(hasPointer(Old));
#endif

  BlockPointer &OldBP = Old->PointeeStorage.BS;
  BlockPointer &NewBP = New->PointeeStorage.BS;

  if (OldBP.Prev)
    OldBP.Prev->PointeeStorage.BS.Next = New;
  if (OldBP.Next)
    OldBP.Next->PointeeStorage.BS.Prev = New;
  NewBP.Prev = OldBP.Prev;
  NewBP.Next = OldBP.Next;
  if (Pointers == Old)
    Pointers = New;

  OldBP.Pointee = nullptr;
  NewBP.Pointee = this;
#ifndef NDEBUG
  assert(!hasPointer(Old));
  assert(hasPointer(New));
#endif
}

#ifndef NDEBUG
bool Block::hasPointer(const Pointer *P) const {
  for (const Pointer *C = Pointers; C; C = C->asBlockPointer().Next) {
    if (C == P)
      return true;
  }
  return false;
}
#endif

DeadBlock::DeadBlock(DeadBlock *&Root, Block *Blk)
    : Root(Root), B(~0u, Blk->Desc, Blk->isExtern(), Blk->IsStatic,
                    Blk->isWeak(), Blk->isDummy(), /*IsDead=*/true) {
  // Add the block to the chain of dead blocks.
  if (Root)
    Root->Prev = this;

  Next = Root;
  Prev = nullptr;
  Root = this;

  B.DynAllocId = Blk->DynAllocId;

  // Transfer pointers.
  B.Pointers = Blk->Pointers;
  for (Pointer *P = Blk->Pointers; P; P = P->asBlockPointer().Next)
    P->PointeeStorage.BS.Pointee = &B;
  Blk->Pointers = nullptr;
}

void DeadBlock::free() {
  assert(!B.isInitialized());

  if (Prev)
    Prev->Next = Next;
  if (Next)
    Next->Prev = Prev;
  if (Root == this)
    Root = Next;
  std::free(this);
}
