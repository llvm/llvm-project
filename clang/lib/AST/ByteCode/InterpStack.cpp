//===--- InterpStack.cpp - Stack implementation for the VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InterpStack.h"
#include "Boolean.h"
#include "FixedPoint.h"
#include "Floating.h"
#include "Integral.h"
#include "MemberPointer.h"
#include "Pointer.h"
#include <cassert>
#include <cstdlib>

using namespace clang;
using namespace clang::interp;

InterpStack::~InterpStack() {
  if (Chunk && Chunk->Next)
    std::free(Chunk->Next);
  if (Chunk)
    std::free(Chunk);
  Chunk = nullptr;
  StackSize = 0;
  ItemTypes.clear();
}

// We keep the last chunk around to reuse.
void InterpStack::clear() {
  for (PrimType Item : llvm::reverse(ItemTypes)) {
    TYPE_SWITCH(Item, { this->discard<T>(); });
  }
  assert(ItemTypes.empty());
  assert(empty());
}

void InterpStack::clearTo(size_t NewSize) {
  if (NewSize == 0)
    return clear();
  if (NewSize == size())
    return;

  assert(NewSize <= size());
  for (PrimType Item : llvm::reverse(ItemTypes)) {
    TYPE_SWITCH(Item, { this->discard<T>(); });

    if (size() == NewSize)
      break;
  }

  // Note: discard() above already removed the types from ItemTypes.
  assert(size() == NewSize);
}

void *InterpStack::grow(size_t Size) {
  assert(Size < ChunkSize - sizeof(StackChunk) && "Object too large");

  if (!Chunk || sizeof(StackChunk) + Chunk->size() + Size > ChunkSize) {
    if (Chunk && Chunk->Next) {
      Chunk = Chunk->Next;
    } else {
      StackChunk *Next = new (std::malloc(ChunkSize)) StackChunk(Chunk);
      if (Chunk)
        Chunk->Next = Next;
      Chunk = Next;
    }
  }

  auto *Object = reinterpret_cast<void *>(Chunk->End);
  Chunk->End += Size;
  StackSize += Size;
  return Object;
}

void *InterpStack::peekData(size_t Size) const {
  assert(Chunk && "Stack is empty!");

  StackChunk *Ptr = Chunk;
  while (Size > Ptr->size()) {
    Size -= Ptr->size();
    Ptr = Ptr->Prev;
    assert(Ptr && "Offset too large");
  }

  return reinterpret_cast<void *>(Ptr->End - Size);
}

void InterpStack::shrink(size_t Size) {
  assert(Chunk && "Chunk is empty!");

  while (Size > Chunk->size()) {
    Size -= Chunk->size();
    if (Chunk->Next) {
      std::free(Chunk->Next);
      Chunk->Next = nullptr;
    }
    Chunk->End = Chunk->start();
    Chunk = Chunk->Prev;
    assert(Chunk && "Offset too large");
  }

  Chunk->End -= Size;
  StackSize -= Size;
}

void InterpStack::dump() const {
  llvm::errs() << "Items: " << ItemTypes.size() << ". Size: " << size() << '\n';
  if (ItemTypes.empty())
    return;

  size_t Index = 0;
  size_t Offset = 0;

  // The type of the item on the top of the stack is inserted to the back
  // of the vector, so the iteration has to happen backwards.
  for (PrimType Item : llvm::reverse(ItemTypes)) {
    Offset += align(primSize(Item));

    llvm::errs() << Index << '/' << Offset << ": ";
    TYPE_SWITCH(Item, {
      const T &V = peek<T>(Offset);
      llvm::errs() << V;
    });
    llvm::errs() << '\n';

    ++Index;
  }
}
