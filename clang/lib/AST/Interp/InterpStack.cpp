//===--- InterpStack.cpp - Stack implementation for the VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdlib>
#include "InterpStack.h"

using namespace clang;
using namespace clang::interp;

InterpStack::~InterpStack() {
  clear();
}

void InterpStack::clear() {
  if (Chunk && Chunk->Next)
    free(Chunk->Next);
  if (Chunk)
    free(Chunk);
  Chunk = nullptr;
  StackSize = 0;
}

void *InterpStack::grow(size_t Size) {
  assert(Size < ChunkSize - sizeof(StackChunk) && "Object too large");

  if (!Chunk || sizeof(StackChunk) + Chunk->size() + Size > ChunkSize) {
    if (Chunk && Chunk->Next) {
      Chunk = Chunk->Next;
    } else {
      StackChunk *Next = new (malloc(ChunkSize)) StackChunk(Chunk);
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
      free(Chunk->Next);
      Chunk->Next = nullptr;
    }
    Chunk->End = Chunk->start();
    Chunk = Chunk->Prev;
    assert(Chunk && "Offset too large");
  }

  Chunk->End -= Size;
  StackSize -= Size;
}
