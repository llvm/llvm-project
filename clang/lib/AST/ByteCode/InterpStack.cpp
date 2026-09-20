//===--- InterpStack.cpp - Stack implementation for the VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InterpStack.h"
#include "Boolean.h"
#include "Char.h"
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

#if __has_cpp_attribute(no_unique_address)
  TYPE_SWITCH(PrimType(), {
    using Frame = StackFrame<T>;
    static_assert(offsetof(Frame, type) == sizeof(Frame) - 1);
    static_assert(sizeof(void *) != 8 || sizeof(Frame) == sizeof(T) ||
                  sizeof(T) < sizeof(void *));
  });
#endif
}

// We keep the last chunk around to reuse.
void InterpStack::clear() {
  while (!empty()) {
    TYPE_SWITCH(getNextObjectType(), { this->discard<T>(); });
  }
}

void InterpStack::clearTo(size_t NewSize) {
  if (NewSize == 0)
    return clear();
  if (NewSize == size())
    return;

  assert(NewSize <= size());
  while (size() != NewSize)
    TYPE_SWITCH(getNextObjectType(), { this->discard<T>(); });

  // Note: discard() above already removed the types from ItemTypes.
  assert(size() == NewSize);
}

void *InterpStack::peekData(size_t Size) const {
  assert(Chunk && "Stack is empty!");

  if (LLVM_LIKELY(Size <= Chunk->size()))
    return reinterpret_cast<void *>(Chunk->start() + Chunk->Size - Size);

  StackChunk *Ptr = Chunk;
  while (Size > Ptr->size()) {
    Size -= Ptr->size();
    Ptr = Ptr->Prev;
    assert(Ptr && "Offset too large");
  }

  return reinterpret_cast<void *>(Ptr->start() + Ptr->Size - Size);
}

void InterpStack::shrink(size_t Size) {
  assert(Chunk && "Chunk is empty!");

  // Likely case is that we simply remove something from the current chunk.
  if (LLVM_LIKELY(Size <= Chunk->size())) {
    Chunk->Size -= Size;
    StackSize -= Size;
    return;
  }

  while (Size > Chunk->size()) {
    Size -= Chunk->size();
    if (Chunk->Next) {
      std::free(Chunk->Next);
      Chunk->Next = nullptr;
    }
    Chunk->Size = 0;
    Chunk = Chunk->Prev;
    assert(Chunk && "Offset too large");
  }

  Chunk->Size -= Size;
  StackSize -= Size;
}

void InterpStack::dump() const {
  size_t Index = 0;
  size_t Offset = 0;

  // The type of the item on the top of the stack is inserted to the back
  // of the vector, so the iteration has to happen backwards.
  while (Offset != size()) {
    PrimType Item = *static_cast<PrimType *>(peekData(Offset + 1));
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

void InterpStack::discardSlow() {
  assert(!empty());

  TYPE_SWITCH(getNextObjectType(), { discard<T>(); });
}
