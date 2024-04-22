//===- CtxInstrProfiling.cpp - contextual instrumented PGO ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CtxInstrProfiling.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_dense_map.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_thread_safety.h"

#include <assert.h>

using namespace __ctx_profile;

// FIXME(mtrofin): use malloc / mmap instead of sanitizer common APIs to reduce
// the dependency on the latter.
Arena *Arena::allocateNewArena(size_t Size, Arena *Prev) {
  assert(!Prev || Prev->Next == nullptr);
  Arena *NewArena =
      new (__sanitizer::InternalAlloc(Size + sizeof(Arena))) Arena(Size);
  if (Prev)
    Prev->Next = NewArena;
  return NewArena;
}

void Arena::freeArenaList(Arena *&A) {
  assert(A);
  for (auto *I = A; I != nullptr;) {
    auto *Current = I;
    I = I->Next;
    __sanitizer::InternalFree(Current);
  }
  A = nullptr;
}
