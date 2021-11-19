//===-- sanitizer_stack_store.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A fast memory allocator that does not support free() nor realloc().
// All allocations are forever.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_STACK_STORE_H
#define SANITIZER_STACK_STORE_H

#include "sanitizer_atomic.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_mutex.h"
#include "sanitizer_stacktrace.h"

namespace __sanitizer {

class StackStore {
 public:
  constexpr StackStore() = default;

  using Id = uptr;

  Id store(const StackTrace &trace);
  StackTrace load(Id id);
  uptr allocated() const { return atomic_load_relaxed(&mapped_size); }

  void TestOnlyUnmap();

 private:
  uptr *alloc(uptr count = 1);
  uptr *tryAlloc(uptr count);
  uptr *refillAndAlloc(uptr count);
  mutable StaticSpinMutex mtx = {};  // Protects alloc of new blocks.
  atomic_uintptr_t region_pos = {};  // Region allocator for Node's.
  atomic_uintptr_t region_end = {};
  atomic_uintptr_t mapped_size = {};

  struct BlockInfo {
    const BlockInfo *next;
    uptr ptr;
    uptr size;
  };
  const BlockInfo *curr = nullptr;
};

}  // namespace __sanitizer

#endif  // SANITIZER_STACK_STORE_H
