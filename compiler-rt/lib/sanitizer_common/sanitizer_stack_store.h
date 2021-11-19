//===-- sanitizer_stack_store.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

  Id Store(const StackTrace &trace);
  StackTrace Load(Id id);
  uptr Allocated() const { return atomic_load_relaxed(&mapped_size_); }

  void TestOnlyUnmap();

 private:
  uptr *Alloc(uptr count = 1);
  uptr *TryAlloc(uptr count);
  uptr *RefillAndAlloc(uptr count);
  mutable StaticSpinMutex mtx_ = {};  // Protects alloc of new blocks.
  atomic_uintptr_t region_pos_ = {};  // Region allocator for Node's.
  atomic_uintptr_t region_end_ = {};
  atomic_uintptr_t mapped_size_ = {};

  struct BlockInfo {
    const BlockInfo *next;
    uptr ptr;
    uptr size;
  };
  const BlockInfo *curr_ = nullptr;
};

}  // namespace __sanitizer

#endif  // SANITIZER_STACK_STORE_H
