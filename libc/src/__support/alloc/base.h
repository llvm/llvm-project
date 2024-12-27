//===-- A generic base allocator. -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ALLOC_BASE_H
#define LLVM_LIBC_SRC___SUPPORT_ALLOC_BASE_H

#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

class BaseAllocator {
public:
  using AllocFunc = void *(BaseAllocator *self, size_t, size_t);
  using ExpandFunc = void *(BaseAllocator *self, void *, size_t, size_t);
  using FreeFunc = bool(BaseAllocator *self, void *);

private:
  // Implementation specific functions
  AllocFunc *impl_alloc;
  ExpandFunc *impl_expand;
  FreeFunc *impl_free;

public:
  constexpr BaseAllocator(AllocFunc *ia, ExpandFunc *ie, FreeFunc *ifr,
                          size_t default_alignment)
      : impl_alloc(ia), impl_expand(ie), impl_free(ifr),
        default_alignment(default_alignment) {}

  size_t default_alignment;

  void *alloc(size_t alignment, size_t size);
  void *expand(void *ptr, size_t alignment, size_t size);
  bool free(void *ptr);
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ALLOC_BASE_H
