//===-- An arena allocator using pages. -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ALLOC_ARENA_H
#define LLVM_LIBC_SRC___SUPPORT_ALLOC_ARENA_H

#include "hdr/types/size_t.h"
#include "src/__support/alloc/base.h"
#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

void *arena_allocate(BaseAllocator *base, size_t alignment, size_t size);
void *arena_expand(BaseAllocator *base, void *ptr, size_t alignment,
                   size_t size);
bool arena_free(BaseAllocator *base, void *ptr);

class ArenaAllocator : public BaseAllocator {
public:
  uint8_t *buffer;
  size_t buffer_size;
  size_t prev_offset;
  size_t curr_offset;

private:
  size_t page_size;

public:
  constexpr ArenaAllocator(size_t page_size, size_t default_alignment)
      : BaseAllocator(arena_allocate, arena_expand, arena_free,
                      default_alignment),
        buffer(nullptr), buffer_size(0), prev_offset(0), curr_offset(0),
        page_size(page_size) {}

  size_t get_page_size();
};

extern BaseAllocator *arena_allocator;

} // namespace LIBC_NAMESPACE_DECL

#endif
