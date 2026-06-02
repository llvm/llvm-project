//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the FreeStore template class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREESTORE_H
#define LLVM_LIBC_SRC___SUPPORT_FREESTORE_H

#include "hdr/types/size_t.h"
#include "src/__support/block.h"
#include "src/__support/freelist.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

template <typename T> class FreeStore {
protected:
  static constexpr size_t MIN_OUTER_SIZE =
      align_up(sizeof(Block) + sizeof(FreeList::Node), Block::MIN_ALIGN);

  LIBC_INLINE static bool too_small(Block *block) {
    return block->outer_size() < MIN_OUTER_SIZE;
  }

public:
  LIBC_INLINE void insert(Block *block) {
    static_cast<T *>(this)->insert_impl(block);
  }

  LIBC_INLINE void remove(Block *block) {
    static_cast<T *>(this)->remove_impl(block);
  }

  LIBC_INLINE Block *find_and_remove_fit(size_t size) {
    return static_cast<T *>(this)->find_and_remove_fit_impl(size);
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREESTORE_H
