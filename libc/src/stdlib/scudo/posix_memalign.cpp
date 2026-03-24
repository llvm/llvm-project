//===-- Scudo posix_memalign ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"
#include "src/stdlib/posix_memalign.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, posix_memalign,
                   (void **memptr, size_t alignment, size_t size)) {
  if (UNLIKELY(
          scudo::checkPosixMemalignAlignment(static_cast<scudo::uptr>(alignment)))) {
    if (!Allocator.canReturnNull())
      scudo::reportInvalidPosixMemalignAlignment(
          static_cast<scudo::uptr>(alignment));
    return EINVAL;
  }

  void *Ptr = Allocator.allocate(size, scudo::Chunk::Origin::Memalign,
                                 static_cast<scudo::uptr>(alignment));
  if (UNLIKELY(!Ptr))
    return ENOMEM;

  reportAllocation(Ptr, size);
  *memptr = Ptr;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
