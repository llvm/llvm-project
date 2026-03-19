//===-- Scudo aligned_alloc ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"
#include "src/stdlib/aligned_alloc.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, aligned_alloc, (size_t alignment, size_t size)) {
  if (UNLIKELY(scudo::checkAlignedAllocAlignmentAndSize(alignment, size))) {
    if (Allocator.canReturnNull()) {
      scudo::LibcPAL::seterrno(EINVAL);
      return nullptr;
    }
    scudo::reportInvalidAlignedAllocAlignment(
        static_cast<scudo::uptr>(alignment), static_cast<scudo::uptr>(size));
  }

  void *Ptr = Allocator.allocate(size, scudo::Chunk::Origin::Malloc,
                                 static_cast<scudo::uptr>(alignment));
  reportAllocation(Ptr, size);
  return scudo::setErrnoOnNull(Ptr);
}

} // namespace LIBC_NAMESPACE_DECL
