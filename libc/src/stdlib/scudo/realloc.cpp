//===-- Scudo realloc -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"
#include "src/stdlib/realloc.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, realloc, (void *ptr, size_t size)) {
  if (!ptr) {
    void *Ptr =
        Allocator.allocate(size, scudo::Chunk::Origin::Malloc,
                           static_cast<scudo::uptr>(SCUDO_MALLOC_ALIGNMENT));
    reportAllocation(Ptr, size);
    return scudo::setErrnoOnNull(Ptr);
  }
  if (size == 0) {
    reportDeallocation(ptr);
    Allocator.deallocate(ptr, scudo::Chunk::Origin::Malloc);
    return nullptr;
  }

  reportReallocDeallocation(ptr);
  void *NewPtr = Allocator.reallocate(
      ptr, size, static_cast<scudo::uptr>(SCUDO_MALLOC_ALIGNMENT));
  if (NewPtr != nullptr) {
    reportReallocAllocation(ptr, NewPtr, size);
  } else {
    reportReallocAllocation(ptr, ptr,
                            static_cast<size_t>(Allocator.getAllocSize(ptr)));
  }

  return scudo::setErrnoOnNull(NewPtr);
}

} // namespace LIBC_NAMESPACE_DECL
