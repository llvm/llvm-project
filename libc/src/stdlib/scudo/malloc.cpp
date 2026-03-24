//===-- Scudo malloc ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"
#include "src/stdlib/malloc.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, malloc, (size_t size)) {
  void *Ptr =
      Allocator.allocate(size, scudo::Chunk::Origin::Malloc,
                         static_cast<scudo::uptr>(SCUDO_MALLOC_ALIGNMENT));
  reportAllocation(Ptr, size);
  return scudo::setErrnoOnNull(Ptr);
}

} // namespace LIBC_NAMESPACE_DECL
