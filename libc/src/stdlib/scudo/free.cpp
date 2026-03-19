//===-- Scudo free --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"
#include "src/stdlib/free.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, free, (void *ptr)) {
  reportDeallocation(ptr);
  Allocator.deallocate(ptr, scudo::Chunk::Origin::Malloc);
}

} // namespace LIBC_NAMESPACE_DECL
