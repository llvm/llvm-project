//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of malloc for UEFI.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/malloc.h"
#include "src/__support/OSUtil/uefi/pool_allocator.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, malloc, (size_t size)) {
  return uefi::pool_allocate(size);
}

} // namespace LIBC_NAMESPACE_DECL
