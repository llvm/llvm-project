//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of calloc for UEFI.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/calloc.h"
#include "src/__support/OSUtil/uefi/pool_allocator.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, calloc, (size_t num, size_t size)) {
  return uefi::pool_calloc(num, size);
}

} // namespace LIBC_NAMESPACE_DECL
