//===-- Implementation of bzero -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/bzero.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_bzero.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, bzero, (void *ptr, size_t count)) {
  inline_bzero(reinterpret_cast<char *>(ptr), count);
}

} // namespace LIBC_NAMESPACE_DECL
