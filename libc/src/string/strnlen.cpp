//===-- Implementation of strnlen------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strnlen.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, strnlen, (const char *src, size_t n)) {
  const void *temp = internal::find_first_character(
      reinterpret_cast<const unsigned char *>(src), '\0', n);
  return temp ? reinterpret_cast<const char *>(temp) - src : n;
}

} // namespace LIBC_NAMESPACE_DECL
