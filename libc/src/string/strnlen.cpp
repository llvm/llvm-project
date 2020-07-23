//===-- Implementation of strnlen------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strnlen.h"

#include "src/__support/common.h"
#include "src/string/memchr.h"
#include <stddef.h>

namespace __llvm_libc {

size_t LLVM_LIBC_ENTRYPOINT(strnlen)(const char *src, size_t n) {
  const char *temp =
      reinterpret_cast<char *>(__llvm_libc::memchr(src, '\0', n));
  return temp ? temp - src : n;
}

} // namespace __llvm_libc
