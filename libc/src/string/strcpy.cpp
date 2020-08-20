//===-- Implementation of strcpy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcpy.h"
#include "src/string/strlen.h"
#include "src/string/memcpy.h"

#include "src/__support/common.h"

namespace __llvm_libc {

char *LLVM_LIBC_ENTRYPOINT(strcpy)(char *__restrict dest,
                                   const char *__restrict src) {
  return reinterpret_cast<char *>(
      __llvm_libc::memcpy(dest, src, __llvm_libc::strlen(src) + 1));
}

} // namespace __llvm_libc
