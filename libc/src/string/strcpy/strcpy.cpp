//===-------------------- Implementation of strcpy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcpy/strcpy.h"

#include "src/__support/common.h"

namespace __llvm_libc {

char *LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {
  return reinterpret_cast<char *>(::memcpy(dest, src, ::strlen(src) + 1));
}

} // namespace __llvm_libc
