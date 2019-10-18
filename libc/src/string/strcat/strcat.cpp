//===-------------------- Implementation of strcat -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcat/strcat.h"

#include "src/__support/common.h"
#include "src/string/strcpy/strcpy.h"

namespace __llvm_libc {

char *LLVM_LIBC_ENTRYPOINT(strcat)(char *dest, const char *src) {
  // We do not yet have an implementaion of strlen in so we will use strlen
  // from another libc.
  __llvm_libc::strcpy(dest + ::strlen(dest), src);
  return dest;
}

} // namespace __llvm_libc
