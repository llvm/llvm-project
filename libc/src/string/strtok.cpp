//===-- Implementation of strtok ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strtok.h"

#include "src/__support/common.h"
#include "src/string/string_utils.h"

namespace __llvm_libc {

static char *strtok_str = nullptr;

char *LLVM_LIBC_ENTRYPOINT(strtok)(char *__restrict src,
                                   const char *__restrict delimiter_string) {
  return internal::string_token(src, delimiter_string, &strtok_str);
}

} // namespace __llvm_libc
