//===-- Implementation of strcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcmp.h"

#include "src/__support/common.h"

namespace __llvm_libc {

// TODO: Look at benefits for comparing words at a time.
int LLVM_LIBC_ENTRYPOINT(strcmp)(const char *left, const char *right) {
  for (; *left && *left == *right; ++left, ++right)
    ;
  return *reinterpret_cast<const unsigned char *>(left) -
         *reinterpret_cast<const unsigned char *>(right);
}

} // namespace __llvm_libc
