//===-- Implementation of strlen ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strlen.h"

#include "src/__support/common.h"

namespace __llvm_libc {

// TODO: investigate the performance of this function.
// There might be potential for compiler optimization.
size_t LLVM_LIBC_ENTRYPOINT(strlen)(const char *src) {
  const char *end = src;
  while (*end != '\0')
    ++end;
  return end - src;
}

} // namespace __llvm_libc
