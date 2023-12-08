//===-- Implementation of swab --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/swab.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void, swab,
                   (const void *__restrict from, void *__restrict to,
                    ssize_t n)) {
  const unsigned char *f = static_cast<const unsigned char *>(from);
  unsigned char *t = static_cast<unsigned char *>(to);
  for (ssize_t i = 1; i < n; i += 2) {
    t[i - 1] = f[i];
    t[i] = f[i - 1];
  }
}

} // namespace LIBC_NAMESPACE
