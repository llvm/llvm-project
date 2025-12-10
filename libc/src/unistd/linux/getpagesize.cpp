//===-- Linux implementation of getpagesize -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getpagesize.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/unistd/sysconf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getpagesize, ()) {
  return static_cast<int>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
}

} // namespace LIBC_NAMESPACE_DECL
