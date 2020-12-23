//===-- Linux implementation of sigaddset ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigaddset.h"
#include "include/errno.h" // For E* macros.
#include "src/errno/llvmlibc_errno.h"
#include "src/signal/linux/signal.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, sigaddset, (sigset_t * set, int signum)) {
  if (!set || (unsigned)(signum - 1) >= (8 * sizeof(sigset_t))) {
    llvmlibc_errno = EINVAL;
    return -1;
  }
  auto *sigset = reinterpret_cast<__llvm_libc::Sigset *>(set);
  sigset->addset(signum);
  return 0;
}

} // namespace __llvm_libc
