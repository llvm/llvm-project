//===-- Linux implementation of sigfillset --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigfillset.h"
#include "include/errno.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/signal/linux/signal.h"

#include "src/__support/common.h"

namespace __llvm_libc {

int LLVM_LIBC_ENTRYPOINT(sigfillset)(sigset_t *set) {
  if (!set) {
    llvmlibc_errno = EINVAL;
    return -1;
  }
  auto *sigset = reinterpret_cast<__llvm_libc::Sigset *>(set);
  *sigset = __llvm_libc::Sigset::fullset();
  return 0;
}

} // namespace __llvm_libc
