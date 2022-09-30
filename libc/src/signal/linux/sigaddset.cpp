//===-- Linux implementation of sigaddset ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigaddset.h"
#include "src/__support/common.h"
#include "src/signal/linux/signal_utils.h"

#include <errno.h>
#include <signal.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, sigaddset, (sigset_t * set, int signum)) {
  if (set != nullptr && add_signal(*set, signum))
    return 0;
  errno = EINVAL;
  return -1;
}

} // namespace __llvm_libc
