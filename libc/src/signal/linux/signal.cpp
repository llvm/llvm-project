//===-- Linux implementation of signal ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __LLVM_LIBC_INTERNAL_SIGACTION
#include "src/signal/signal.h"
#include "src/signal/sigaction.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(sighandler_t, signal, (int signum, sighandler_t handler)) {
  struct __sigaction action, old;
  action.sa_handler = handler;
  action.sa_flags = SA_RESTART;
  // Errno will already be set so no need to worry about changing errno here.
  return __llvm_libc::sigaction(signum, &action, &old) == -1 ? SIG_ERR
                                                             : old.sa_handler;
}

} // namespace __llvm_libc
