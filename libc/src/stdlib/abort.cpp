//===-- Implementation of abort -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/signal/raise.h"
#include "src/stdlib/_Exit.h"

#include "src/stdlib/abort.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, abort, ()) {
  // TODO: When sigprocmask and sigaction land:
  // Unblock SIGABRT, raise it, if it was ignored or the handler returned,
  // change its action to SIG_DFL, raise it again.
  // TODO: When C11 mutexes land:
  // Acquire recursive mutex (in case the current signal handler for SIGABRT
  // itself calls abort we don't want to deadlock on the same thread trying
  // to acquire it's own mutex.)
  __llvm_libc::raise(SIGABRT);
  __llvm_libc::raise(SIGKILL);
  __llvm_libc::_Exit(127);
}

} // namespace __llvm_libc
