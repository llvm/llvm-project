//===-- Linux implementation of signal ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/signal.h"
#include "hdr/signal_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/signal/sigaction.h"

namespace LIBC_NAMESPACE_DECL {

// Our LLVM_LIBC_FUNCTION macro doesn't handle function pointer return types.
using signal_handler = void (*)(int);

LLVM_LIBC_FUNCTION(signal_handler, signal,
                   (int signum, signal_handler handler)) {
  struct sigaction action, old;
  action.sa_handler = handler;
  action.sa_flags = SA_RESTART;
  // Errno will already be set so no need to worry about changing errno here.
  return LIBC_NAMESPACE::sigaction(signum, &action, &old) == -1
             ? SIG_ERR
             : old.sa_handler;
}

} // namespace LIBC_NAMESPACE_DECL
