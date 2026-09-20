//===-- Darwin implementation of sigdelset --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigdelset.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigdelset, (sigset_t * set, int signum)) {
  if (!set || signum <= 0 || signum >= NSIG) {
    libc_errno = EINVAL;
    return -1;
  }
  set->__signals[0] &= ~(1U << (signum - 1));
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
