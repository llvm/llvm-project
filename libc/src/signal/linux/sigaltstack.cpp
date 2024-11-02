//===-- Linux implementation of sigaltstack -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigaltstack.h"
#include "src/signal/linux/signal_utils.h"

#include "src/__support/common.h"

#include <errno.h>
#include <signal.h>
#include <sys/syscall.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, sigaltstack,
                   (const stack_t *__restrict ss, stack_t *__restrict oss)) {
  if (ss != nullptr) {
    unsigned not_ss_disable = ~unsigned(SS_DISABLE);
    if ((unsigned(ss->ss_flags) & not_ss_disable) != 0) {
      // Flags cannot have anything other than SS_DISABLE set.
      // We do the type-casting to unsigned because the |ss_flags|
      // field of stack_t is of type "int".
      errno = EINVAL;
      return -1;
    }
    if (ss->ss_size < MINSIGSTKSZ) {
      errno = ENOMEM;
      return -1;
    }
  }

  int ret = __llvm_libc::syscall_impl(SYS_sigaltstack, ss, oss);
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc
