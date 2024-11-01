//===-- Linux implementation of sigaltstack -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigaltstack.h"
#include "hdr/types/stack_t.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include "src/signal/linux/signal_utils.h"

#include "src/__support/common.h"

#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigaltstack,
                   (const stack_t *__restrict ss, stack_t *__restrict oss)) {
  if (ss != nullptr) {
    unsigned not_ss_disable = ~unsigned(SS_DISABLE);
    if ((unsigned(ss->ss_flags) & not_ss_disable) != 0) {
      // Flags cannot have anything other than SS_DISABLE set.
      // We do the type-casting to unsigned because the |ss_flags|
      // field of stack_t is of type "int".
      libc_errno = EINVAL;
      return -1;
    }
    if (ss->ss_size < MINSIGSTKSZ) {
      libc_errno = ENOMEM;
      return -1;
    }
  }

  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_sigaltstack, ss, oss);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
