//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of sigaltstack.
///
//===----------------------------------------------------------------------===//

#include "src/signal/sigaltstack.h"

#include "hdr/signal_macros.h"
#include "hdr/types/stack_t.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigaltstack,
                   (const stack_t *__restrict ss, stack_t *__restrict oss)) {
  if (ss != nullptr) {
    if (ss->ss_flags != 0 && ss->ss_flags != SS_DISABLE) {
      libc_errno = EINVAL;
      return -1;
    }
    // ss_size is only validated if the alternate stack is being enabled.
    // When disabling (SS_DISABLE), ss_size and ss_sp are ignored.
    if (ss->ss_flags != SS_DISABLE && ss->ss_size < MINSIGSTKSZ) {
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
