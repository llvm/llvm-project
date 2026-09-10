//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of nice.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/nice.h"

#include "hdr/errno_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/sys_resource_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getpriority.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/setpriority.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/math_extras.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, nice, (int incr)) {
  // POSIX specifies default nice value as NZERO (20 in <limits.h>).
  // Priority range in Linux is [-NZERO, NZERO - 1].
  constexpr int MIN_PRIO = -NZERO;
  constexpr int MAX_PRIO = NZERO - 1;

  // Query current nice value. Linux raw SYS_getpriority returns (20 - nice),
  // so current priority is (NZERO - prio_result.value()).
  auto prio_result = linux_syscalls::getpriority(PRIO_PROCESS, 0);
  if (!prio_result) {
    libc_errno = prio_result.error();
    return -1;
  }
  int current_prio = NZERO - prio_result.value();

  int target_prio = 0;
  if (add_overflow(current_prio, incr, target_prio))
    target_prio = (incr > 0) ? MAX_PRIO : MIN_PRIO;
  else if (target_prio > MAX_PRIO)
    target_prio = MAX_PRIO;
  else if (target_prio < MIN_PRIO)
    target_prio = MIN_PRIO;

  auto set_result = linux_syscalls::setpriority(PRIO_PROCESS, 0, target_prio);
  if (!set_result) {
    int err = set_result.error();
    // POSIX mandates [EPERM] when lowering nice value without privileges.
    // Linux setpriority returns EACCES; remap to EPERM for POSIX compliance.
    if (err == EACCES)
      err = EPERM;
    libc_errno = err;
    return -1;
  }

  // Returns new nice value - NZERO (in [-NZERO, NZERO - 1]).
  // Note: libc_errno is left unmodified on success to allow -1 disambiguation.
  return target_prio;
}

} // namespace LIBC_NAMESPACE_DECL
