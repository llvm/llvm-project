//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of clock_getcpuclockid function.
///
//===----------------------------------------------------------------------===//

#include "src/time/clock_getcpuclockid.h"
#include "hdr/errno_macros.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/pid_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/clock_getres.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, clock_getcpuclockid, (pid_t pid, clockid_t *clock_id)) {
  if (pid < 0)
    return ESRCH;
  // On Linux, process CPU clocks are encoded as (~pid << 3) | clock_id with
  // clock_id CPUCLOCK_SCHED being 2:
  // https://github.com/torvalds/linux/blob/master/include/linux/posix-timers.h#L22
  const clockid_t pid_clockid =
      static_cast<clockid_t>((static_cast<unsigned int>(~pid) << 3) | 2);

  auto result = linux_syscalls::clock_getres(pid_clockid, nullptr);
  // Note that the clock_getcpuclockid doesn't set errno, it returns the
  // error directly.
  if (!result)
    return result.error() == EINVAL ? ESRCH : result.error();

  *clock_id = pid_clockid;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
