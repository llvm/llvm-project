//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of getloadavg.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/getloadavg.h"

#include "hdr/types/struct_sysinfo.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/sysinfo.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getloadavg, (double loadavg[], int nelem)) {
  if (nelem <= 0)
    return 0;

  struct sysinfo info;
  auto result = linux_syscalls::sysinfo(&info);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }

  if (nelem > 3)
    nelem = 3;

  // The Linux kernel represents 1, 5, and 15 minute load averages in info.loads
  // scaled by (1 << SI_LOAD_SHIFT), where SI_LOAD_SHIFT is 16 (i.e. 65536.0).
  constexpr double KERNEL_LOAD_SCALE = 65536.0;
  for (int i = 0; i < nelem; ++i)
    loadavg[i] = static_cast<double>(info.loads[i]) / KERNEL_LOAD_SCALE;

  return nelem;
}

} // namespace LIBC_NAMESPACE_DECL
