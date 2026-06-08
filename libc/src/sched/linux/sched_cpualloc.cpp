//===-- Implementation of sched_cpualloc ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sched/sched_cpualloc.h"
#include "hdr/errno_macros.h"
#include "hdr/sched_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/cpu_set_t.h"
#include "src/__support/CPP/new.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(cpu_set_t *, __sched_cpualloc, (int count)) {
  AllocChecker ac;
  uint8_t *bytes = new (ac) uint8_t[CPU_ALLOC_SIZE(count)];
  if (!ac) {
    libc_errno = ENOMEM;
    return nullptr;
  }
  return reinterpret_cast<cpu_set_t *>(bytes);
}

} // namespace LIBC_NAMESPACE_DECL
