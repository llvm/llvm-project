//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of __sched_cpuequal.
///
//===----------------------------------------------------------------------===//

#include "src/sched/sched_cpuequal.h"
#include "hdr/types/cpu_set_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/common.h"            // LLVM_LIBC_FUNCTION
#include "src/__support/macros/config.h"     // LIBC_NAMESPACE_DECL
#include "src/__support/macros/null_check.h" // LIBC_CRASH_ON_NULLPTR
#include "src/string/memory_utils/inline_memcmp.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, __sched_cpuequal,
                   (size_t cpuset_size, const cpu_set_t *set1,
                    const cpu_set_t *set2)) {
  LIBC_CRASH_ON_NULLPTR(set1);
  LIBC_CRASH_ON_NULLPTR(set2);
  return inline_memcmp(set1, set2, cpuset_size) == 0;
}

} // namespace LIBC_NAMESPACE_DECL
