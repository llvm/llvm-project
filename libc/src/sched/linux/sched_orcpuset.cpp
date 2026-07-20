//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of __sched_orcpuset.
///
//===----------------------------------------------------------------------===//

#include "src/sched/sched_orcpuset.h"
#include "hdr/types/cpu_set_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/common.h"            // LLVM_LIBC_FUNCTION
#include "src/__support/macros/config.h"     // LIBC_NAMESPACE_DECL
#include "src/__support/macros/null_check.h" // LIBC_CRASH_ON_NULLPTR

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, __sched_orcpuset,
                   (size_t cpuset_size, cpu_set_t *destset,
                    const cpu_set_t *srcset1, const cpu_set_t *srcset2)) {
  LIBC_CRASH_ON_NULLPTR(destset);
  LIBC_CRASH_ON_NULLPTR(srcset1);
  LIBC_CRASH_ON_NULLPTR(srcset2);
  size_t limit = cpuset_size / sizeof(__cpu_set_mask_t);
  for (size_t i = 0; i < limit; ++i)
    destset->__mask[i] = srcset1->__mask[i] | srcset2->__mask[i];
}

} // namespace LIBC_NAMESPACE_DECL
