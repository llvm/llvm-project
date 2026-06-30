//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for __sched_orcpuset.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SCHED_SCHED_ORCPUSET_H
#define LLVM_LIBC_SRC_SCHED_SCHED_ORCPUSET_H

#include "hdr/types/cpu_set_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// for internal use in the CPU_OR macro
void __sched_orcpuset(size_t cpuset_size, cpu_set_t *destset,
                      const cpu_set_t *srcset1, const cpu_set_t *srcset2);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SCHED_SCHED_ORCPUSET_H
