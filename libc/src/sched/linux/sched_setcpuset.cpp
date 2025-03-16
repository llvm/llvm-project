//===-- Implementation of sched_setcpuset ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sched/sched_setcpuset.h"

#include "src/__support/common.h"        // LLVM_LIBC_FUNCTION
#include "src/__support/macros/config.h" // LIBC_NAMESPACE_DECL

#include "hdr/types/cpu_set_t.h"
#include "hdr/types/size_t.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, __sched_setcpuset,
                   (int cpu, const size_t cpuset_size, cpu_set_t *set)) {
  if (cpu / 8 < cpuset_size) {
    const size_t element_index = cpu / NCPUBITS;
    const size_t bit_position = cpu % NCPUBITS;

    const unsigned long mask = 1UL << bit_position;
    set->__bits[element_index] |= mask;
  }
}

} // namespace LIBC_NAMESPACE_DECL
