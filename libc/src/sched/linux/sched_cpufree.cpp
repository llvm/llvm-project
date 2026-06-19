//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of __sched_cpufree.
///
//===----------------------------------------------------------------------===//

#include "src/sched/sched_cpufree.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/cpu_set_t.h"
#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, __sched_cpufree, (cpu_set_t * set)) {
  delete[] reinterpret_cast<uint8_t *>(set);
}

} // namespace LIBC_NAMESPACE_DECL
