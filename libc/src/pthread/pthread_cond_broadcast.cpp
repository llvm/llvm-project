//===-- Implementation of pthread_cond_broadcast --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_cond_broadcast.h"

#include "pthread_cond_utils.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_cond_broadcast, (pthread_cond_t * cond)) {
  pthread_cond_utils::to_cndvar(cond)->broadcast();
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
