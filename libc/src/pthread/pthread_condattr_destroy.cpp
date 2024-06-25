//===-- Implementation of the pthread_condattr_destroy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_condattr_destroy.h"

#include "src/__support/common.h"

#include <pthread.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_condattr_destroy,
                   (pthread_condattr_t * attr [[gnu::unused]])) {
  // Initializing a pthread_condattr_t acquires no resources, so this is a
  // no-op.
  return 0;
}

} // namespace LIBC_NAMESPACE
