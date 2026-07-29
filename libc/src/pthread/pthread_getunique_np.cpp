//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the pthread_getunique_np function.
///
//===----------------------------------------------------------------------===//

#include "pthread_getunique_np.h"

#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_getunique_np,
                   (const pthread_t *__restrict thread,
                    pthread_id_np_t *__restrict id)) {
  if (id == nullptr) {
    return EINVAL;
  }
  // We assume that unique thread ID is an integer value of a pointer to TCB.
  *id = (thread == nullptr)
            ? 0
            : reinterpret_cast<pthread_id_np_t>(thread->__attrib);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
