//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the pthread_getthreadid_np function.
///
//===----------------------------------------------------------------------===//

#include "pthread_getthreadid_np.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(pthread_id_np_t, pthread_getthreadid_np, ()) {
  // We assume that unique thread ID is an integer value of a pointer to TCB.
  return reinterpret_cast<pthread_id_np_t>(self.attrib);
}

} // namespace LIBC_NAMESPACE_DECL
