//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of __pthread_cleanup_pop.
///
//===----------------------------------------------------------------------===//

#include "src/pthread/__pthread_cleanup_pop.h"
#include "hdr/types/struct___pthread_cleanup_frame.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, __pthread_cleanup_pop, (int execute)) {
  __pthread_cleanup_frame *frame = current_thread().attrib->cleanup_stack.pop();
  LIBC_ASSERT(frame);

  if (execute)
    frame->__routine(frame->__arg);
}

} // namespace LIBC_NAMESPACE_DECL
