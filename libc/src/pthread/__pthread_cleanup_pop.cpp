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
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, __pthread_cleanup_pop,
                   (struct __pthread_cleanup_frame * frame, int execute)) {
  LIBC_CRASH_ON_NULLPTR(frame);
  auto *attrib = current_thread().attrib;
  attrib->cleanup_stack = frame->__next;
  if (execute)
    frame->__routine(frame->__arg);
}

} // namespace LIBC_NAMESPACE_DECL
