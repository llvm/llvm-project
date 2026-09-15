//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of __pthread_cleanup_push.
///
//===----------------------------------------------------------------------===//

#include "src/pthread/__pthread_cleanup_push.h"
#include "hdr/types/struct___pthread_cleanup_frame.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, __pthread_cleanup_push,
                   (struct __pthread_cleanup_frame * frame,
                    void (*routine)(void *), void *arg)) {
  LIBC_CRASH_ON_NULLPTR(frame);
  LIBC_CRASH_ON_NULLPTR(routine);
  auto *attrib = current_thread().attrib;
  frame->__routine = routine;
  frame->__arg = arg;
  frame->__next = attrib->cleanup_stack;
  attrib->cleanup_stack = frame;
}

} // namespace LIBC_NAMESPACE_DECL
