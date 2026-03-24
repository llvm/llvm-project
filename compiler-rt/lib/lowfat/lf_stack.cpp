//===-- lf_stack.cpp - LowFat stack trace implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides BufferedStackTrace::UnwindImpl for the LowFat runtime.
//
// LowFat does not maintain a thread registry, so we obtain the stack bounds
// directly from the OS via GetThreadStackTopAndBottom().
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

void __sanitizer::BufferedStackTrace::UnwindImpl(
    uptr pc, uptr bp, void *context, bool request_fast, u32 max_depth) {
  size = 0;
  request_fast = StackTrace::WillUseFastUnwind(request_fast);
  uptr stack_top = 0, stack_bottom = 0;
  GetThreadStackTopAndBottom(/*at_initialization=*/false, &stack_top,
                             &stack_bottom);
  Unwind(max_depth, pc, bp, context, stack_top, stack_bottom, request_fast);
}
