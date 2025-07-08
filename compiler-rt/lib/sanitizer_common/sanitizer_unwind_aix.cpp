//===-- sanitizer_unwind_aix.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the unwind.h-based (aka "slow") stack unwinding routines
// available to the tools on AIX.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_AIX
#  include <unwind.h>

#  include "sanitizer_common.h"
#  include "sanitizer_stacktrace.h"

namespace __sanitizer {

struct UnwindTraceArg {
  BufferedStackTrace *stack;
  u32 max_depth;
};

static _Unwind_Reason_Code Unwind_Trace(struct _Unwind_Context *ctx,
                                        void *param) {
  UnwindTraceArg *arg = (UnwindTraceArg *)param;
  CHECK_LT(arg->stack->size, arg->max_depth);
  uptr pc = _Unwind_GetIP(ctx);
  // On AIX 32-bit and 64-bit, addresses up through 0x0fffffff are for kernel.
  if (pc <= 0x0fffffff)
    return _URC_NORMAL_STOP;
  arg->stack->trace_buffer[arg->stack->size++] = pc;
  if (arg->stack->size == arg->max_depth)
    return _URC_NORMAL_STOP;
  return _URC_NO_REASON;
}

void BufferedStackTrace::UnwindSlow(uptr pc, u32 max_depth) {
  CHECK_GE(max_depth, 2);
  size = 0;
  UnwindTraceArg arg = {this, Min(max_depth + 1, kStackTraceMax)};
  _Unwind_Backtrace(Unwind_Trace, &arg);
  // We need to pop a few frames so that pc is on top.
  uptr to_pop = LocatePcInTrace(pc);
  // trace_buffer[0] belongs to the current function so we always pop it,
  // unless there is only 1 frame in the stack trace (1 frame is always better
  // than 0!).
  // 1-frame stacks don't normally happen, but this depends on the actual
  // unwinder implementation (libgcc, libunwind, etc) which is outside of our
  // control.
  if (to_pop == 0 && size > 1)
    to_pop = 1;

  PopStackFrames(to_pop);
  trace_buffer[0] = pc;
}

void BufferedStackTrace::UnwindSlow(uptr pc, void *context, u32 max_depth) {
  CHECK(context);
  UnwindSlow(pc, max_depth);
}
}  // namespace __sanitizer

#endif  // SANITIZER_AIX
