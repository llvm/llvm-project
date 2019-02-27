//===-- tsan_stack_trace.cc -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_stack_trace.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"

namespace __tsan {

VarSizeStackTrace::VarSizeStackTrace()
    : StackTrace(nullptr, 0), trace_buffer(nullptr) {}

VarSizeStackTrace::~VarSizeStackTrace() {
  ResizeBuffer(0);
}

void VarSizeStackTrace::ResizeBuffer(uptr new_size) {
  if (trace_buffer) {
    internal_free(trace_buffer);
  }
  trace_buffer =
      (new_size > 0)
          ? (uptr *)internal_alloc(MBlockStackTrace,
                                   new_size * sizeof(trace_buffer[0]))
          : nullptr;
  trace = trace_buffer;
  size = new_size;
}

void VarSizeStackTrace::Init(const uptr *pcs, uptr cnt, uptr extra_top_pc) {
  ResizeBuffer(cnt + !!extra_top_pc);
  internal_memcpy(trace_buffer, pcs, cnt * sizeof(trace_buffer[0]));
  if (extra_top_pc)
    trace_buffer[cnt] = extra_top_pc;
}

void VarSizeStackTrace::ReverseOrder() {
  for (u32 i = 0; i < (size >> 1); i++)
    Swap(trace_buffer[i], trace_buffer[size - 1 - i]);
}

}  // namespace __tsan

void __sanitizer::GetStackTrace(BufferedStackTrace *stack, uptr max_depth,
                                uptr pc, uptr bp, void *context,
                                bool request_fast_unwind) {
  uptr top = 0;
  uptr bottom = 0;
  if (StackTrace::WillUseFastUnwind(request_fast_unwind)) {
    GetThreadStackTopAndBottom(false, &top, &bottom);
    stack->Unwind(kStackTraceMax, pc, bp, nullptr, top, bottom, true);
  } else
    stack->Unwind(kStackTraceMax, pc, 0, context, 0, 0, false);
}
