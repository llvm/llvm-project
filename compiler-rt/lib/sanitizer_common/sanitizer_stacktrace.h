//===-- sanitizer_stacktrace.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_STACKTRACE_H
#define SANITIZER_STACKTRACE_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

static const uptr kStackTraceMax = 256;

#if SANITIZER_LINUX && (defined(__arm__) || \
    defined(__powerpc__) || defined(__powerpc64__) || \
    defined(__sparc__) || \
    defined(__mips__))
# define SANITIZER_CAN_FAST_UNWIND 0
#elif SANITIZER_WINDOWS
# define SANITIZER_CAN_FAST_UNWIND 0
#else
# define SANITIZER_CAN_FAST_UNWIND 1
#endif

struct StackTrace {
  typedef bool (*SymbolizeCallback)(const void *pc, char *out_buffer,
                                     int out_size);
  uptr top_frame_bp;
  uptr size;
  uptr trace[kStackTraceMax];

  // Prints a symbolized stacktrace, followed by an empty line.
  static void PrintStack(const uptr *addr, uptr size,
                         SymbolizeCallback symbolize_callback = 0);

  void CopyFrom(const uptr *src, uptr src_size) {
    top_frame_bp = 0;
    size = src_size;
    if (size > kStackTraceMax) size = kStackTraceMax;
    for (uptr i = 0; i < size; i++)
      trace[i] = src[i];
  }

  static bool WillUseFastUnwind(bool request_fast_unwind) {
    // Check if fast unwind is available. Fast unwind is the only option on Mac.
    if (!SANITIZER_CAN_FAST_UNWIND)
      return false;
    else if (SANITIZER_MAC)
      return true;
    return request_fast_unwind;
  }

  void Unwind(uptr max_depth, uptr pc, uptr bp, uptr stack_top,
              uptr stack_bottom, bool request_fast_unwind);

  static uptr GetCurrentPc();
  static uptr GetPreviousInstructionPc(uptr pc);

 private:
  void FastUnwindStack(uptr pc, uptr bp, uptr stack_top, uptr stack_bottom,
                       uptr max_depth);
  void SlowUnwindStack(uptr pc, uptr max_depth);
  void PopStackFrames(uptr count);
  uptr LocatePcInTrace(uptr pc);
};

}  // namespace __sanitizer

// Use this macro if you want to print stack trace with the caller
// of the current function in the top frame.
#define GET_CALLER_PC_BP_SP \
  uptr bp = GET_CURRENT_FRAME();              \
  uptr pc = GET_CALLER_PC();                  \
  uptr local_stack;                           \
  uptr sp = (uptr)&local_stack

// Use this macro if you want to print stack trace with the current
// function in the top frame.
#define GET_CURRENT_PC_BP_SP \
  uptr bp = GET_CURRENT_FRAME();              \
  uptr pc = StackTrace::GetCurrentPc();   \
  uptr local_stack;                           \
  uptr sp = (uptr)&local_stack


#endif  // SANITIZER_STACKTRACE_H
