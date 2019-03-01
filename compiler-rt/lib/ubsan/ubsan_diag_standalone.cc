//===-- ubsan_diag_standalone.cc ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostic reporting for the standalone UBSan runtime.
//
//===----------------------------------------------------------------------===//

#include "ubsan_platform.h"
#if CAN_SANITIZE_UB
#include "ubsan_diag.h"

using namespace __ubsan;

void __sanitizer::GetStackTrace(BufferedStackTrace *stack, uptr max_depth,
                                uptr pc, uptr bp, void *context, bool fast) {
  uptr top = 0;
  uptr bottom = 0;
  if (StackTrace::WillUseFastUnwind(fast)) {
    GetThreadStackTopAndBottom(false, &top, &bottom);
    stack->Unwind(max_depth, pc, bp, nullptr, top, bottom, true);
  } else
    stack->Unwind(max_depth, pc, bp, context, 0, 0, false);
}

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_stack_trace() {
  GET_CURRENT_PC_BP;
  BufferedStackTrace stack;
  stack.Unwind(pc, bp, nullptr, common_flags()->fast_unwind_on_fatal);
  stack.Print();
}
} // extern "C"

#endif  // CAN_SANITIZE_UB
