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

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_stack_trace() {
  uptr top = 0;
  uptr bottom = 0;
  bool request_fast_unwind = common_flags()->fast_unwind_on_fatal;
  if (request_fast_unwind)
    __sanitizer::GetThreadStackTopAndBottom(false, &top, &bottom);

  GET_CURRENT_PC_BP_SP;
  (void)sp;
  BufferedStackTrace stack;
  stack.Unwind(kStackTraceMax, pc, bp, nullptr, top, bottom,
               request_fast_unwind);
  stack.Print();
}
} // extern "C"

#endif  // CAN_SANITIZE_UB
