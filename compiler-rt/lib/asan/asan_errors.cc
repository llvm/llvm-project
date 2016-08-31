//===-- asan_errors.cc ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan implementation for error structures.
//===----------------------------------------------------------------------===//

#include "asan_errors.h"
#include "asan_stack.h"

namespace __asan {

void ErrorStackOverflow::Print() {
  Decorator d;
  Printf("%s", d.Warning());
  Report(
      "ERROR: AddressSanitizer: stack-overflow on address %p"
      " (pc %p bp %p sp %p T%d)\n",
      (void *)addr, (void *)pc, (void *)bp, (void *)sp, tid);
  Printf("%s", d.EndWarning());
  scariness.Print();
  BufferedStackTrace stack;
  GetStackTraceWithPcBpAndContext(&stack, kStackTraceMax, pc, bp, context,
                                  common_flags()->fast_unwind_on_fatal);
  stack.Print();
  ReportErrorSummary("stack-overflow", &stack);
}

}  // namespace __asan
