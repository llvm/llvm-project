//===--- rtsan_diagnostics.cpp - Realtime Sanitizer -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "rtsan/rtsan_diagnostics.h"

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __sanitizer;
using namespace __rtsan;

// We must define our own implementation of this method for our runtime.
// This one is just copied from UBSan.
namespace __sanitizer {
void BufferedStackTrace::UnwindImpl(uptr pc, uptr bp, void *context,
                                    bool request_fast, u32 max_depth) {
  uptr top = 0;
  uptr bottom = 0;
  GetThreadStackTopAndBottom(false, &top, &bottom);
  bool fast = StackTrace::WillUseFastUnwind(request_fast);
  Unwind(max_depth, pc, bp, context, top, bottom, fast);
}
} // namespace __sanitizer

namespace {
class Decorator : public __sanitizer::SanitizerCommonDecorator {
public:
  Decorator() : SanitizerCommonDecorator() {}
  const char *FunctionName() { return Green(); }
  const char *Reason() { return Blue(); }
};
} // namespace

static void PrintStackTrace(uptr pc, uptr bp) {
  BufferedStackTrace stack{};

  stack.Unwind(pc, bp, nullptr, common_flags()->fast_unwind_on_fatal);
  stack.Print();
}

void __rtsan::PrintDiagnostics(const char *intercepted_function_name, uptr pc,
                               uptr bp) {
  ScopedErrorReportLock l;

  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: RealtimeSanitizer: unsafe-library-call\n");
  Printf("%s", d.Reason());
  Printf("Intercepted call to real-time unsafe function "
         "`%s%s%s` in real-time context!\n",
         d.FunctionName(), intercepted_function_name, d.Reason());

  Printf("%s", d.Default());
  PrintStackTrace(pc, bp);
}
