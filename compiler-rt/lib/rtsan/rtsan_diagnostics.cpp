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
class Decorator : public SanitizerCommonDecorator {
public:
  Decorator() : SanitizerCommonDecorator() {}
  const char *FunctionName() const { return Green(); }
  const char *Reason() const { return Blue(); }
};
} // namespace

static const char *GetErrorTypeStr(const DiagnosticsInfo &info) {
  switch (info.type) {
  case DiagnosticsInfoType::InterceptedCall:
    return "unsafe-library-call";
  case DiagnosticsInfoType::BlockingCall:
    return "blocking-call";
  }
  CHECK(false);
  return "(unknown error)";
}

static void PrintError(const Decorator &decorator,
                       const DiagnosticsInfo &info) {

  Printf("%s", decorator.Error());
  Report("ERROR: RealtimeSanitizer: %s\n", GetErrorTypeStr(info));
}

static void PrintReason(const Decorator &decorator,
                        const DiagnosticsInfo &info) {
  Printf("%s", decorator.Reason());

  switch (info.type) {
  case DiagnosticsInfoType::InterceptedCall: {
    Printf("Intercepted call to real-time unsafe function "
           "`%s%s%s` in real-time context!",
           decorator.FunctionName(), info.func_name, decorator.Reason());
    break;
  }
  case DiagnosticsInfoType::BlockingCall: {
    Printf("Call to blocking function "
           "`%s%s%s` in real-time context!",
           decorator.FunctionName(), info.func_name, decorator.Reason());
    break;
  }
  }

  Printf("\n");
}

void __rtsan::PrintDiagnostics(const DiagnosticsInfo &info) {
  ScopedErrorReportLock::CheckLocked();

  Decorator d;
  PrintError(d, info);
  PrintReason(d, info);
  Printf("%s", d.Default());
}

void __rtsan::PrintErrorSummary(const DiagnosticsInfo &info,
                                const BufferedStackTrace &stack) {
  ScopedErrorReportLock::CheckLocked();
  ReportErrorSummary(GetErrorTypeStr(info), &stack);
}
