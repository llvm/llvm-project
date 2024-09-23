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
  const char *FunctionName() const { return Green(); }
  const char *Reason() const { return Blue(); }
};

template <class... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};
// TODO: Remove below when c++20
template <class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;
} // namespace

static void PrintStackTrace(uptr pc, uptr bp) {
  BufferedStackTrace stack{};

  stack.Unwind(pc, bp, nullptr, common_flags()->fast_unwind_on_fatal);
  stack.Print();
}

static void PrintError(const Decorator &decorator,
                       const DiagnosticsCallerInfo &info) {
  const char *violation_type = std::visit(
      Overloaded{
          [](const InterceptedCallInfo &) { return "unsafe-library-call"; },
          [](const BlockingCallInfo &) { return "blocking-call"; }},
      info);

  Printf("%s", decorator.Error());
  Report("ERROR: RealtimeSanitizer: %s\n", violation_type);
}

static void PrintReason(const Decorator &decorator,
                        const DiagnosticsCallerInfo &info) {
  Printf("%s", decorator.Reason());

  std::visit(
      Overloaded{[decorator](const InterceptedCallInfo &call) {
                   Printf("Intercepted call to real-time unsafe function "
                          "`%s%s%s` in real-time context!",
                          decorator.FunctionName(),
                          call.intercepted_function_name_, decorator.Reason());
                 },
                 [decorator](const BlockingCallInfo &arg) {
                   Printf("Call to blocking function "
                          "`%s%s%s` in real-time context!",
                          decorator.FunctionName(), arg.blocking_function_name_,
                          decorator.Reason());
                 }},
      info);

  Printf("\n");
}

void __rtsan::PrintDiagnostics(const DiagnosticsInfo &info) {
  ScopedErrorReportLock l;

  Decorator d;
  PrintError(d, info.call_info);
  PrintReason(d, info.call_info);
  Printf("%s", d.Default());
  PrintStackTrace(info.pc, info.bp);
}
