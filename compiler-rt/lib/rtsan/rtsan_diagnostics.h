//===--- rtsan_diagnostics.h - Realtime Sanitizer ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the RealtimeSanitizer runtime library
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

#include <variant>

namespace __rtsan {

struct InterceptedCallInfo {
  const char *intercepted_function_name_;
};

struct BlockingCallInfo {
public:
  const char *blocking_function_name_;
};

using DiagnosticsCallerInfo =
    std::variant<InterceptedCallInfo, BlockingCallInfo>;

struct DiagnosticsInfo {
  DiagnosticsCallerInfo call_info;

  __sanitizer::uptr pc;
  __sanitizer::uptr bp;
};

void PrintDiagnostics(const DiagnosticsInfo &info);
} // namespace __rtsan
