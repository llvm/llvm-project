//===--- rtsan_suppressions.cpp - Realtime Sanitizer ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the RTSan runtime, providing support for suppressions
//
//===----------------------------------------------------------------------===//

#include "rtsan/rtsan_suppressions.h"

#include "rtsan/rtsan_flags.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_suppressions.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

#include <new>

using namespace __sanitizer;
using namespace __rtsan;

namespace {
enum class ErrorType {
#define RTSAN_CHECK(Name, FSanitizeFlagName) Name,
#include "rtsan_checks.inc"
#undef RTSAN_CHECK
};
} // namespace

alignas(64) static char suppression_placeholder[sizeof(SuppressionContext)];
static SuppressionContext *suppression_ctx = nullptr;

static const char *kSuppressionTypes[] = {
#define RTSAN_CHECK(Name, FSanitizeFlagName) FSanitizeFlagName,
#include "rtsan_checks.inc"
#undef RTSAN_CHECK
};

static const char *ConvertTypeToFlagName(ErrorType Type) {
  switch (Type) {
#define RTSAN_CHECK(Name, FSanitizeFlagName)                                   \
  case ErrorType::Name:                                                        \
    return FSanitizeFlagName;
#include "rtsan_checks.inc"
#undef RTSAN_CHECK
  }
  UNREACHABLE("unknown ErrorType!");
}

void __rtsan::InitializeSuppressions() {
  CHECK_EQ(nullptr, suppression_ctx);

  // We will use suppression_ctx == nullptr as an early out
  if (!flags().ContainsSuppresionFile())
    return;

  suppression_ctx = new (suppression_placeholder)
      SuppressionContext(kSuppressionTypes, ARRAY_SIZE(kSuppressionTypes));
  suppression_ctx->ParseFromFile(flags().suppressions);
}

bool __rtsan::IsStackTraceSuppressed(const StackTrace &stack) {
  if (suppression_ctx == nullptr)
    return false;

  const char *call_stack_flag =
      ConvertTypeToFlagName(ErrorType::CallStackContains);
  if (!suppression_ctx->HasSuppressionType(call_stack_flag))
    return false;

  Symbolizer *symbolizer = Symbolizer::GetOrInit();
  for (uptr i = 0; i < stack.size && stack.trace[i]; i++) {
    const uptr addr = stack.trace[i];

    SymbolizedStackHolder symbolized_stack(symbolizer->SymbolizePC(addr));
    const SymbolizedStack *frames = symbolized_stack.get();
    CHECK(frames);
    for (const SymbolizedStack *cur = frames; cur; cur = cur->next) {
      const char *function_name = cur->info.function;
      if (!function_name)
        continue;

      Suppression *s;
      if (suppression_ctx->Match(function_name, call_stack_flag, &s))
        return true;
    }
  }
  return false;
}

bool __rtsan::IsFunctionSuppressed(const char *function_name) {
  if (suppression_ctx == nullptr)
    return false;

  const char *flag_name = ConvertTypeToFlagName(ErrorType::FunctionNameMatches);

  if (!suppression_ctx->HasSuppressionType(flag_name))
    return false;

  Suppression *s;
  return suppression_ctx->Match(function_name, flag_name, &s);
}
