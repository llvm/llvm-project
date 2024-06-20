//===-- nsan_suppressions.cc ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "nsan_suppressions.h"

#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

#include "nsan_flags.h"

// Can be overriden in frontend.
SANITIZER_WEAK_DEFAULT_IMPL
const char *__nsan_default_suppressions() { return 0; }

namespace __nsan {

const char *const kSuppressionFcmp = "fcmp";
const char *const kSuppressionConsistency = "consistency";

using namespace __sanitizer;

alignas(64) static char suppressionPlaceholder[sizeof(SuppressionContext)];
static SuppressionContext *suppressionCtx = nullptr;

// The order should match the enum CheckKind.
static const char *kSuppressionTypes[] = {kSuppressionFcmp,
                                          kSuppressionConsistency};

void InitializeSuppressions() {
  CHECK_EQ(nullptr, suppressionCtx);
  suppressionCtx = new (suppressionPlaceholder)
      SuppressionContext(kSuppressionTypes, ARRAY_SIZE(kSuppressionTypes));
  suppressionCtx->ParseFromFile(flags().suppressions);
  suppressionCtx->Parse(__nsan_default_suppressions());
}

static Suppression *GetSuppressionForAddr(uptr addr, const char *supprType) {
  Suppression *s = nullptr;

  // Suppress by module name.
  SuppressionContext *suppressions = suppressionCtx;
  if (const char *moduleName =
          Symbolizer::GetOrInit()->GetModuleNameForPc(addr)) {
    if (suppressions->Match(moduleName, supprType, &s))
      return s;
  }

  // Suppress by file or function name.
  SymbolizedStack *frames = Symbolizer::GetOrInit()->SymbolizePC(addr);
  for (SymbolizedStack *cur = frames; cur; cur = cur->next) {
    if (suppressions->Match(cur->info.function, supprType, &s) ||
        suppressions->Match(cur->info.file, supprType, &s)) {
      break;
    }
  }
  frames->ClearAll();
  return s;
}

Suppression *GetSuppressionForStack(const StackTrace *stack, CheckKind k) {
  for (uptr i = 0, e = stack->size; i < e; i++) {
    Suppression *s = GetSuppressionForAddr(
        StackTrace::GetPreviousInstructionPc(stack->trace[i]),
        kSuppressionTypes[static_cast<int>(k)]);
    if (s)
      return s;
  }
  return nullptr;
}

} // end namespace __nsan
