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

ALIGNED(64) static char SuppressionPlaceholder[sizeof(SuppressionContext)];
static SuppressionContext *SuppressionCtx = nullptr;

// The order should match the enum CheckKind.
static const char *kSuppressionTypes[] = {kSuppressionFcmp,
                                          kSuppressionConsistency};

void InitializeSuppressions() {
  CHECK_EQ(nullptr, SuppressionCtx);
  SuppressionCtx = new (SuppressionPlaceholder)
      SuppressionContext(kSuppressionTypes, ARRAY_SIZE(kSuppressionTypes));
  SuppressionCtx->ParseFromFile(flags().suppressions);
  SuppressionCtx->Parse(__nsan_default_suppressions());
}

static Suppression *GetSuppressionForAddr(uptr Addr, const char *SupprType) {
  Suppression *S = nullptr;

  // Suppress by module name.
  SuppressionContext *Suppressions = SuppressionCtx;
  if (const char *ModuleName =
          Symbolizer::GetOrInit()->GetModuleNameForPc(Addr)) {
    if (Suppressions->Match(ModuleName, SupprType, &S))
      return S;
  }

  // Suppress by file or function name.
  SymbolizedStack *Frames = Symbolizer::GetOrInit()->SymbolizePC(Addr);
  for (SymbolizedStack *Cur = Frames; Cur; Cur = Cur->next) {
    if (Suppressions->Match(Cur->info.function, SupprType, &S) ||
        Suppressions->Match(Cur->info.file, SupprType, &S)) {
      break;
    }
  }
  Frames->ClearAll();
  return S;
}

Suppression *GetSuppressionForStack(const StackTrace *Stack, CheckKind K) {
  for (uptr I = 0, E = Stack->size; I < E; I++) {
    Suppression *S = GetSuppressionForAddr(
        StackTrace::GetPreviousInstructionPc(Stack->trace[I]),
        kSuppressionTypes[static_cast<int>(K)]);
    if (S)
      return S;
  }
  return nullptr;
}

} // end namespace __nsan
