//===--- FunctionUtils.cpp - Shared function emission queries -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/FunctionUtils.h"

namespace clang::CodeGenUtils {

bool shouldEmitLifetimeMarkers(const CodeGenOptions &CGOpts,
                               const LangOptions &LangOpts) {
  if (CGOpts.DisableLifetimeMarkers)
    return false;

  // Sanitizers may use markers.
  if (CGOpts.SanitizeAddressUseAfterScope ||
      LangOpts.Sanitize.has(SanitizerKind::HWAddress) ||
      LangOpts.Sanitize.has(SanitizerKind::Memory) ||
      LangOpts.Sanitize.has(SanitizerKind::MemtagStack))
    return true;

  // For now, only in optimized builds.
  return CGOpts.OptimizationLevel != 0;
}

} // namespace clang::CodeGenUtils
