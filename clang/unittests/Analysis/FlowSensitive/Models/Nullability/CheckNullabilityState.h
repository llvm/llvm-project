//===-- CheckNullabilityState.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file contains utilities to check nullability state at specific
// program points in tests.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_UNITTESTS_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_CHECKNULLABILITYSTATE_H
#define CLANG_UNITTESTS_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_CHECKNULLABILITYSTATE_H

#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/StringRef.h"

namespace clang::dataflow::nullability {

/// Runs `PointerNullabilityAnalysis` on the function `target` in `Code` and
/// returns the nullability of the pointer variable `target_p` at the program
/// point marked `// [[check]]`.
NullabilityKind evaluateNullabilityAtCheck(llvm::StringRef Code);

} // namespace clang::dataflow::nullability

#endif // CLANG_UNITTESTS_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_CHECKNULLABILITYSTATE_H
