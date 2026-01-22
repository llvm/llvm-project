//==- AnnexKDetection.h - Annex K availability detection ---------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for detecting C11 Annex K (Bounds-checking
// interfaces) availability.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANNEXKDETECTION_H
#define LLVM_CLANG_ANALYSIS_ANNEXKDETECTION_H

namespace clang {
class Preprocessor;
class LangOptions;
} // namespace clang

namespace clang::analysis {

/// Calculates whether Annex K is available for the current translation unit
/// based on the macro definitions and the language options.
///
/// Annex K (Bounds-checking interfaces) is available when:
/// 1. C11 standard is enabled
/// 2. __STDC_LIB_EXT1__ macro is defined (indicates library support)
/// 3. __STDC_WANT_LIB_EXT1__ macro is defined and equals "1" (indicates user
///    opt-in)
///
/// \param PP The preprocessor instance to check macro definitions.
/// \param LO The language options to check C11 standard.
/// \returns true if Annex K is available, false otherwise.
[[nodiscard]] bool isAnnexKAvailable(Preprocessor *PP, const LangOptions &LO);

} // namespace clang::analysis

#endif // LLVM_CLANG_ANALYSIS_ANNEXKDETECTION_H
