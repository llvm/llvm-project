//==- AnnexKDetection.cpp - Annex K availability detection -------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of utilities for detecting C11 Annex K
// (Bounds-checking interfaces) availability.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/AnnexKDetection.h"

#include "clang/Basic/LangOptions.h"
#include "clang/Lex/Preprocessor.h"

namespace clang::analysis {

[[nodiscard]] bool isAnnexKAvailable(Preprocessor *PP, const LangOptions &LO) {
  if (!LO.C11)
    return false;

  assert(PP && "No Preprocessor registered.");

  if (!PP->isMacroDefined("__STDC_LIB_EXT1__") ||
      !PP->isMacroDefined("__STDC_WANT_LIB_EXT1__"))
    return false;

  const auto *MI =
      PP->getMacroInfo(PP->getIdentifierInfo("__STDC_WANT_LIB_EXT1__"));
  if (!MI || MI->tokens_empty())
    return false;

  const Token &T = MI->tokens().back();
  if (!T.isLiteral() || !T.getLiteralData())
    return false;

  return StringRef(T.getLiteralData(), T.getLength()) == "1";
}

} // namespace clang::analysis
