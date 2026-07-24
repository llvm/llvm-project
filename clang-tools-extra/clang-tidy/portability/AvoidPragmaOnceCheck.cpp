//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidPragmaOnceCheck.h"

#include "../utils/LexerUtils.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::portability {

namespace {

class PragmaOnceCallbacks : public PPCallbacks {
public:
  PragmaOnceCallbacks(AvoidPragmaOnceCheck *Check, const SourceManager &SM)
      : Check(Check), SM(SM) {}
  void PragmaDirective(SourceLocation Loc,
                       PragmaIntroducerKind Introducer) override {
    if (utils::lexer::isPragmaOnce(Loc, SM))
      Check->diag(Loc,
                  "avoid 'pragma once' directive; use include guards instead");
  }

private:
  AvoidPragmaOnceCheck *Check;
  const SourceManager &SM;
};

} // namespace

void AvoidPragmaOnceCheck::registerPPCallbacks(const SourceManager &SM,
                                               Preprocessor *PP,
                                               Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<PragmaOnceCallbacks>(this, SM));
}

} // namespace clang::tidy::portability
