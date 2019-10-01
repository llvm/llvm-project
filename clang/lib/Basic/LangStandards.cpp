//===--- LangStandards.cpp - Language Standard Definitions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/LangStandard.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;

#define LANGSTANDARD(id, name, lang, desc, features)                           \
  static const LangStandard Lang_##id = {name, desc, features, Language::lang};
#include "clang/Basic/LangStandards.def"

const LangStandard &LangStandard::getLangStandardForKind(Kind K) {
  switch (K) {
  case lang_unspecified:
    llvm::report_fatal_error("getLangStandardForKind() on unspecified kind");
#define LANGSTANDARD(id, name, lang, desc, features) \
    case lang_##id: return Lang_##id;
#include "clang/Basic/LangStandards.def"
  }
  llvm_unreachable("Invalid language kind!");
}

LangStandard::Kind LangStandard::getLangKind(StringRef Name) {
  return llvm::StringSwitch<Kind>(Name)
#define LANGSTANDARD(id, name, lang, desc, features) .Case(name, lang_##id)
#define LANGSTANDARD_ALIAS(id, alias) .Case(alias, lang_##id)
#include "clang/Basic/LangStandards.def"
      .Default(lang_unspecified);
}

const LangStandard *LangStandard::getLangStandardForName(StringRef Name) {
  Kind K = getLangKind(Name);
  if (K == lang_unspecified)
    return nullptr;

  return &getLangStandardForKind(K);
}


