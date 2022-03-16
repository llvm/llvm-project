//===--- LangStandards.cpp - Language Standard Definitions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/LangStandard.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
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

LangStandard::Kind clang::getDefaultLanguageStandard(clang::Language Lang,
                                                     const llvm::Triple &T) {
  switch (Lang) {
  case Language::Unknown:
  case Language::LLVM_IR:
    llvm_unreachable("Invalid input kind!");
  case Language::OpenCL:
    return LangStandard::lang_opencl12;
  case Language::OpenCLCXX:
    return LangStandard::lang_openclcpp10;
  case Language::CUDA:
    return LangStandard::lang_cuda;
  case Language::Asm:
  case Language::C:
#if defined(CLANG_DEFAULT_STD_C)
    return CLANG_DEFAULT_STD_C;
#else
    // The PS4 uses C99 as the default C standard.
    if (T.isPS4())
      return LangStandard::lang_gnu99;
    return LangStandard::lang_gnu17;
#endif
  case Language::ObjC:
#if defined(CLANG_DEFAULT_STD_C)
    return CLANG_DEFAULT_STD_C;
#else
    return LangStandard::lang_gnu11;
#endif
  case Language::CXX:
  case Language::ObjCXX:
#if defined(CLANG_DEFAULT_STD_CXX)
    return CLANG_DEFAULT_STD_CXX;
#else
    if (T.isDriverKit())
      return LangStandard::lang_gnucxx17;
    else
      return LangStandard::lang_gnucxx14;
#endif
  case Language::RenderScript:
    return LangStandard::lang_c99;
  case Language::HIP:
    return LangStandard::lang_hip;
  case Language::HLSL:
    return LangStandard::lang_hlsl2021;
  }
  llvm_unreachable("unhandled Language kind!");
}
