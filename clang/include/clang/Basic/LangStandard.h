//===--- LangStandard.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_LANGSTANDARD_H
#define LLVM_CLANG_BASIC_LANGSTANDARD_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Triple;
}

namespace clang {

/// The language for the input, used to select and validate the language
/// standard and possible actions.
enum class Language : uint8_t {
  Unknown,

  /// Assembly: we accept this only so that we can preprocess it.
  Asm,

  /// LLVM IR & CIR: we accept these so that we can run the optimizer on them,
  /// and compile them to assembly or object code (or LLVM for CIR).
  CIR,
  LLVM_IR,

  ///@{ Languages that the frontend can parse and compile.
  C,
  CXX,
  ObjC,
  ObjCXX,
  OpenCL,
  OpenCLCXX,
  CUDA,
  HIP,
  HLSL,
  ///@}
};
StringRef languageToString(Language L);

enum LangFeatures {
  LineComment = (1 << 0),
  C99 = (1 << 1),
  C11 = (1 << 2),
  C17 = (1 << 3),
  C23 = (1 << 4),
  C2y = (1 << 5),
  CPlusPlus = (1 << 8),
  CPlusPlus11 = (1 << 9),
  CPlusPlus14 = (1 << 10),
  CPlusPlus17 = (1 << 11),
  CPlusPlus20 = (1 << 12),
  CPlusPlus23 = (1 << 13),
  CPlusPlus26 = (1 << 14),
  CPlusPlus29 = (1 << 15),
  Digraphs = (1 << 20),
  GNUMode = (1 << 21),
  HexFloat = (1 << 22),
  OpenCL = (1 << 23),
  HLSL = (1 << 24)
};

/// LangStandard - Information about the properties of a particular language
/// standard.
struct LangStandard {
  enum Kind {
#define LANGSTANDARD(id, name, lang, desc, features, version) lang_##id,
#include "clang/Basic/LangStandards.def"
    lang_unspecified
  };

  const char *ShortName;
  const char *Description;
  unsigned Flags;
  clang::Language Language;
  std::optional<uint32_t> Version;

public:
  /// getName - Get the name of this standard.
  const char *getName() const { return ShortName; }

  /// getDescription - Get the description of this standard.
  const char *getDescription() const { return Description; }

  /// Get the language that this standard describes.
  clang::Language getLanguage() const { return Language; }

  /// Get the version code for this language standard.
  std::optional<uint32_t> getVersion() const { return Version; }

  /// Language supports '//' comments.
  bool hasLineComments() const { return Flags & LineComment; }

  /// isC99 - Language is a superset of C99.
  bool isC99() const { return Flags & C99; }

  /// isC11 - Language is a superset of C11.
  bool isC11() const { return Flags & C11; }

  /// isC17 - Language is a superset of C17.
  bool isC17() const { return Flags & C17; }

  /// isC23 - Language is a superset of C23.
  bool isC23() const { return Flags & C23; }

  /// isC2y - Language is a superset of C2y.
  bool isC2y() const { return Flags & C2y; }

  /// isCPlusPlus - Language is a C++ variant.
  bool isCPlusPlus() const { return Flags & CPlusPlus; }

  /// isCPlusPlus11 - Language is a C++11 variant (or later).
  bool isCPlusPlus11() const { return Flags & CPlusPlus11; }

  /// isCPlusPlus14 - Language is a C++14 variant (or later).
  bool isCPlusPlus14() const { return Flags & CPlusPlus14; }

  /// isCPlusPlus17 - Language is a C++17 variant (or later).
  bool isCPlusPlus17() const { return Flags & CPlusPlus17; }

  /// isCPlusPlus20 - Language is a C++20 variant (or later).
  bool isCPlusPlus20() const { return Flags & CPlusPlus20; }

  /// isCPlusPlus23 - Language is a post-C++23 variant (or later).
  bool isCPlusPlus23() const { return Flags & CPlusPlus23; }

  /// isCPlusPlus26 - Language is a post-C++26 variant (or later).
  bool isCPlusPlus26() const { return Flags & CPlusPlus26; }

  /// isCPlusPlus2c - Language is a C++2c (expected 2029) variant (or later).
  bool isCPlusPlus29() const { return Flags & CPlusPlus29; }

  /// hasDigraphs - Language supports digraphs.
  bool hasDigraphs() const { return Flags & Digraphs; }

  /// hasRawStringLiterals - Language supports R"()" raw string literals.
  bool hasRawStringLiterals() const {
    // GCC supports raw string literals in C99 and later, but not in C++
    // before C++11.
    return isCPlusPlus11() || (!isCPlusPlus() && isC99() && isGNUMode());
  }

  /// allowLiteralDigitSeparator - Language supports literal digit seperator
  bool allowLiteralDigitSeparator() const { return isCPlusPlus14() || isC23(); }

  /// isGNUMode - Language includes GNU extensions.
  bool isGNUMode() const { return Flags & GNUMode; }

  /// hasHexFloats - Language supports hexadecimal float constants.
  bool hasHexFloats() const { return Flags & HexFloat; }

  /// isOpenCL - Language is a OpenCL variant.
  bool isOpenCL() const { return Flags & OpenCL; }

  static Kind getLangKind(StringRef Name);
  static Kind getHLSLLangKind(StringRef Name);
  static const LangStandard &getLangStandardForKind(Kind K);
  static const LangStandard *getLangStandardForName(StringRef Name);
};

LangStandard::Kind getDefaultLanguageStandard(clang::Language Lang,
                                              const llvm::Triple &T);

}  // end namespace clang

#endif
