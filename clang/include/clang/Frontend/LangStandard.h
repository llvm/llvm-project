//===--- LangStandard.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_LANGSTANDARD_H
#define LLVM_CLANG_FRONTEND_LANGSTANDARD_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

namespace frontend {

enum LangFeatures {
  LineComment = (1 << 0),
  C89 = (1 << 1),
  C99 = (1 << 2),
  C11 = (1 << 3),
  CPlusPlus = (1 << 4),
  CPlusPlus11 = (1 << 5),
  CPlusPlus14 = (1 << 6),
  CPlusPlus1z = (1 << 7),
  Digraphs = (1 << 8),
  GNUMode = (1 << 9),
  HexFloat = (1 << 10),
  ImplicitInt = (1 << 11),
  OpenCL = (1 << 12)
};

}

/// LangStandard - Information about the properties of a particular language
/// standard.
struct LangStandard {
  enum Kind {
#define LANGSTANDARD(id, name, desc, features) \
    lang_##id,
#include "clang/Frontend/LangStandards.def"
    lang_unspecified
  };

  const char *ShortName;
  const char *Description;
  unsigned Flags;

public:
  /// getName - Get the name of this standard.
  const char *getName() const { return ShortName; }

  /// getDescription - Get the description of this standard.
  const char *getDescription() const { return Description; }

  /// Language supports '//' comments.
  bool hasLineComments() const { return Flags & frontend::LineComment; }

  /// isC89 - Language is a superset of C89.
  bool isC89() const { return Flags & frontend::C89; }

  /// isC99 - Language is a superset of C99.
  bool isC99() const { return Flags & frontend::C99; }

  /// isC11 - Language is a superset of C11.
  bool isC11() const { return Flags & frontend::C11; }

  /// isCPlusPlus - Language is a C++ variant.
  bool isCPlusPlus() const { return Flags & frontend::CPlusPlus; }

  /// isCPlusPlus11 - Language is a C++11 variant (or later).
  bool isCPlusPlus11() const { return Flags & frontend::CPlusPlus11; }

  /// isCPlusPlus14 - Language is a C++14 variant (or later).
  bool isCPlusPlus14() const { return Flags & frontend::CPlusPlus14; }

  /// isCPlusPlus1z - Language is a C++17 variant (or later).
  bool isCPlusPlus1z() const { return Flags & frontend::CPlusPlus1z; }

  /// hasDigraphs - Language supports digraphs.
  bool hasDigraphs() const { return Flags & frontend::Digraphs; }

  /// isGNUMode - Language includes GNU extensions.
  bool isGNUMode() const { return Flags & frontend::GNUMode; }

  /// hasHexFloats - Language supports hexadecimal float constants.
  bool hasHexFloats() const { return Flags & frontend::HexFloat; }

  /// hasImplicitInt - Language allows variables to be typed as int implicitly.
  bool hasImplicitInt() const { return Flags & frontend::ImplicitInt; }

  /// isOpenCL - Language is a OpenCL variant.
  bool isOpenCL() const { return Flags & frontend::OpenCL; }

  static const LangStandard &getLangStandardForKind(Kind K);
  static const LangStandard *getLangStandardForName(StringRef Name);
};

}  // end namespace clang

#endif
