//===----- FormatStringParsing.h - Format String Parsing --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides some shared functions between printf and scanf format string
// parsing code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_ANALYSIS_FORMATSTRINGPARSING_H
#define LLVM_CLANG_LIB_ANALYSIS_FORMATSTRINGPARSING_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/FormatString.h"
#include "clang/AST/Type.h"

namespace clang {

class LangOptions;

template <typename T> class UpdateOnReturn {
  T &ValueToUpdate;
  const T &ValueToCopy;

public:
  UpdateOnReturn(T &valueToUpdate, const T &valueToCopy)
      : ValueToUpdate(valueToUpdate), ValueToCopy(valueToCopy) {}

  ~UpdateOnReturn() { ValueToUpdate = ValueToCopy; }
};

namespace analyze_format_string {

class FormatStringParser {
protected:
  FormatStringParser(FormatStringHandler &H, const LangOptions &LO,
                     const TargetInfo &Target, bool IsScanf)
      : H(H), LO(LO), Target(Target), IsScanf(IsScanf) {}

  OptionalAmount parseAmount(const char *&Beg, const char *E) const;

  OptionalAmount parseNonPositionAmount(const char *&Beg, const char *E,
                                        unsigned &argIndex) const;

  OptionalAmount parsePositionAmount(const char *Start, const char *&Beg,
                                     const char *E, PositionContext p) const;

  bool parseFieldWidth(FormatSpecifier &CS, const char *Start, const char *&Beg,
                       const char *E, unsigned *argIndex) const;

  bool parseArgPosition(FormatSpecifier &CS, const char *Start,
                        const char *&Beg, const char *E) const;

  bool parseVectorModifier(FormatSpecifier &FS, const char *&Beg,
                           const char *E) const;

  /// Returns true if a LengthModifier was parsed and installed in the
  /// FormatSpecifier& argument, and false otherwise.
  bool parseLengthModifier(FormatSpecifier &FS, const char *&Beg,
                           const char *E) const;

  /// Returns true if the invalid specifier in \p SpecifierBegin is a UTF-8
  /// string; check that it won't go further than \p FmtStrEnd and write
  /// up the total size in \p Len.
  bool parseUTF8InvalidSpecifier(const char *SpecifierBegin,
                                 const char *FmtStrEnd, unsigned &Len) const;

  FormatStringHandler &H;
  const LangOptions &LO;
  const TargetInfo &Target;
  bool IsScanf;

private:
  bool parseWidthModifier(const char *&I, const char *E, unsigned &BitWidth,
                          unsigned &ModifierLength) const;
};

template <typename T> class SpecifierResult {
  T FS;
  const char *Start;
  bool Stop;

public:
  SpecifierResult(bool stop = false) : Start(nullptr), Stop(stop) {}
  SpecifierResult(const char *start, const T &fs)
      : FS(fs), Start(start), Stop(false) {}

  const char *getStart() const { return Start; }
  bool shouldStop() const { return Stop; }
  bool hasValue() const { return Start != nullptr; }
  const T &getValue() const {
    assert(hasValue());
    return FS;
  }
  const T &getValue() { return FS; }
};

} // namespace analyze_format_string
} // namespace clang

#endif
