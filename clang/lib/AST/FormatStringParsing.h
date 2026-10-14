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
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/TextEncoding.h"

namespace clang {

class LangOptions;

/// Creates a TextEncodingConverter for format string character comparisons.
/// On z/OS format strings are IBM-1047 encoded, so each byte must be converted
/// to UTF-8 before comparing against ASCII specifier characters.
/// On all other targets a no-op converter is returned.
/// Create this once per format string parse and pass it through.
inline llvm::TextEncodingConverter
makeFormatStrConverter(const TargetInfo &Target) {
  auto MaybeConverter =
      Target.getTriple().isOSzOS()
          ? llvm::TextEncodingConverter::create(llvm::TextEncoding::IBM1047,
                                               llvm::TextEncoding::UTF8)
          : llvm::TextEncodingConverter::createNoopConverter();
  assert(MaybeConverter && "Failed to create format string converter");
  return std::move(*MaybeConverter);
}

template <typename T> class UpdateOnReturn {
  T &ValueToUpdate;
  const T &ValueToCopy;

public:
  UpdateOnReturn(T &valueToUpdate, const T &valueToCopy)
      : ValueToUpdate(valueToUpdate), ValueToCopy(valueToCopy) {}

  ~UpdateOnReturn() { ValueToUpdate = ValueToCopy; }
};

namespace analyze_format_string {

OptionalAmount ParseAmount(const char *&Beg, const char *E,
                           const llvm::TextEncodingConverter &Conv);

OptionalAmount ParseNonPositionAmount(const char *&Beg, const char *E,
                                      unsigned &argIndex,
                                      const llvm::TextEncodingConverter &Conv);

OptionalAmount ParsePositionAmount(FormatStringHandler &H, const char *Start,
                                   const char *&Beg, const char *E,
                                   PositionContext p,
                                   const llvm::TextEncodingConverter &Conv);

bool ParseFieldWidth(FormatStringHandler &H, FormatSpecifier &CS,
                     const char *Start, const char *&Beg, const char *E,
                     unsigned *argIndex,
                     const llvm::TextEncodingConverter &Conv);

bool ParseArgPosition(FormatStringHandler &H, FormatSpecifier &CS,
                      const char *Start, const char *&Beg, const char *E,
                      const llvm::TextEncodingConverter &Conv);

bool ParseVectorModifier(FormatStringHandler &H, FormatSpecifier &FS,
                         const char *&Beg, const char *E, const LangOptions &LO,
                         const llvm::TextEncodingConverter &Conv);

/// Returns true if a LengthModifier was parsed and installed in the
/// FormatSpecifier& argument, and false otherwise.
bool ParseLengthModifier(FormatSpecifier &FS, const char *&Beg, const char *E,
                         const LangOptions &LO,
                         const llvm::TextEncodingConverter &Conv,
                         bool IsScanf = false);

/// Returns true if the invalid specifier in \p SpecifierBegin is a UTF-8
/// string; check that it won't go further than \p FmtStrEnd and write
/// up the total size in \p Len.
bool ParseUTF8InvalidSpecifier(const char *SpecifierBegin,
                               const char *FmtStrEnd, unsigned &Len);

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
