//===-- clang/Lex/TextEncoding.h - Text Encoding Conversion ------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_TEXTENCODING_H
#define LLVM_CLANG_LEX_TEXTENCODING_H

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class TextEncodingConverter;
} // namespace llvm

namespace clang {
enum ConversionAction {
  CA_NoConversion,
  CA_ToLiteralEncodingForAsm, // IBM-1047 on z/OS, UTF-8 on other platforms
  CA_ToLiteralEncoding
};

class TextEncoding {
  llvm::StringRef LiteralEncoding;
  std::unique_ptr<llvm::TextEncodingConverter> ToLiteralEncodingConverter;

  // Only non-null on z/OS, where the system default encoding is IBM-1047.
  // This converts UTF-8 to IBM-1047 for asm string literals so that
  // octal/hex escape sequences are interpreted as IBM-1047 code points,
  // regardless of -fexec-charset.
  std::unique_ptr<llvm::TextEncodingConverter> ToIBM1047Converter;
  std::unique_ptr<llvm::TextEncodingConverter> FromIBM1047Converter;

public:
  llvm::TextEncodingConverter *getConverter(ConversionAction Action) const;

  /// Returns the converter from IBM-1047 to UTF-8, or nullptr if not on z/OS.
  llvm::TextEncodingConverter *getFromIBM1047Converter() const {
    return FromIBM1047Converter.get();
  }

  static std::error_code
  setConvertersFromOptions(TextEncoding &TE, const clang::LangOptions &Opts,
                           clang::TargetInfo &TInfo);

  llvm::StringRef getLiteralEncoding() { return LiteralEncoding; }
};
} // namespace clang
#endif
