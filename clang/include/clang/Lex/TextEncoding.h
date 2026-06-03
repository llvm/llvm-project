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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TextEncoding.h"

enum ConversionAction { CA_NoConversion, CA_ToExecEncoding };

class TextEncoding {
  llvm::StringRef ExecEncoding;
  llvm::TextEncodingConverter *ToExecEncodingConverter = nullptr;

public:
  llvm::TextEncodingConverter *getConverter(ConversionAction Action) const;
  static std::error_code
  setConvertersFromOptions(TextEncoding &TE, const clang::LangOptions &Opts);

  llvm::StringRef getExecEncoding() { return ExecEncoding; }
};

#endif
