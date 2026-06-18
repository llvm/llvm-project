//===-- clang/Lex/TextEncoding.h - Text Conversion Config -*- C++ -*-===//
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

enum ConversionAction { CA_NoConversion, CA_FromInputEncoding };

class TextEncoding {
std::unique_ptr<llvm::TextEncodingConverter> FromInputEncodingConverter;

public:
  llvm::TextEncodingConverter *getConverter(ConversionAction Action) const;
};

#endif
