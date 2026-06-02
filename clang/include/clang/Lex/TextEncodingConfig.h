//===-- clang/Lex/TextEncodingConfig.h - Text Conversion Config -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_TEXTENCODINGCONFIG_H
#define LLVM_CLANG_LEX_TEXTENCODINGCONFIG_H

#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TextEncoding.h"

enum ConversionAction { CA_NoConversion };

class TextEncodingConfig {
public:
  llvm::TextEncodingConverter *getConverter(ConversionAction Action) const;
};

#endif
