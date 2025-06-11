//===--- clang/Lex/LiteralConverter.h - Translator for Literals -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_LITERALCONVERTER_H
#define LLVM_CLANG_LEX_LITERALCONVERTER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TextEncoding.h"

enum ConversionAction { NoConversion, ToSystemEncoding, ToExecEncoding };

class LiteralConverter {
  llvm::StringRef InternalEncoding;
  llvm::StringRef SystemEncoding;
  llvm::StringRef ExecEncoding;
  llvm::TextEncodingConverter *ToSystemEncodingConverter;
  llvm::TextEncodingConverter *ToExecEncodingConverter;

public:
  llvm::TextEncodingConverter *getConverter(ConversionAction Action);
  void setConvertersFromOptions(const clang::LangOptions &Opts,
                                const clang::TargetInfo &TInfo,
                                clang::DiagnosticsEngine &Diags);
};

#endif
