//===-- clang/Lex/TextEncodingConfig.h - Text Conversion Config -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_TEXTENCODINGCONFIG_H
#define LLVM_CLANG_LEX_TEXTENCODINGCONFIG_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TextEncoding.h"

enum ConversionAction { CA_NoConversion, CA_FromInputEncoding };

class TextEncodingConfig {
  llvm::StringRef InputEncoding;
  std::string FileTagEncoding;
  std::unique_ptr<llvm::TextEncodingConverter> FromInputEncodingConverter;

public:
  llvm::TextEncodingConverter *getConverter(ConversionAction Action) const;
  static std::unique_ptr<llvm::TextEncodingConverter>
#ifdef __MVS__      
  createInputConverterFromFiletag(__ccsid_t Ccsid,
                                   clang::DiagnosticsEngine &Diags);
#endif  
  static std::error_code
  setFromInputConverter(TextEncodingConfig &TEC,
                        std::unique_ptr<llvm::TextEncodingConverter> Converter);
  llvm::StringRef getInputEncoding() { return InputEncoding; }
};

#endif
