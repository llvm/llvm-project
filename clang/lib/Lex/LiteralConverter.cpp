//===--- LiteralConverter.cpp - Translator for String Literals -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/LiteralConverter.h"
#include "clang/Basic/DiagnosticDriver.h"

using namespace llvm;

llvm::TextEncodingConverter *
LiteralConverter::getConverter(ConversionAction Action) {
  if (Action == CA_ToSystemEncoding)
    return ToSystemEncodingConverter;
  else if (Action == CA_ToExecEncoding)
    return ToExecEncodingConverter;
  else
    return nullptr;
}

std::error_code
LiteralConverter::setConvertersFromOptions(LiteralConverter &LiteralConv,
                                           const clang::LangOptions &Opts,
                                           const clang::TargetInfo &TInfo) {
  using namespace llvm;
  LiteralConv.InternalEncoding = "UTF-8";
  LiteralConv.SystemEncoding = TInfo.getTriple().getDefaultNarrowTextEncoding();
  LiteralConv.ExecEncoding = Opts.ExecEncoding.empty()
                                 ? LiteralConv.InternalEncoding
                                 : Opts.ExecEncoding;

  // Create converter between internal and system encoding
  if (LiteralConv.InternalEncoding != LiteralConv.SystemEncoding) {
    ErrorOr<TextEncodingConverter> ErrorOrConverter =
        llvm::TextEncodingConverter::create(LiteralConv.InternalEncoding,
                                            LiteralConv.SystemEncoding);
    if (ErrorOrConverter) {
      LiteralConv.ToSystemEncodingConverter =
          new TextEncodingConverter(std::move(*ErrorOrConverter));
    } else
      return ErrorOrConverter.getError();
  }

  // Create converter between internal and exec encoding specified
  // in fexec-charset option.
  if (LiteralConv.InternalEncoding == LiteralConv.ExecEncoding)
    return std::error_code();
  ErrorOr<TextEncodingConverter> ErrorOrConverter =
      llvm::TextEncodingConverter::create(LiteralConv.InternalEncoding,
                                          LiteralConv.ExecEncoding);
  if (ErrorOrConverter) {
    LiteralConv.ToExecEncodingConverter =
        new TextEncodingConverter(std::move(*ErrorOrConverter));
  } else
    return ErrorOrConverter.getError();
  return std::error_code();
}
