//===--- TextEncodingConfig.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/TextEncodingConfig.h"
#include "clang/Basic/DiagnosticDriver.h"

using namespace llvm;

llvm::TextEncodingConverter *
TextEncodingConfig::getConverter(ConversionAction Action) const {
  switch (Action) {
  case CA_ToExecEncoding:
    return ToExecEncodingConverter;
  default:
    return nullptr;
  }
}

std::error_code
TextEncodingConfig::setConvertersFromOptions(TextEncodingConfig &TEC,
                                             const clang::LangOptions &Opts,
                                             clang::TargetInfo &TInfo) {
  using namespace llvm;

  const char *UTF8 = "UTF-8";
  TEC.ExecEncoding =
      Opts.ExecEncoding.empty() ? UTF8 : Opts.ExecEncoding.c_str();

  // Create converter between internal and exec encoding specified
  // in fexec-charset option.
  if (TEC.ExecEncoding == UTF8)
    return std::error_code();
  ErrorOr<TextEncodingConverter> ErrorOrConverter =
      llvm::TextEncodingConverter::create(UTF8, TEC.ExecEncoding);
  if (ErrorOrConverter)
    TEC.ToExecEncodingConverter =
        new TextEncodingConverter(std::move(*ErrorOrConverter));
  else
    return ErrorOrConverter.getError();

  ErrorOrConverter = llvm::TextEncodingConverter::create(TEC.SystemEncoding,
                                                         TEC.InternalEncoding);

  if (ErrorOrConverter)
    TInfo.FormatStrConverter =
        new TextEncodingConverter(std::move(*ErrorOrConverter));

  return std::error_code();
}
