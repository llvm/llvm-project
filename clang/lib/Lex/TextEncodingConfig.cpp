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
  case CA_ToSystemEncoding:
    return ToSystemEncodingConverter;
  case CA_ToExecEncoding:
    return ToExecEncodingConverter;
  default:
    return nullptr;
  }
}

std::error_code
TextEncodingConfig::setConvertersFromOptions(TextEncodingConfig &TEC,
                                             const clang::LangOptions &Opts,
                                             const clang::TargetInfo &TInfo) {
  using namespace llvm;
  TEC.InternalEncoding = "UTF-8";
  TEC.SystemEncoding = TInfo.getTriple().getDefaultNarrowTextEncoding();
  TEC.ExecEncoding =
      Opts.ExecEncoding.empty() ? TEC.InternalEncoding : Opts.ExecEncoding;

  // Create converter between internal and system encoding
  if (TEC.InternalEncoding != TEC.SystemEncoding) {
    ErrorOr<TextEncodingConverter> ErrorOrConverter =
        llvm::TextEncodingConverter::create(TEC.InternalEncoding,
                                            TEC.SystemEncoding);
    if (ErrorOrConverter) {
      TEC.ToSystemEncodingConverter =
          new TextEncodingConverter(std::move(*ErrorOrConverter));
    } else
      return ErrorOrConverter.getError();
  }

  // Create converter between internal and exec encoding specified
  // in fexec-charset option.
  if (TEC.InternalEncoding == TEC.ExecEncoding)
    return std::error_code();
  ErrorOr<TextEncodingConverter> ErrorOrConverter =
      llvm::TextEncodingConverter::create(TEC.InternalEncoding,
                                          TEC.ExecEncoding);
  if (ErrorOrConverter) {
    TEC.ToExecEncodingConverter =
        new TextEncodingConverter(std::move(*ErrorOrConverter));
  } else
    return ErrorOrConverter.getError();
  return std::error_code();
}
