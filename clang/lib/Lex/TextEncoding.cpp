//===--- TextEncoding.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/TextEncoding.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "llvm/Support/TextEncoding.h"

using namespace clang;

llvm::TextEncodingConverter *
TextEncoding::getConverter(ConversionAction Action) const {
  switch (Action) {
  case CA_ToLiteralEncoding:
    return ToLiteralEncodingConverter.get();
  case CA_ToSystemEncoding:
    return ToSystemEncodingConverter.get();
  default:
    return nullptr;
  }
}

std::error_code
TextEncoding::setConvertersFromOptions(TextEncoding &TE,
                                       const clang::LangOptions &Opts,
                                       clang::TargetInfo &TInfo) {
  using namespace llvm;

  const char *UTF8 = "UTF-8";
  TE.LiteralEncoding =
      Opts.LiteralEncoding.empty() ? UTF8 : Opts.LiteralEncoding.c_str();

  if (TE.LiteralEncoding != UTF8) {
    ErrorOr<TextEncodingConverter> ErrorOrLiteralConverter =
        llvm::TextEncodingConverter::create(UTF8, TE.LiteralEncoding);
    if (ErrorOrLiteralConverter)
      TE.ToLiteralEncodingConverter = std::make_unique<TextEncodingConverter>(
          std::move(*ErrorOrLiteralConverter));
    else
      return ErrorOrLiteralConverter.getError();
  }

  if (TInfo.getDefaultOrdinaryLiteralEncoding() == UTF8)
    return std::error_code();

  // Create converter between internal and default ordinary encoding for the
  // target
  ErrorOr<TextEncodingConverter> ErrorOrConverter =
      llvm::TextEncodingConverter::create(
          UTF8, TInfo.getDefaultOrdinaryLiteralEncoding());
  if (ErrorOrConverter)
    TE.ToSystemEncodingConverter =
        std::make_unique<TextEncodingConverter>(std::move(*ErrorOrConverter));
  else
    return ErrorOrConverter.getError();

  ErrorOrConverter = llvm::TextEncodingConverter::create(
      TInfo.getDefaultOrdinaryLiteralEncoding(), UTF8);

  if (ErrorOrConverter)
    TInfo.FromSystemEncodingConverter =
        std::make_unique<TextEncodingConverter>(std::move(*ErrorOrConverter));
  else
    return ErrorOrConverter.getError();

  return std::error_code();
}
