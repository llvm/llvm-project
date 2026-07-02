//===--- TextEncoding.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/TextEncoding.h"
#include "clang/Basic/DiagnosticDriver.h"

using namespace clang;

llvm::TextEncodingConverter *
TextEncoding::getConverter(ConversionAction Action) const {
  switch (Action) {
  case CA_ToLiteralEncoding:
    return ToLiteralEncodingConverter;
  default:
    return nullptr;
  }
}

std::error_code
TextEncoding::setConvertersFromOptions(TextEncoding &TE,
                                       const clang::LangOptions &Opts) {
  using namespace llvm;

  const char *UTF8 = "UTF-8";
  TE.LiteralEncoding =
      Opts.LiteralEncoding.empty() ? UTF8 : Opts.LiteralEncoding.c_str();

  // Create converter between internal and literal encoding specified
  // in fexec-charset option.
  if (TE.LiteralEncoding == UTF8)
    return std::error_code();
  ErrorOr<TextEncodingConverter> ErrorOrConverter =
      llvm::TextEncodingConverter::create(UTF8, TE.LiteralEncoding);
  if (ErrorOrConverter)
    TE.ToLiteralEncodingConverter =
        new TextEncodingConverter(std::move(*ErrorOrConverter));
  else
    return ErrorOrConverter.getError();
  return std::error_code();
}
