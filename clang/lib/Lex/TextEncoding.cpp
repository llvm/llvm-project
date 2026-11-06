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
  case CA_ToLiteralEncodingForAsm:
    // return null for non-z/OS targets
    return ToIBM1047Converter.get();
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

  // Create converters between internal and literal encoding specified
  // in fexec-charset option.
  if (TE.LiteralEncoding != UTF8) {
    ErrorOr<TextEncodingConverter> ErrorOrLiteralConverter =
        llvm::TextEncodingConverter::create(UTF8, TE.LiteralEncoding);
    if (ErrorOrLiteralConverter)
      TE.ToLiteralEncodingConverter = std::make_unique<TextEncodingConverter>(
          std::move(*ErrorOrLiteralConverter));
    else
      return ErrorOrLiteralConverter.getError();

    ErrorOrLiteralConverter =
        llvm::TextEncodingConverter::create(TE.LiteralEncoding, UTF8);
    if (ErrorOrLiteralConverter)
      TE.FromLiteralEncodingConverter = std::make_unique<TextEncodingConverter>(
          std::move(*ErrorOrLiteralConverter));
    else
      return ErrorOrLiteralConverter.getError();
  }

   TInfo.ExecStrConverter = TE.ToLiteralEncodingConverter.get();

  // The IBM-1047 converters are only needed on z/OS, where the system
  // (assembler/linker) default encoding is IBM-1047 rather than UTF-8.
  if (TInfo.getTriple().getOS() != llvm::Triple::ZOS)
    return std::error_code();

  // Create converters between UTF-8 and IBM-1047 for use when processing asm
  // string literals on z/OS, where escape sequences are IBM-1047 code points.
  const char *IBM1047 = "IBM-1047";
  ErrorOr<TextEncodingConverter> ErrorOrConverter =
      llvm::TextEncodingConverter::create(UTF8, IBM1047);
  assert(ErrorOrConverter &&
         "Failed to create TextEncodingConverter from UTF-8 to IBM-1047");
  TE.ToIBM1047Converter =
      std::make_unique<TextEncodingConverter>(std::move(*ErrorOrConverter));

  ErrorOrConverter = llvm::TextEncodingConverter::create(IBM1047, UTF8);
  assert(ErrorOrConverter &&
         "Failed to create TextEncodingConverter from IBM-1047 to UTF-8");
  TE.FromIBM1047Converter =
      std::make_unique<TextEncodingConverter>(std::move(*ErrorOrConverter));

  return std::error_code();
}
