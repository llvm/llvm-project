//===--- TextEncodingConfig.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/TextEncodingConfig.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "llvm/Support/AutoConvert.h"

using namespace llvm;

llvm::TextEncodingConverter *
TextEncodingConfig::getConverter(ConversionAction Action) const {
  switch (Action) {
  case CA_FromInputEncoding:
    return FromInputEncodingConverter.get();
  default:
    return nullptr;
  }
}

std::unique_ptr<llvm::TextEncodingConverter>
TextEncodingConfig::createInputConverterFromFiletag(
    __ccsid_t Ccsid, clang::DiagnosticsEngine &Diags) {
  using namespace llvm;

  std::string FileTagEncoding = std::to_string(Ccsid);

  llvm::StringRef InputEncoding = FileTagEncoding;
  const char *UTF8 = "UTF-8";

  // Create a converter between the input and internal encodings
  if (llvm::TextEncodingConverter::getKnownEncoding(InputEncoding) !=
      llvm::TextEncodingConverter::getKnownEncoding(UTF8)) {
    ErrorOr<TextEncodingConverter> ErrorOrConverter =
        llvm::TextEncodingConverter::create(InputEncoding, UTF8);
    if (!ErrorOrConverter) {
      Diags.Report(clang::diag::err_drv_invalid_value)
          << "filetag" << InputEncoding;
      return nullptr;
    } else {
      return std::make_unique<llvm::TextEncodingConverter>(
          std::move(*ErrorOrConverter));
    }
  }
  return nullptr;
}

std::error_code
TextEncodingConfig::setFromInputConverter(
    TextEncodingConfig &TEC,
    std::unique_ptr<llvm::TextEncodingConverter> Converter) {
  TEC.FromInputEncodingConverter = std::move(Converter);
  return std::error_code();
}
