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
LiteralConverter::getConverter(const char *Codepage) {
  auto Iter = TextEncodingConverters.find(Codepage);
  if (Iter != TextEncodingConverters.end())
    return &Iter->second;
  return nullptr;
}

llvm::TextEncodingConverter *
LiteralConverter::getConverter(ConversionAction Action) {
  StringRef CodePage;
  if (Action == ToSystemCharset)
    CodePage = SystemCharset;
  else if (Action == ToExecCharset)
    CodePage = ExecCharset;
  else
    CodePage = InternalCharset;
  return getConverter(CodePage.data());
}

llvm::TextEncodingConverter *
LiteralConverter::createAndInsertCharConverter(const char *To) {
  const char *From = InternalCharset.data();
  llvm::TextEncodingConverter *Converter = getConverter(To);
  if (Converter)
    return Converter;

  ErrorOr<TextEncodingConverter> ErrorOrConverter =
      llvm::TextEncodingConverter::create(From, To);
  if (!ErrorOrConverter)
    return nullptr;
  TextEncodingConverters.insert_or_assign(StringRef(To),
                                          std::move(*ErrorOrConverter));
  return getConverter(To);
}

void LiteralConverter::setConvertersFromOptions(
    const clang::LangOptions &Opts, const clang::TargetInfo &TInfo,
    clang::DiagnosticsEngine &Diags) {
  using namespace llvm;
  SystemCharset = TInfo.getTriple().getSystemCharset();
  InternalCharset = "UTF-8";
  ExecCharset = Opts.ExecCharset.empty() ? InternalCharset : Opts.ExecCharset;
  // Create converter between internal and system charset
  if (InternalCharset != SystemCharset)
    createAndInsertCharConverter(SystemCharset.data());

  // Create converter between internal and exec charset specified
  // in fexec-charset option.
  if (InternalCharset == ExecCharset)
    return;
  if (!createAndInsertCharConverter(ExecCharset.data())) {
    Diags.Report(clang::diag::err_drv_invalid_value)
        << "-fexec-charset" << ExecCharset;
  }
}
