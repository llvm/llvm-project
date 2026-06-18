//===--- TextEncoding.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/TextEncoding.h"
#include "clang/Basic/DiagnosticDriver.h"

llvm::TextEncodingConverter *
TextEncoding::getConverter(ConversionAction Action) const {
  switch (Action) {
  case CA_FromInputEncoding:
    return FromInputEncodingConverter.get();
  default:
    return nullptr;
  }
}
