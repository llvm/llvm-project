//===--- NumericLiteralInfo.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_NUMERICLITERALINFO_H
#define LLVM_CLANG_LIB_FORMAT_NUMERICLITERALINFO_H

#include "llvm/ADT/StringRef.h"

namespace clang {
namespace format {

struct NumericLiteralInfo {
  size_t BaseLetterPos = llvm::StringRef::npos;     // as in 0b1, 0xF, etc.
  size_t DotPos = llvm::StringRef::npos;            // pos of decimal/hex point
  size_t ExponentLetterPos = llvm::StringRef::npos; // as in 9e9 and 0xFp9
  size_t SuffixPos = llvm::StringRef::npos;         // starting pos of suffix

  NumericLiteralInfo(llvm::StringRef Text, char Separator = '\'');
};

} // end namespace format
} // end namespace clang

#endif
