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
  size_t BaseLetterPos;     // Position of the base letter.
  size_t DotPos;            // Position of the decimal/hexadecimal point.
  size_t ExponentLetterPos; // Position of the exponent letter.
  size_t SuffixPos;         // Starting position of the suffix.

  NumericLiteralInfo(size_t BaseLetterPos = llvm::StringRef::npos,
                     size_t DotPos = llvm::StringRef::npos,
                     size_t ExponentLetterPos = llvm::StringRef::npos,
                     size_t SuffixPos = llvm::StringRef::npos)
      : BaseLetterPos(BaseLetterPos), DotPos(DotPos),
        ExponentLetterPos(ExponentLetterPos), SuffixPos(SuffixPos) {}

  NumericLiteralInfo(llvm::StringRef Text, char Separator);

  bool operator==(const NumericLiteralInfo &R) const {
    return BaseLetterPos == R.BaseLetterPos && DotPos == R.DotPos &&
           ExponentLetterPos == R.ExponentLetterPos && SuffixPos == R.SuffixPos;
  }
};

} // end namespace format
} // end namespace clang

#endif
