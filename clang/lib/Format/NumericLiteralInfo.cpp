//===--- NumericLiteralInfo.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the functionality of getting information about a
/// numeric literal string, including 0-based positions of the base letter, the
/// decimal/hexadecimal point, the exponent letter, and the suffix, or npos if
/// absent.
///
//===----------------------------------------------------------------------===//

#include "NumericLiteralInfo.h"
#include "llvm/ADT/StringExtras.h"

namespace clang {
namespace format {

using namespace llvm;

NumericLiteralInfo::NumericLiteralInfo(StringRef Text, char Separator) {
  if (Text.size() < 2)
    return;

  bool IsHex = false;
  if (Text[0] == '0') {
    switch (Text[1]) {
    case 'x':
    case 'X':
      IsHex = true;
      [[fallthrough]];
    case 'b':
    case 'B':
    case 'o': // JavaScript octal.
    case 'O':
      BaseLetterPos = 1; // e.g. 0xF
      break;
    }
  }

  DotPos = Text.find('.', BaseLetterPos + 1); // e.g. 0x.1 or .1

  // e.g. 1.e2 or 0xFp2
  const auto Pos = DotPos != StringRef::npos ? DotPos + 1 : BaseLetterPos + 2;

  ExponentLetterPos =
      // Trim C++ user-defined suffix as in `1_Pa`.
      (Separator == '\'' ? Text.take_front(Text.find('_')) : Text)
          .find_insensitive(IsHex ? 'p' : 'e', Pos);

  const bool HasExponent = ExponentLetterPos != StringRef::npos;
  SuffixPos = Text.find_if_not(
      [&](char C) {
        return (HasExponent || !IsHex ? isDigit : isHexDigit)(C) ||
               C == Separator;
      },
      HasExponent ? ExponentLetterPos + 2 : Pos); // e.g. 1e-2f
}

} // namespace format
} // namespace clang
