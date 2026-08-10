//===-- lib/Parser/char-set.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/char-set.h"

namespace Fortran::parser {

std::string SetOfChars::ToString() const {
  std::string result;
  SetOfChars set{*this};
  for (char ch{' '}; !set.empty(); ++ch) {
    if (set.Has(ch)) {
      set = set.Difference(ch);
      result += ch;
    }
  }
  return result;
}
} // namespace Fortran::parser
