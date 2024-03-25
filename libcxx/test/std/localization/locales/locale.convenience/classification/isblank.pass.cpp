//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> bool isblank (charT c, const locale& loc);

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  std::locale l;
  assert(std::isblank(' ', l));
  assert(!std::isblank('<', l));
  assert(!std::isblank('\x8', l));
  assert(!std::isblank('A', l));
  assert(!std::isblank('a', l));
  assert(!std::isblank('z', l));
  assert(!std::isblank('3', l));
  assert(!std::isblank('.', l));
  assert(!std::isblank('f', l));
  assert(!std::isblank('9', l));
  assert(!std::isblank('+', l));

  return 0;
}
