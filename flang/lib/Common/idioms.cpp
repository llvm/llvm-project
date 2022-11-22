//===-- lib/Common/idioms.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/idioms.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <regex>

namespace Fortran::common {

[[noreturn]] void die(const char *msg, ...) {
  va_list ap;
  va_start(ap, msg);
  std::fputs("\nfatal internal error: ", stderr);
  std::vfprintf(stderr, msg, ap);
  va_end(ap);
  fputc('\n', stderr);
  std::abort();
}

// Converts the comma separated list of enumerators into tokens which are then
// stored into the provided array of strings. This is intended for use from the
// expansion of ENUM_CLASS.
void BuildIndexToString(
    const char *commaSeparated, std::string enumNames[], int enumSize) {
  std::string input(commaSeparated);
  std::regex reg("\\s*,\\s*");

  std::sregex_token_iterator iter(input.begin(), input.end(), reg, -1);
  std::sregex_token_iterator end;
  int index = 0;
  while (iter != end) {
    enumNames[index] = *iter;
    iter++;
    index++;
  }
  CHECK(index == enumSize);
}
} // namespace Fortran::common
