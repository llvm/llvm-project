//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// typedef match_results<wstring::const_iterator>   wsmatch;

// XFAIL: no-wide-characters

#include <regex>
#include <type_traits>
#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<std::match_results<std::wstring::const_iterator>, std::wsmatch>::value), "");

  return 0;
}
