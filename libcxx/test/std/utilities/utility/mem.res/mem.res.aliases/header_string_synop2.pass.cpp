//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// <string>

// namespace std::pmr {
//
// typedef ... string
// typedef ... u16string
// typedef ... u32string
// typedef ... wstring
//
// } // namespace std::pmr

#include <string>

#include "test_macros.h"

int main(int, char**) {
  {
    // Check that std::pmr::string is usable without <memory_resource>.
    std::pmr::string s;
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    std::pmr::wstring ws;
#endif
    std::pmr::u16string u16s;
    std::pmr::u32string u32s;
  }

  return 0;
}
