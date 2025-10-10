//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <filesystem>

// class path

// template <class charT, class traits>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& os, const path& p);
//
// template <class charT, class traits>
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is, path& p)
//

// TODO(EricWF) This test fails because "std::quoted" fails to compile
// for char16_t and char32_t types. Combine with path.io.pass.cpp when this
// passes.
// XFAIL: *

#include <filesystem>
#include <type_traits>
#include <sstream>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;

MultiStringType InStr =  MKSTR("abcdefg/\"hijklmnop\"/qrstuvwxyz/123456789");
MultiStringType OutStr = MKSTR("\"abcdefg/\\\"hijklmnop\\\"/qrstuvwxyz/123456789\"");

template <class CharT>
void doIOTest() {
  using namespace fs;
  using Ptr = const CharT*;
  using StrStream = std::basic_stringstream<CharT>;
  const char* const InCStr = InStr;
  const Ptr E = OutStr;
  const path p((const char*)InStr);
  StrStream ss;
  { // test output
    auto& ret = (ss << p);
    assert(ss.str() == E);
    assert(&ret == &ss);
  }
  { // test input
    path p_in;
    auto& ret = ss >> p_in;
    assert(p_in.native() == (const char*)InStr);
    assert(&ret == &ss);
  }
}


int main(int, char**) {
  doIOTest<char16_t>();
  doIOTest<char32_t>();

  return 0;
}
