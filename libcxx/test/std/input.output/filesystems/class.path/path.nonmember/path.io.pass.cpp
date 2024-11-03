//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-localization
// UNSUPPORTED: availability-filesystem-missing

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

#include "filesystem_include.h"
#include <type_traits>
#include <sstream>
#include <cassert>
#include <iostream>

#include "count_new.h"
#include "make_string.h"
#include "test_iterators.h"
#include "test_macros.h"

MultiStringType InStr =  MKSTR("abcdefg/\"hijklmnop\"/qrstuvwxyz/123456789");
MultiStringType OutStr = MKSTR("\"abcdefg/\\\"hijklmnop\\\"/qrstuvwxyz/123456789\"");

template <class CharT>
void doIOTest() {
  using namespace fs;
  using Ptr = const CharT*;
  using StrStream = std::basic_stringstream<CharT>;
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
    assert(p_in.native() == (const path::value_type*)InStr);
    assert(&ret == &ss);
  }
}

namespace impl {
using namespace fs;

template <class Stream, class Tp, class = decltype(std::declval<Stream&>() << std::declval<Tp&>())>
std::true_type is_ostreamable_imp(int);

template <class Stream, class Tp>
std::false_type is_ostreamable_imp(long);

template <class Stream, class Tp, class = decltype(std::declval<Stream&>() >> std::declval<Tp&>())>
std::true_type is_istreamable_imp(int);

template <class Stream, class Tp>
std::false_type is_istreamable_imp(long);


} // namespace impl

template <class Stream, class Tp>
struct is_ostreamable : decltype(impl::is_ostreamable_imp<Stream, Tp>(0)) {};
template <class Stream, class Tp>
struct is_istreamable : decltype(impl::is_istreamable_imp<Stream, Tp>(0)) {};

void test_LWG2989() {
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  static_assert(!is_ostreamable<decltype(std::cout), std::wstring>::value, "");
  static_assert(!is_ostreamable<decltype(std::wcout), std::string>::value, "");
  static_assert(!is_istreamable<decltype(std::cin), std::wstring>::value, "");
  static_assert(!is_istreamable<decltype(std::wcin), std::string>::value, "");
#endif
}

int main(int, char**) {
  doIOTest<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  doIOTest<wchar_t>();
#endif
  // TODO(var-const): uncomment when it becomes possible to instantiate a `basic_ostream` object with a sized character
  // type (see https://llvm.org/PR53119).
  //doIOTest<char16_t>();
  //doIOTest<char32_t>();
  test_LWG2989();

  return 0;
}
