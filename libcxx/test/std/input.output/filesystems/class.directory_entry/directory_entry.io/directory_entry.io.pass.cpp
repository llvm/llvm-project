//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-localization

// <filesystem>
//
// class directory_entry
//
// template<class charT, class traits>
//   friend basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const directory_entry& d);

#include <filesystem>
#include <cassert>
#include <sstream>

#include "test_macros.h"
#include "make_string.h"
namespace fs = std::filesystem;

MultiStringType InStr =  MKSTR("abcdefg/\"hijklmnop\"/qrstuvwxyz/123456789");
MultiStringType OutStr = MKSTR("\"abcdefg/\\\"hijklmnop\\\"/qrstuvwxyz/123456789\"");

template <class CharT>
void TestOutput() {
  const char* input = static_cast<const char*>(InStr);
  const CharT* expected_output = static_cast<const CharT*>(OutStr);
  const fs::directory_entry dir = fs::directory_entry(fs::path(input));
  std::basic_stringstream<CharT> stream;

  auto& result = stream << dir;
  assert(stream.str() == expected_output);
  assert(&result == &stream);
}

int main(int, char**) {
  TestOutput<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  TestOutput<wchar_t>();
#endif
  // TODO(var-const): uncomment when it becomes possible to instantiate a `basic_ostream` object with a sized character
  // type (see https://llvm.org/PR53119).
  //TestOutput<char8_t>();
  //TestOutput<char16_t>();
  //TestOutput<char32_t>();

  return 0;
}
