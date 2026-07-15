//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <fstream>

// plate <class charT, class traits = char_traits<charT> >
// class basic_fstream

// template<class T>
// explicit basic_fstream(const T& s, ios_base::openmode mode = ios_base::in); // Since C++17
// Constraints: is_same_v<T, filesystem::path> is true

#include <fstream>
#include <filesystem>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"
#include "platform_support.h"
#include "operator_hijacker.h"

namespace fs = std::filesystem;

template <class CharT>
constexpr bool test_non_convert_to_path() {
  // String types
  static_assert(!std::is_constructible_v<std::fstream, std::basic_string_view<CharT>>);
  static_assert(!std::is_constructible_v<std::fstream, const std::basic_string_view<CharT>>);

  // Char* pointers
  if constexpr (!std::is_same_v<CharT, char> && !std::is_same_v<CharT, fs::path::value_type>)
    static_assert(!std::is_constructible_v<std::fstream, const CharT*>);

  // Iterators
  static_assert(!std::is_convertible_v<std::fstream, cpp17_input_iterator<const CharT*>>);

  return true;
}

static_assert(test_non_convert_to_path<char>());

#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && !defined(TEST_HAS_OPEN_WITH_WCHAR)
static_assert(test_non_convert_to_path<wchar_t>());
#endif // !TEST_HAS_NO_WIDE_CHARACTERS && !TEST_HAS_OPEN_WITH_WCHAR

#ifndef TEST_HAS_NO_CHAR8_T
static_assert(test_non_convert_to_path<char8_t>());
#endif // TEST_HAS_NO_CHAR8_T

static_assert(test_non_convert_to_path<char16_t>());
static_assert(test_non_convert_to_path<char32_t>());

int main(int, char**) {
  fs::path p = get_temp_file_name();
  {
    std::fstream fs(p, std::ios_base::in | std::ios_base::out |
                           std::ios_base::trunc);
    double x = 0;
    fs << 3.25;
    fs.seekg(0);
    fs >> x;
    assert(x == 3.25);
  }
  std::remove(p.string().c_str());

  {
    std::basic_fstream<char, operator_hijacker_char_traits<char> > fs(
        p, std::ios_base::in | std::ios_base::out | std::ios_base::trunc);
    std::basic_string<char, operator_hijacker_char_traits<char> > x;
    fs << "3.25";
    fs.seekg(0);
    fs >> x;
    assert(x == "3.25");
  }
  std::remove(p.string().c_str());

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::wfstream fs(p, std::ios_base::in | std::ios_base::out |
                            std::ios_base::trunc);
    double x = 0;
    fs << 3.25;
    fs.seekg(0);
    fs >> x;
    assert(x == 3.25);
  }
  std::remove(p.string().c_str());

  {
    std::basic_fstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fs(
        p, std::ios_base::in | std::ios_base::out | std::ios_base::trunc);
    std::basic_string<wchar_t, operator_hijacker_char_traits<wchar_t> > x;
    fs << L"3.25";
    fs.seekg(0);
    fs >> x;
    assert(x == L"3.25");
  }
  std::remove(p.string().c_str());

#endif

  return 0;
}
