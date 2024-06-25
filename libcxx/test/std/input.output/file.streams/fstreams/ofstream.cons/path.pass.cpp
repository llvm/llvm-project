//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: availability-filesystem-missing

// <fstream>

// plate <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// template<class T>
// explicit basic_ifstream(const T& s, ios_base::openmode mode = ios_base::in); // Since C++17
// Constraints: is_same_v<T, filesystem::path> is true

#include <cassert>
#include <filesystem>
#include <fstream>
#include <type_traits>

#include "platform_support.h"
#include "test_macros.h"
#include "test_iterators.h"

namespace fs = std::filesystem;

template <class CharT>
constexpr bool test_non_convert_to_path() {
  // String types
  static_assert(!std::is_constructible_v<std::ofstream, std::basic_string_view<CharT>>);
  static_assert(!std::is_constructible_v<std::ofstream, const std::basic_string_view<CharT>>);

  // Char* pointers
  if constexpr (!std::is_same_v<CharT, char> && !std::is_same_v<CharT, fs::path::value_type>)
    static_assert(!std::is_constructible_v<std::ofstream, const CharT*>);

  // Iterators
  static_assert(!std::is_convertible_v<std::ofstream, cpp17_input_iterator<const CharT*>>);

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
    static_assert(!std::is_convertible<fs::path, std::ofstream>::value,
                  "ctor should be explicit");
    static_assert(std::is_constructible<std::ofstream, fs::path const&,
                                        std::ios_base::openmode>::value,
                  "");
  }
  {
    std::ofstream stream(p);
    stream << 3.25;
  }
  {
    std::ifstream stream(p);
    double x = 0;
    stream >> x;
    assert(x == 3.25);
  }
  {
    std::ifstream stream(p, std::ios_base::out);
    double x = 0;
    stream >> x;
    assert(x == 3.25);
  }
  std::remove(p.string().c_str());

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::wofstream stream(p);
    stream << 3.25;
  }
  {
    std::wifstream stream(p);
    double x = 0;
    stream >> x;
    assert(x == 3.25);
  }
  {
    std::wifstream stream(p, std::ios_base::out);
    double x = 0;
    stream >> x;
    assert(x == 3.25);
  }
  std::remove(p.string().c_str());
#endif

  return 0;
}
