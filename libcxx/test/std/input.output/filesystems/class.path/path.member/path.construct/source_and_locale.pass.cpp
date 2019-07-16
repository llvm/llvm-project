//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// class path

// template <class Source>
//      path(const Source& source);
// template <class InputIterator>
//      path(InputIterator first, InputIterator last);


#include "filesystem_include.hpp"
#include <locale>
#include <type_traits>
#include <cassert>
#include <iostream>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"
#include "filesystem_test_helper.hpp"

template <class ...Args>
void RunTestCase(const char* TestPath, const char* Expect, std::locale Locale, Args... args) {
  using namespace fs;
  const char* TestPathEnd = StrEnd(TestPath);
  const std::size_t Size = TestPathEnd - TestPath;
  const std::size_t SSize = StrEnd(Expect) - Expect;
  assert(Size == SSize);
  // StringTypes
  {
    const std::string S(TestPath);
    path p(S, Locale, args...); 
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  {
    const std::string_view S(TestPath);
    path p(S, Locale, args...); 
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  // Char* pointers
  {
    path p(TestPath, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  {
    path p(TestPath, TestPathEnd, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  // Iterators
  {
    using It = input_iterator<const char*>;
    path p(It{TestPath}, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  {
    using It = input_iterator<const char*>;
    path p(It{TestPath}, It{TestPathEnd}, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
}

void test_sfinae() {
  using namespace fs;
  {
    using It = const char* const;
    static_assert(std::is_constructible<path, It, std::locale>::value, "");
  }
  {
    using It = input_iterator<const char*>;
    static_assert(std::is_constructible<path, It, std::locale>::value, "");
  }
  {
    struct Traits {
      using iterator_category = std::input_iterator_tag;
      using value_type = const char;
      using pointer = const char*;
      using reference = const char&;
      using difference_type = std::ptrdiff_t;
    };
    using It = input_iterator<const char*, Traits>;
    static_assert(std::__is_input_iterator<It>::value, "");
    //static_assert(std::is_constructible<path, It, std::locale>::value, "");
  }
  {
    using It = const wchar_t* const;
    static_assert(!std::is_constructible<path, It, std::locale>::value, "");
  }
  {
    using It = input_iterator<const wchar_t*>;
    static_assert(!std::is_constructible<path, It, std::locale>::value, "");
  }
  {
    struct Traits {
      using iterator_category = std::input_iterator_tag;
      using value_type = const wchar_t;
      using pointer = const wchar_t*;
      using reference = const wchar_t&;
      using difference_type = std::ptrdiff_t;
    };
    using It = input_iterator<const wchar_t*, Traits>;
    static_assert(!std::is_constructible<path, It, std::locale>::value, "");
  }
  {
    using It = output_iterator<const char*>;
    static_assert(!std::is_constructible<path, It, std::locale>::value, "");
  }
  {
    static_assert(!std::is_constructible<path, int*, std::locale>::value, "");
  }
}

struct CustomCodeCvt : std::codecvt<wchar_t, char, std::mbstate_t> {
protected:
  result do_in(state_type&,
               const extern_type* from, const extern_type* from_end, const extern_type*& from_next,
               intern_type* to, intern_type* to_end, intern_type*& to_next) const override {
    for (; from < from_end && to < to_end; ++from, ++to)
      *to = 'o';
    
    from_next = from;
    to_next = to;

    return result::ok;
  }
};

int main(int, char**) {
  std::locale Locale;

  // Ensure std::codecvt<wchar_t, char, std::mbstate_t> is used.
  std::locale CustomLocale(Locale, new CustomCodeCvt());
  std::string TestPath("aaaa");
  std::string Expect("oooo");
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale);
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale, fs::path::format::auto_format);
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale, fs::path::format::native_format);
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale, fs::path::format::generic_format);

  for (auto const& MS : PathList) {
    RunTestCase(MS, MS, Locale);
    RunTestCase(MS, MS, Locale, fs::path::format::auto_format);
    RunTestCase(MS, MS, Locale, fs::path::format::native_format);
    RunTestCase(MS, MS, Locale, fs::path::format::generic_format);
  }

  test_sfinae();

  return 0;
}
