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
    assert(p.string<char>() == TestPath);
    assert(p.string<char>() == S);
  }
  {
    const std::string_view S(TestPath);
    path p(S, Locale, args...); 
    assert(p.native() == Expect);
    assert(p.string<char>() == TestPath);
    assert(p.string<char>() == S);
  }
  // Char* pointers
  {
    path p(TestPath, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == TestPath);
  }
  {
    path p(TestPath, TestPathEnd, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == TestPath);
  }
  // Iterators
  {
    using It = input_iterator<const char*>;
    path p(It{TestPath}, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == TestPath);
  }
  {
    using It = input_iterator<const char*>;
    path p(It{TestPath}, It{TestPathEnd}, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == TestPath);
  }
}

void test_sfinae() {
  using namespace fs;
  {
    using It = const char* const;
    static_assert(std::is_constructible<path, It>::value, "");
  }
  {
    using It = input_iterator<const char*>;
    static_assert(std::is_constructible<path, It>::value, "");
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
    static_assert(std::is_constructible<path, It>::value, "");
  }
  {
    using It = output_iterator<const char*>;
    static_assert(!std::is_constructible<path, It>::value, "");

  }
  {
    static_assert(!std::is_constructible<path, int*>::value, "");
  }
}

struct CustomCodeCvt : std::codecvt<wchar_t, char, std::mbstate_t> {
protected:
  result do_in(state_type&,
               const extern_type* __frm, const extern_type* __frm_end, const extern_type*& __frm_nxt,
               intern_type* __to, intern_type* __to_end, intern_type*& __to_nxt) const override {
    for (; __frm < __frm_end && __to < __to_end; ++__frm, ++__to)
      *__to = 'o';
    
    __frm_nxt = __frm;
    __to_nxt = __to;

    return result::ok;
  }
};

int main(int, char**) {
  std::locale Locale;
  std::locale CustomLocale(Locale, new CustomCodeCvt());

  // Ensure std::codecvt<wchar_t, char, std::mbstate_t> is used.
  std::string TestPath("aaaa");
  std::string Expect("oooo");
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale);
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale, fs::path::format::auto_format);
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale, fs::path::format::native_format);
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale, fs::path::format::generic_format);

  // Test on paths with global locale.
  for (auto const& MS : PathList) {
    RunTestCase(MS, MS, Locale);
    RunTestCase(MS, MS, Locale, fs::path::format::auto_format);
    RunTestCase(MS, MS, Locale, fs::path::format::native_format);
    RunTestCase(MS, MS, Locale, fs::path::format::generic_format);
  }

  test_sfinae();

  return 0;
}
