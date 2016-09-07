//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// class path

// template <class Source>
//      path(const Source& source);
// template <class InputIterator>
//      path(InputIterator first, InputIterator last);


#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"
#include "filesystem_test_helper.hpp"

namespace fs = std::experimental::filesystem;

template <class CharT>
void RunTestCase(MultiStringType const& MS) {
  using namespace fs;
  const char* Expect = MS;
  const CharT* TestPath = MS;
  const CharT* TestPathEnd = StrEnd(TestPath);
  const std::size_t Size = TestPathEnd - TestPath;
  const std::size_t SSize = StrEnd(Expect) - Expect;
  assert(Size == SSize);
  // StringTypes
  {
    const std::basic_string<CharT> S(TestPath);
    path p(S);
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
    assert(p.string<CharT>() == S);
  }
  {
    const std::basic_string_view<CharT> S(TestPath);
    path p(S);
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
    assert(p.string<CharT>() == S);
  }
  // Char* pointers
  {
    path p(TestPath);
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    path p(TestPath, TestPathEnd);
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  // Iterators
  {
    using It = input_iterator<const CharT*>;
    path p(It{TestPath});
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    using It = input_iterator<const CharT*>;
    path p(It{TestPath}, It{TestPathEnd});
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
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

int main() {
  for (auto const& MS : PathList) {
    RunTestCase<char>(MS);
    RunTestCase<wchar_t>(MS);
    RunTestCase<char16_t>(MS);
    RunTestCase<char32_t>(MS);
  }
  test_sfinae();
}
