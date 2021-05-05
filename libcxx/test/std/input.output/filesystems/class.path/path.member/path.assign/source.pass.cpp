//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// These tests require locale for non-char paths
// UNSUPPORTED: libcpp-has-no-localization

// <filesystem>

// class path

// template <class Source>
//      path& operator=(Source const&);
//  path& operator=(string_type&&);
// template <class Source>
//      path& assign(Source const&);
// template <class InputIterator>
//      path& assign(InputIterator first, InputIterator last);


#include "filesystem_include.h"
#include <type_traits>
#include <string_view>
#include <cassert>

// On Windows, charset conversions cause allocations in the path class in
// cases where no allocations are done on other platforms.

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"


template <class CharT>
void RunTestCase(MultiStringType const& MS) {
  using namespace fs;
  const fs::path::value_type* Expect = MS;
  const CharT* TestPath = MS;
  const CharT* TestPathEnd = StrEnd(TestPath);
  const std::size_t Size = TestPathEnd - TestPath;
  const std::size_t SSize = StrEnd(Expect) - Expect;
  assert(Size == SSize);
  //////////////////////////////////////////////////////////////////////////////
  // basic_string<Char, Traits, Alloc>
  {
    const std::basic_string<CharT> S(TestPath);
    path p; PathReserve(p, S.length() + 1);
    {
      // string provides a contiguous iterator. No allocation needed.
      TEST_NOT_WIN32(DisableAllocationGuard g);
      path& pref = (p = S);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
    assert(p.string<CharT>() == S);
  }
  {
    const std::basic_string<CharT> S(TestPath);
    path p; PathReserve(p, S.length() + 1);
    {
      TEST_NOT_WIN32(DisableAllocationGuard g);
      path& pref = p.assign(S);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
    assert(p.string<CharT>() == S);
  }
  // basic_string<Char, Traits, Alloc>
  {
    const std::basic_string_view<CharT> S(TestPath);
    path p; PathReserve(p, S.length() + 1);
    {
      // string provides a contiguous iterator. No allocation needed.
      TEST_NOT_WIN32(DisableAllocationGuard g);
      path& pref = (p = S);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
    assert(p.string<CharT>() == S);
  }
  {
    const std::basic_string_view<CharT> S(TestPath);
    path p; PathReserve(p, S.length() + 1);
    {
      TEST_NOT_WIN32(DisableAllocationGuard g);
      path& pref = p.assign(S);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
    assert(p.string<CharT>() == S);
  }
  //////////////////////////////////////////////////////////////////////////////
  // Char* pointers
  {
    path p; PathReserve(p, Size + 1);
    {
      // char* pointers are contiguous and can be used with code_cvt directly.
      // no allocations needed.
      TEST_NOT_WIN32(DisableAllocationGuard g);
      path& pref = (p = TestPath);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    path p; PathReserve(p, Size + 1);
    {
      TEST_NOT_WIN32(DisableAllocationGuard g);
      path& pref = p.assign(TestPath);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    path p; PathReserve(p, Size + 1);
    {
      TEST_NOT_WIN32(DisableAllocationGuard g);
      path& pref = p.assign(TestPath, TestPathEnd);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  //////////////////////////////////////////////////////////////////////////////
  // Iterators
  {
    using It = cpp17_input_iterator<const CharT*>;
    path p; PathReserve(p, Size + 1);
    It it(TestPath);
    {
      // Iterators cannot be used with code_cvt directly. This assignment
      // may allocate if it's larger than a "short-string".
      path& pref = (p = it);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    using It = cpp17_input_iterator<const CharT*>;
    path p; PathReserve(p, Size + 1);
    It it(TestPath);
    {
      path& pref = p.assign(it);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    using It = cpp17_input_iterator<const CharT*>;
    path p; PathReserve(p, Size + 1);
    It it(TestPath);
    It e(TestPathEnd);
    {
      path& pref = p.assign(it, e);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
}

template <class It, class = decltype(fs::path{}.assign(std::declval<It>()))>
constexpr bool has_assign(int) { return true; }
template <class It>
constexpr bool has_assign(long) { return false; }
template <class It>
constexpr bool has_assign() { return has_assign<It>(0); }

void test_sfinae() {
  using namespace fs;
  {
    using It = const char* const;
    static_assert(std::is_assignable<path, It>::value, "");
    static_assert(has_assign<It>(), "");
  }
  {
    using It = cpp17_input_iterator<const char*>;
    static_assert(std::is_assignable<path, It>::value, "");
    static_assert(has_assign<It>(), "");
  }
  {
    struct Traits {
      using iterator_category = std::input_iterator_tag;
      using value_type = const char;
      using pointer = const char*;
      using reference = const char&;
      using difference_type = std::ptrdiff_t;
    };
    using It = cpp17_input_iterator<const char*, Traits>;
    static_assert(std::is_assignable<path, It>::value, "");
    static_assert(has_assign<It>(), "");
  }
  {
    using It = output_iterator<const char*>;
    static_assert(!std::is_assignable<path, It>::value, "");
    static_assert(!has_assign<It>(), "");

  }
  {
    static_assert(!std::is_assignable<path, int*>::value, "");
    static_assert(!has_assign<int*>(), "");
  }
}

void RunStringMoveTest(const fs::path::value_type* Expect) {
  using namespace fs;
  fs::path::string_type ss(Expect);
  path p;
  {
    DisableAllocationGuard g; ((void)g);
    path& pr = (p = std::move(ss));
    assert(&pr == &p);
  }
  assert(p == Expect);
  {
    // Signature test
    LIBCPP_ONLY(ASSERT_NOEXCEPT(p = std::move(ss)));
  }
}

int main(int, char**) {
  for (auto const& MS : PathList) {
    RunTestCase<char>(MS);
    RunTestCase<wchar_t>(MS);
    RunTestCase<char16_t>(MS);
    RunTestCase<char32_t>(MS);
    RunStringMoveTest(MS);
  }
  test_sfinae();

  return 0;
}
