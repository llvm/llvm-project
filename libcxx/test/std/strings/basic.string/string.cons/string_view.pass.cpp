//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// explicit basic_string(basic_string_view<CharT, traits> sv, const Allocator& a = Allocator()); // constexpr since C++20

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string_view>
#include <string>
#include <type_traits>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

static_assert(!std::is_convertible<std::string_view, std::string const&>::value, "");
static_assert(!std::is_convertible<std::string_view, std::string>::value, "");

template <class Alloc, class CharT>
TEST_CONSTEXPR_CXX20 void test(std::basic_string_view<CharT> sv) {
  typedef std::basic_string<CharT, std::char_traits<CharT>, Alloc> S;
  typedef typename S::traits_type T;
  {
    S s2(sv);
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == sv.size());
    assert(T::compare(s2.data(), sv.data(), sv.size()) == 0);
    assert(s2.get_allocator() == Alloc());
    assert(s2.capacity() >= s2.size());
  }
  {
    S s2;
    s2 = sv;
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == sv.size());
    assert(T::compare(s2.data(), sv.data(), sv.size()) == 0);
    assert(s2.get_allocator() == Alloc());
    assert(s2.capacity() >= s2.size());
  }
}

template <class Alloc, class CharT>
TEST_CONSTEXPR_CXX20 void test(std::basic_string_view<CharT> sv, const Alloc& a) {
  typedef std::basic_string<CharT, std::char_traits<CharT>, Alloc> S;
  typedef typename S::traits_type T;
  {
    S s2(sv, a);
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == sv.size());
    assert(T::compare(s2.data(), sv.data(), sv.size()) == 0);
    assert(s2.get_allocator() == a);
    assert(s2.capacity() >= s2.size());
  }
  {
    S s2(a);
    s2 = sv;
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == sv.size());
    assert(T::compare(s2.data(), sv.data(), sv.size()) == 0);
    assert(s2.get_allocator() == a);
    assert(s2.capacity() >= s2.size());
  }
}

template <class Alloc>
TEST_CONSTEXPR_CXX20 void test_string(const Alloc& a) {
  typedef std::basic_string_view<char, std::char_traits<char> > SV;

  test<Alloc>(SV(""));
  test<Alloc>(SV(""), Alloc(a));

  test<Alloc>(SV("1"));
  test<Alloc>(SV("1"), Alloc(a));

  test<Alloc>(SV("1234567980"));
  test<Alloc>(SV("1234567980"), Alloc(a));

  test<Alloc>(SV("123456798012345679801234567980123456798012345679801234567980"));
  test<Alloc>(SV("123456798012345679801234567980123456798012345679801234567980"), Alloc(a));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string(std::allocator<char>());
  test_string(test_allocator<char>());
  test_string(test_allocator<char>(2));
#if TEST_STD_VER >= 11
  test_string(min_allocator<char>());
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
