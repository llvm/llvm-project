//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string<charT,traits,Allocator>&
//   assign(basic_string_view<charT,traits> sv); // constexpr since C++20

#include <string>
#include <string_view>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"

template <class S, class SV>
TEST_CONSTEXPR_CXX20 void
test(S s, SV sv, S expected)
{
    s.assign(sv);
    LIBCPP_ASSERT(s.__invariants());
    assert(s == expected);
}

template <class S, class SV>
TEST_CONSTEXPR_CXX20 void
testAlloc(S s, SV sv, const typename S::allocator_type& a)
{
    s.assign(sv);
    LIBCPP_ASSERT(s.__invariants());
    assert(s == sv);
    assert(s.get_allocator() == a);
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    typedef std::string_view SV;
    test(S(), SV(), S());
    test(S(), SV("12345"), S("12345"));
    test(S(), SV("1234567890"), S("1234567890"));
    test(S(), SV("12345678901234567890"), S("12345678901234567890"));

    test(S("12345"), SV(), S());
    test(S("12345"), SV("12345"), S("12345"));
    test(S("12345"), SV("1234567890"), S("1234567890"));
    test(S("12345"), SV("12345678901234567890"), S("12345678901234567890"));

    test(S("1234567890"), SV(), S());
    test(S("1234567890"), SV("12345"), S("12345"));
    test(S("1234567890"), SV("1234567890"), S("1234567890"));
    test(S("1234567890"), SV("12345678901234567890"), S("12345678901234567890"));

    test(S("12345678901234567890"), SV(), S());
    test(S("12345678901234567890"), SV("12345"), S("12345"));
    test(S("12345678901234567890"), SV("1234567890"), S("1234567890"));
    test(S("12345678901234567890"), SV("12345678901234567890"),
         S("12345678901234567890"));

    testAlloc(S(), SV(), std::allocator<char>());
    testAlloc(S(), SV("12345"), std::allocator<char>());
    testAlloc(S(), SV("1234567890"), std::allocator<char>());
    testAlloc(S(), SV("12345678901234567890"), std::allocator<char>());
  }

#if TEST_STD_VER >= 11
  {
    typedef std::basic_string     <char, std::char_traits<char>, min_allocator<char>> S;
    typedef std::basic_string_view<char, std::char_traits<char> > SV;
    test(S(), SV(), S());
    test(S(), SV("12345"), S("12345"));
    test(S(), SV("1234567890"), S("1234567890"));
    test(S(), SV("12345678901234567890"), S("12345678901234567890"));

    test(S("12345"), SV(), S());
    test(S("12345"), SV("12345"), S("12345"));
    test(S("12345"), SV("1234567890"), S("1234567890"));
    test(S("12345"), SV("12345678901234567890"), S("12345678901234567890"));

    test(S("1234567890"), SV(), S());
    test(S("1234567890"), SV("12345"), S("12345"));
    test(S("1234567890"), SV("1234567890"), S("1234567890"));
    test(S("1234567890"), SV("12345678901234567890"), S("12345678901234567890"));

    test(S("12345678901234567890"), SV(), S());
    test(S("12345678901234567890"), SV("12345"), S("12345"));
    test(S("12345678901234567890"), SV("1234567890"), S("1234567890"));
    test(S("12345678901234567890"), SV("12345678901234567890"),
         S("12345678901234567890"));

    testAlloc(S(), SV(), min_allocator<char>());
    testAlloc(S(), SV("12345"), min_allocator<char>());
    testAlloc(S(), SV("1234567890"), min_allocator<char>());
    testAlloc(S(), SV("12345678901234567890"), min_allocator<char>());
  }
#endif

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
