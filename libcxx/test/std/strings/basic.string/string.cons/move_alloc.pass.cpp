//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// XFAIL: LIBCXX-AIX-FIXME

// <string>

// basic_string(basic_string&& str, const Allocator& alloc); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(S s0, const typename S::allocator_type& a)
{
    S s1 = s0;
    S s2(std::move(s0), a);
    LIBCPP_ASSERT(s2.__invariants());
    LIBCPP_ASSERT(s0.__invariants());
    assert(s2 == s1);
    assert(s2.capacity() >= s2.size());
    assert(s2.get_allocator() == a);
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_allocator_statistics alloc_stats;
  {
    typedef test_allocator<char> A;
    typedef std::basic_string<char, std::char_traits<char>, A> S;
#if TEST_STD_VER > 14
    static_assert((noexcept(S{})), "" );
#elif TEST_STD_VER >= 11
    static_assert((noexcept(S()) == std::is_nothrow_move_constructible<A>::value), "" );
#endif
    test(S(), A(3, &alloc_stats));
    test(S("1"), A(5, &alloc_stats));
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890"), A(7, &alloc_stats));
  }

    int alloc_count = alloc_stats.alloc_count;
  {
    typedef test_allocator<char> A;
    typedef std::basic_string<char, std::char_traits<char>, A> S;
#if TEST_STD_VER > 14
    static_assert((noexcept(S{})), "" );
#elif TEST_STD_VER >= 11
    static_assert((noexcept(S()) == std::is_nothrow_move_constructible<A>::value), "" );
#endif
    S s1 ( "Twas brillig, and the slivy toves did gyre and gymbal in the wabe", A(&alloc_stats));
    S s2 (std::move(s1), A(1, &alloc_stats));
  }
    assert ( alloc_stats.alloc_count == alloc_count );
  {
    typedef min_allocator<char> A;
    typedef std::basic_string<char, std::char_traits<char>, A> S;
#if TEST_STD_VER > 14
    static_assert((noexcept(S{})), "" );
#elif TEST_STD_VER >= 11
    static_assert((noexcept(S()) == std::is_nothrow_move_constructible<A>::value), "" );
#endif
    test(S(), A());
    test(S("1"), A());
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890"), A());
  }

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
