//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// <string>

// size_type max_size() const; // constexpr since C++20

// NOTE: asan and msan will fail for one of two reasons
// 1. If allocator_may_return_null=0 then they will fail because the allocation
//    returns null.
// 2. If allocator_may_return_null=1 then they will fail because the allocation
//    is too large to succeed.
// UNSUPPORTED: sanitizer-new-delete

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test1(const S& s)
{
    S s2(s);
    const size_t sz = s2.max_size() - 1;
    try { s2.resize(sz, 'x'); }
    catch ( const std::bad_alloc & ) { return ; }
    assert ( s2.size() ==  sz );
}

template <class S>
TEST_CONSTEXPR_CXX20 void
test2(const S& s)
{
    S s2(s);
    const size_t sz = s2.max_size();
    try { s2.resize(sz, 'x'); }
    catch ( const std::bad_alloc & ) { return ; }
    assert ( s.size() ==  sz );
}

template <class S>
TEST_CONSTEXPR_CXX20 void
test(const S& s)
{
    assert(s.max_size() >= s.size());
    test1(s);
    test2(s);
}

void test() {
  {
    typedef std::string S;
    test(S());
    test(S("123"));
    test(S("12345678901234567890123456789012345678901234567890"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S());
    test(S("123"));
    test(S("12345678901234567890123456789012345678901234567890"));
  }
#endif
}

#if TEST_STD_VER > 17
constexpr bool test_constexpr() {
  std::string str;

  size_t size = str.max_size();
  assert(size > 0);

  return true;
}
#endif

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  test_constexpr();
  static_assert(test_constexpr());
#endif

  return 0;
}
