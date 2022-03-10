//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

//       reverse_iterator rend();
// const_reverse_iterator rend() const;

#include <string>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(S s)
{
    const S& cs = s;
    typename S::reverse_iterator e = s.rend();
    typename S::const_reverse_iterator ce = cs.rend();
    if (s.empty())
    {
        assert(e == s.rbegin());
        assert(ce == cs.rbegin());
    }
    assert(static_cast<std::size_t>(e - s.rbegin()) == s.size());
    assert(static_cast<std::size_t>(ce - cs.rbegin()) == cs.size());
}

bool test() {
  {
    typedef std::string S;
    test(S());
    test(S("123"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S());
    test(S("123"));
  }
#endif

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  // static_assert(test());
#endif

  return 0;
}
