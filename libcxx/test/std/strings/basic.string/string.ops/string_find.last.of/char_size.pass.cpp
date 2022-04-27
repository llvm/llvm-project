//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <string>

// size_type find_last_of(charT c, size_type pos = npos) const; // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(const S& s, typename S::value_type c, typename S::size_type pos,
     typename S::size_type x)
{
    LIBCPP_ASSERT_NOEXCEPT(s.find_last_of(c, pos));
    assert(s.find_last_of(c, pos) == x);
    if (x != S::npos)
        assert(x <= pos && x < s.size());
}

template <class S>
TEST_CONSTEXPR_CXX20 void
test(const S& s, typename S::value_type c, typename S::size_type x)
{
    LIBCPP_ASSERT_NOEXCEPT(s.find_last_of(c));
    assert(s.find_last_of(c) == x);
    if (x != S::npos)
        assert(x < s.size());
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    test(S(""), 'm', 0, S::npos);
    test(S(""), 'm', 1, S::npos);
    test(S("kitcj"), 'm', 0, S::npos);
    test(S("qkamf"), 'm', 1, S::npos);
    test(S("nhmko"), 'm', 2, 2);
    test(S("tpsaf"), 'm', 4, S::npos);
    test(S("lahfb"), 'm', 5, S::npos);
    test(S("irkhs"), 'm', 6, S::npos);
    test(S("gmfhdaipsr"), 'm', 0, S::npos);
    test(S("kantesmpgj"), 'm', 1, S::npos);
    test(S("odaftiegpm"), 'm', 5, S::npos);
    test(S("oknlrstdpi"), 'm', 9, S::npos);
    test(S("eolhfgpjqk"), 'm', 10, S::npos);
    test(S("pcdrofikas"), 'm', 11, S::npos);
    test(S("nbatdlmekrgcfqsophij"), 'm', 0, S::npos);
    test(S("bnrpehidofmqtcksjgla"), 'm', 1, S::npos);
    test(S("jdmciepkaqgotsrfnhlb"), 'm', 10, 2);
    test(S("jtdaefblsokrmhpgcnqi"), 'm', 19, 12);
    test(S("hkbgspofltajcnedqmri"), 'm', 20, 17);
    test(S("oselktgbcapndfjihrmq"), 'm', 21, 18);

    test(S(""), 'm', S::npos);
    test(S("csope"), 'm', S::npos);
    test(S("gfsmthlkon"), 'm', 3);
    test(S("laenfsbridchgotmkqpj"), 'm', 15);
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S(""), 'm', 0, S::npos);
    test(S(""), 'm', 1, S::npos);
    test(S("kitcj"), 'm', 0, S::npos);
    test(S("qkamf"), 'm', 1, S::npos);
    test(S("nhmko"), 'm', 2, 2);
    test(S("tpsaf"), 'm', 4, S::npos);
    test(S("lahfb"), 'm', 5, S::npos);
    test(S("irkhs"), 'm', 6, S::npos);
    test(S("gmfhdaipsr"), 'm', 0, S::npos);
    test(S("kantesmpgj"), 'm', 1, S::npos);
    test(S("odaftiegpm"), 'm', 5, S::npos);
    test(S("oknlrstdpi"), 'm', 9, S::npos);
    test(S("eolhfgpjqk"), 'm', 10, S::npos);
    test(S("pcdrofikas"), 'm', 11, S::npos);
    test(S("nbatdlmekrgcfqsophij"), 'm', 0, S::npos);
    test(S("bnrpehidofmqtcksjgla"), 'm', 1, S::npos);
    test(S("jdmciepkaqgotsrfnhlb"), 'm', 10, 2);
    test(S("jtdaefblsokrmhpgcnqi"), 'm', 19, 12);
    test(S("hkbgspofltajcnedqmri"), 'm', 20, 17);
    test(S("oselktgbcapndfjihrmq"), 'm', 21, 18);

    test(S(""), 'm', S::npos);
    test(S("csope"), 'm', S::npos);
    test(S("gfsmthlkon"), 'm', 3);
    test(S("laenfsbridchgotmkqpj"), 'm', 15);
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
