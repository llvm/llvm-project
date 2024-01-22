//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string substr(size_type pos = 0, size_type n = npos) const; // constexpr since C++20, removed in C++23
// basic_string substr(size_type pos = 0, size_type n = npos) const&; // since in C++23

#include <string>
#include <stdexcept>
#include <algorithm>
#include <cassert>

#include "test_allocator.h"
#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(const S& s, typename S::size_type pos, typename S::size_type n) {
  if (pos <= s.size()) {
    S str = s.substr(pos, n);
    LIBCPP_ASSERT(str.__invariants());
    assert(pos <= s.size());
    typename S::size_type rlen = std::min(n, s.size() - pos);
    assert(str.size() == rlen);
    assert(S::traits_type::compare(s.data() + pos, str.data(), rlen) == 0);
    LIBCPP_ASSERT(is_string_asan_correct(s));
    LIBCPP_ASSERT(is_string_asan_correct(str));
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  else if (!TEST_IS_CONSTANT_EVALUATED) {
    try {
      S str = s.substr(pos, n);
      assert(false);
    } catch (std::out_of_range&) {
      assert(pos > s.size());
    }
  }
#endif
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(""), 0, 0);
  test(S(""), 1, 0);
  test(S("pniot"), 0, 0);
  test(S("htaob"), 0, 1);
  test(S("fodgq"), 0, 2);
  test(S("hpqia"), 0, 4);
  test(S("qanej"), 0, 5);
  test(S("dfkap"), 1, 0);
  test(S("clbao"), 1, 1);
  test(S("ihqrf"), 1, 2);
  test(S("mekdn"), 1, 3);
  test(S("ngtjf"), 1, 4);
  test(S("srdfq"), 2, 0);
  test(S("qkdrs"), 2, 1);
  test(S("ikcrq"), 2, 2);
  test(S("cdaih"), 2, 3);
  test(S("dmajb"), 4, 0);
  test(S("karth"), 4, 1);
  test(S("lhcdo"), 5, 0);
  test(S("acbsj"), 6, 0);
  test(S("pbsjikaole"), 0, 0);
  test(S("pcbahntsje"), 0, 1);
  test(S("mprdjbeiak"), 0, 5);
  test(S("fhepcrntko"), 0, 9);
  test(S("eqmpaidtls"), 0, 10);
  test(S("joidhalcmq"), 1, 0);
  test(S("omigsphflj"), 1, 1);
  test(S("kocgbphfji"), 1, 4);
  test(S("onmjekafbi"), 1, 8);
  test(S("fbslrjiqkm"), 1, 9);
  test(S("oqmrjahnkg"), 5, 0);
  test(S("jeidpcmalh"), 5, 1);
  test(S("schfalibje"), 5, 2);
  test(S("crliponbqe"), 5, 4);
  test(S("igdscopqtm"), 5, 5);
  test(S("qngpdkimlc"), 9, 0);
  test(S("thdjgafrlb"), 9, 1);
  test(S("hcjitbfapl"), 10, 0);
  test(S("mgojkldsqh"), 11, 0);
  test(S("gfshlcmdjreqipbontak"), 0, 0);
  test(S("nadkhpfemgclosibtjrq"), 0, 1);
  test(S("nkodajteqplrbifhmcgs"), 0, 10);
  test(S("ofdrqmkeblthacpgijsn"), 0, 19);
  test(S("gbmetiprqdoasckjfhln"), 0, 20);
  test(S("bdfjqgatlksriohemnpc"), 1, 0);
  test(S("crnklpmegdqfiashtojb"), 1, 1);
  test(S("ejqcnahdrkfsmptilgbo"), 1, 9);
  test(S("jsbtafedocnirgpmkhql"), 1, 18);
  test(S("prqgnlbaejsmkhdctoif"), 1, 19);
  test(S("qnmodrtkebhpasifgcjl"), 10, 0);
  test(S("pejafmnokrqhtisbcdgl"), 10, 1);
  test(S("cpebqsfmnjdolhkratgi"), 10, 5);
  test(S("odnqkgijrhabfmcestlp"), 10, 9);
  test(S("lmofqdhpkibagnrcjste"), 10, 10);
  test(S("lgjqketopbfahrmnsicd"), 19, 0);
  test(S("ktsrmnqagdecfhijpobl"), 19, 1);
  test(S("lsaijeqhtrbgcdmpfkno"), 20, 0);
  test(S("dplqartnfgejichmoskb"), 21, 0);
  test(S("gbmetiprqdoasckjfhlnxx"), 0, 22);
  test(S("gbmetiprqdoasckjfhlnxa"), 0, 8);
  test(S("gbmetiprqdoasckjfhlnxb"), 1, 0);
  test(S("LONGtiprqdoasckjfhlnxxo"), 0, 23);
  test(S("LONGtiprqdoasckjfhlnxap"), 0, 8);
  test(S("LONGtiprqdoasckjfhlnxbl"), 1, 0);
  test(S("LONGtiprqdoasckjfhlnxxyy"), 0, 24);
  test(S("LONGtiprqdoasckjfhlnxxyr"), 0, 8);
  test(S("LONGtiprqdoasckjfhlnxxyz"), 1, 0);
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char>>>();
#endif

  return true;
}

TEST_CONSTEXPR_CXX20 bool test_alloc() {
  {
    using alloc  = test_allocator<char>;
    using string = std::basic_string<char, std::char_traits<char>, alloc>;
    test_allocator_statistics stats;
    {
      string str = string(alloc(&stats));
      stats      = test_allocator_statistics();
      (void)str.substr();
      assert(stats.moved == 0);
      assert(stats.copied == 0);
    }
    {
      string str = string(alloc(&stats));
      stats      = test_allocator_statistics();
      (void)std::move(str).substr();
      assert(stats.moved == 0);
      assert(stats.copied == 0);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  test_alloc();
#if TEST_STD_VER > 17
  static_assert(test());
  static_assert(test_alloc());
#endif

  return 0;
}
