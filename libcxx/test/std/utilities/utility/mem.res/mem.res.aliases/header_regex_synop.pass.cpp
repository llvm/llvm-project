//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-localization
// TODO: Change to XFAIL once https://github.com/llvm/llvm-project/issues/40340 is fixed
// UNSUPPORTED: availability-pmr-missing

// <regex>

// namespace std::pmr {
//
//  template <class BidirectionalIterator>
//  using match_results =
//    std::match_results<BidirectionalIterator,
//                       polymorphic_allocator<sub_match<BidirectionalIterator>>>;
//
//  typedef match_results<const char*> cmatch;
//  typedef match_results<const wchar_t*> wcmatch;
//  typedef match_results<string::const_iterator> smatch;
//  typedef match_results<wstring::const_iterator> wsmatch;
//
// } // namespace std::pmr

#include <regex>
#include <cassert>
#include <memory_resource>
#include <type_traits>

#include "test_macros.h"

template <class Iter, class PmrTypedef>
void test_match_result_typedef() {
  using StdMR = std::match_results<Iter, std::pmr::polymorphic_allocator<std::sub_match<Iter>>>;
  using PmrMR = std::pmr::match_results<Iter>;
  static_assert(std::is_same<StdMR, PmrMR>::value, "");
  static_assert(std::is_same<PmrMR, PmrTypedef>::value, "");
}

int main(int, char**) {
  {
    test_match_result_typedef<const char*, std::pmr::cmatch>();
    test_match_result_typedef<std::pmr::string::const_iterator, std::pmr::smatch>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test_match_result_typedef<const wchar_t*, std::pmr::wcmatch>();
    test_match_result_typedef<std::pmr::wstring::const_iterator, std::pmr::wsmatch>();
#endif
  }
  {
    // Check that std::match_results has been included and is complete.
    std::pmr::smatch s;
    assert(s.get_allocator().resource() == std::pmr::get_default_resource());
  }

  return 0;
}
