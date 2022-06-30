//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-localization

// Aligned allocation is required by std::experimental::pmr, but it was not provided
// before macosx10.13 and as a result we get linker errors when deploying to older than
// macosx10.13.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <experimental/regex>

// namespace std { namespace experimental { namespace pmr {
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
// }}} // namespace std::experimental::pmr

#include <experimental/regex>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace pmr = std::experimental::pmr;

template <class Iter, class PmrTypedef>
void test_match_result_typedef() {
    using StdMR = std::match_results<Iter, pmr::polymorphic_allocator<std::sub_match<Iter>>>;
    using PmrMR = pmr::match_results<Iter>;
    static_assert(std::is_same<StdMR, PmrMR>::value, "");
    static_assert(std::is_same<PmrMR, PmrTypedef>::value, "");
}

int main(int, char**)
{
    {
        test_match_result_typedef<const char*, pmr::cmatch>();
        test_match_result_typedef<pmr::string::const_iterator, pmr::smatch>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        test_match_result_typedef<const wchar_t*, pmr::wcmatch>();
        test_match_result_typedef<pmr::wstring::const_iterator, pmr::wsmatch>();
#endif
    }
    {
        // Check that std::match_results has been included and is complete.
        pmr::smatch s;
        assert(s.get_allocator().resource() == pmr::get_default_resource());
    }

  return 0;
}
