//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03
// <regex>

// class match_results<BidirectionalIterator, Allocator>

// match_results(match_results&& m) noexcept;
//
//  Additionally, the stored Allocator value is move constructed from m.get_allocator().

#include <regex>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"

template <class CharT, class Allocator>
void
test(const Allocator& a)
{
    typedef std::match_results<const CharT*, Allocator> SM;
    ASSERT_NOEXCEPT(SM(std::declval<SM&&>()));

    SM m0(a);
    assert(m0.get_allocator() == a);

    SM m1(std::move(m0));
    assert(m1.size() == 0);
    assert(!m1.ready());
    assert(m1.get_allocator() == a);
}

int main(int, char**)
{
    test<char>   (std::allocator<std::sub_match<const char *> >());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>(std::allocator<std::sub_match<const wchar_t *> >());
#endif

    test<char>   (test_allocator<std::sub_match<const char*> >(3));
    assert(test_alloc_base::moved == 1);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>(test_allocator<std::sub_match<const wchar_t*> >(3));
    assert(test_alloc_base::moved == 2);
#endif

  return 0;
}
