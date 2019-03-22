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

// match_results& operator=(match_results&& m);

#include <regex>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"

template <class CharT, class Allocator>
void
test(const Allocator& a)
{
    typedef std::match_results<const CharT*, Allocator> SM;
    SM m0(a);
    SM m1;

    m1 = std::move(m0);
    assert(m1.size()          == 0);
    assert(m1.str()           == std::basic_string<CharT>());
    if (std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value)
        assert(m1.get_allocator() == a);
    else
        assert(m1.get_allocator() == Allocator());
}

int main(int, char**)
{
    test<char>   (std::allocator<std::sub_match<const char *> >());
    test<wchar_t>(std::allocator<std::sub_match<const wchar_t *> >());

//  test_allocator has POCMA -> false
    test<char>   (test_allocator<std::sub_match<const char*> >(3));
    test<wchar_t>(test_allocator<std::sub_match<const wchar_t*> >(3));

//  other_allocator has POCMA -> true
    test<char>   (other_allocator<std::sub_match<const char*> >(3));
    test<wchar_t>(other_allocator<std::sub_match<const wchar_t*> >(3));

  return 0;
}
