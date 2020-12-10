//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void reserve(size_type res_arg);

// This test relies on https://llvm.org/PR45368 being fixed, which isn't in
// older Apple dylibs
//
// XFAIL: with_system_cxx_lib=macosx10.15
// XFAIL: with_system_cxx_lib=macosx10.14
// XFAIL: with_system_cxx_lib=macosx10.13
// XFAIL: with_system_cxx_lib=macosx10.12
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9

#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(typename S::size_type min_cap, typename S::size_type erased_index, typename S::size_type res_arg)
{
    S s(min_cap, 'a');
    s.erase(erased_index);
    assert(s.size() == erased_index);
    assert(s.capacity() >= min_cap); // Check that we really have at least this capacity.

#if TEST_STD_VER > 17
    typename S::size_type old_cap = s.capacity();
#endif
    S s0 = s;
    if (res_arg <= s.max_size())
    {
        s.reserve(res_arg);
        LIBCPP_ASSERT(s.__invariants());
        assert(s == s0);
        assert(s.capacity() >= res_arg);
        assert(s.capacity() >= s.size());
#if TEST_STD_VER > 17
        assert(s.capacity() >= old_cap); // reserve never shrinks as of P0966 (C++20)
#endif
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    else
    {
        try
        {
            s.reserve(res_arg);
            LIBCPP_ASSERT(s.__invariants());
            assert(false);
        }
        catch (std::length_error&)
        {
            assert(res_arg > s.max_size());
        }
    }
#endif
}

int main(int, char**)
{
    {
    typedef std::string S;
    {
    test<S>(0, 0, 5);
    test<S>(0, 0, 10);
    test<S>(0, 0, 50);
    }
    {
    test<S>(100, 50, 5);
    test<S>(100, 50, 10);
    test<S>(100, 50, 50);
    test<S>(100, 50, 100);
    test<S>(100, 50, 1000);
    test<S>(100, 50, S::npos);
    }
    }
#if TEST_STD_VER >= 11
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    {
    test<S>(0, 0, 5);
    test<S>(0, 0, 10);
    test<S>(0, 0, 50);
    }
    {
    test<S>(100, 50, 5);
    test<S>(100, 50, 10);
    test<S>(100, 50, 50);
    test<S>(100, 50, 100);
    test<S>(100, 50, 1000);
    test<S>(100, 50, S::npos);
    }
    }
#endif

  return 0;
}
