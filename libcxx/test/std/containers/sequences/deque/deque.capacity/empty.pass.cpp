//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// class deque

// bool empty() const noexcept;

#include "asan_testing.h"
#include <deque>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef std::deque<int> C;
    C c;
    ASSERT_NOEXCEPT(c.empty());
    assert(c.empty());
    c.push_back(C::value_type(1));
    assert(!c.empty());
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    c.clear();
    assert(c.empty());
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
#if TEST_STD_VER >= 11
    {
    typedef std::deque<int, min_allocator<int>> C;
    C c;
    ASSERT_NOEXCEPT(c.empty());
    assert(c.empty());
    c.push_back(C::value_type(1));
    assert(!c.empty());
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    c.clear();
    assert(c.empty());
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
#endif

  return 0;
}
