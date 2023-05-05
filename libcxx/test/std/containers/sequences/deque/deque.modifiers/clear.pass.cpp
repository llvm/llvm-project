//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// void clear() noexcept;

#include "asan_testing.h"
#include <deque>
#include <cassert>

#include "test_macros.h"
#include "../../../NotConstructible.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef NotConstructible T;
        typedef std::deque<T> C;
        C c;
        ASSERT_NOEXCEPT(c.clear());
        c.clear();
        assert(std::distance(c.begin(), c.end()) == 0);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
    {
        typedef int T;
        typedef std::deque<T> C;
        const T t[] = {0, 1, 2, 3, 4};
        C c(std::begin(t), std::end(t));

        ASSERT_NOEXCEPT(c.clear());
        c.clear();
        assert(std::distance(c.begin(), c.end()) == 0);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));

        c.clear();
        assert(std::distance(c.begin(), c.end()) == 0);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
#if TEST_STD_VER >= 11
    {
        typedef NotConstructible T;
        typedef std::deque<T, min_allocator<T>> C;
        C c;
        ASSERT_NOEXCEPT(c.clear());
        c.clear();
        assert(std::distance(c.begin(), c.end()) == 0);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
    {
        typedef int T;
        typedef std::deque<T, min_allocator<T>> C;
        const T t[] = {0, 1, 2, 3, 4};
        C c(std::begin(t), std::end(t));

        ASSERT_NOEXCEPT(c.clear());
        c.clear();
        assert(std::distance(c.begin(), c.end()) == 0);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));

        c.clear();
        assert(std::distance(c.begin(), c.end()) == 0);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
#endif

  return 0;
}
