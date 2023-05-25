//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

//       reference operator[](size_type __i);
// const_reference operator[](size_type __i) const;
//
//       reference at(size_type __i);
// const_reference at(size_type __i) const;
//
//       reference front();
// const_reference front() const;
//
//       reference back();
// const_reference back() const;
// libc++ marks these as 'noexcept'

#include "asan_testing.h"
#include <deque>
#include <cassert>

#include "min_allocator.h"
#include "test_macros.h"

template <class C>
C
make(int size, int start = 0 )
{
    const int b = 4096 / sizeof(int);
    int init = 0;
    if (start > 0)
    {
        init = (start+1) / b + ((start+1) % b != 0);
        init *= b;
        --init;
    }
    C c(init, 0);
    for (int i = 0; i < init-start; ++i)
        c.pop_back();
    for (int i = 0; i < size; ++i)
        c.push_back(i);
    for (int i = 0; i < start; ++i)
        c.pop_front();
    return c;
}

int main(int, char**)
{
    {
        typedef std::deque<int> C;
        C c = make<std::deque<int> >(10);
        ASSERT_SAME_TYPE(decltype(c[0]), C::reference);
        LIBCPP_ASSERT_NOEXCEPT(   c[0]);
        LIBCPP_ASSERT_NOEXCEPT(   c.front());
        ASSERT_SAME_TYPE(decltype(c.front()), C::reference);
        LIBCPP_ASSERT_NOEXCEPT(   c.back());
        ASSERT_SAME_TYPE(decltype(c.back()), C::reference);
        for (int i = 0; i < 10; ++i)
            assert(c[i] == i);
        for (int i = 0; i < 10; ++i)
            assert(c.at(i) == i);
        assert(c.front() == 0);
        assert(c.back() == 9);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
    {
        typedef std::deque<int> C;
        const C c = make<std::deque<int> >(10);
        ASSERT_SAME_TYPE(decltype(c[0]), C::const_reference);
        LIBCPP_ASSERT_NOEXCEPT(   c[0]);
        LIBCPP_ASSERT_NOEXCEPT(   c.front());
        ASSERT_SAME_TYPE(decltype(c.front()), C::const_reference);
        LIBCPP_ASSERT_NOEXCEPT(   c.back());
        ASSERT_SAME_TYPE(decltype(c.back()), C::const_reference);
        for (int i = 0; i < 10; ++i)
            assert(c[i] == i);
        for (int i = 0; i < 10; ++i)
            assert(c.at(i) == i);
        assert(c.front() == 0);
        assert(c.back() == 9);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
#if TEST_STD_VER >= 11
    {
        typedef std::deque<int, min_allocator<int>> C;
        C c = make<std::deque<int, min_allocator<int>> >(10);
        ASSERT_SAME_TYPE(decltype(c[0]), C::reference);
        LIBCPP_ASSERT_NOEXCEPT(   c[0]);
        LIBCPP_ASSERT_NOEXCEPT(   c.front());
        ASSERT_SAME_TYPE(decltype(c.front()), C::reference);
        LIBCPP_ASSERT_NOEXCEPT(   c.back());
        ASSERT_SAME_TYPE(decltype(c.back()), C::reference);
        for (int i = 0; i < 10; ++i)
            assert(c[i] == i);
        for (int i = 0; i < 10; ++i)
            assert(c.at(i) == i);
        assert(c.front() == 0);
        assert(c.back() == 9);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
    {
        typedef std::deque<int, min_allocator<int>> C;
        const C c = make<std::deque<int, min_allocator<int>> >(10);
        ASSERT_SAME_TYPE(decltype(c[0]), C::const_reference);
        LIBCPP_ASSERT_NOEXCEPT(   c[0]);
        LIBCPP_ASSERT_NOEXCEPT(   c.front());
        ASSERT_SAME_TYPE(decltype(c.front()), C::const_reference);
        LIBCPP_ASSERT_NOEXCEPT(   c.back());
        ASSERT_SAME_TYPE(decltype(c.back()), C::const_reference);
        for (int i = 0; i < 10; ++i)
            assert(c[i] == i);
        for (int i = 0; i < 10; ++i)
            assert(c.at(i) == i);
        assert(c.front() == 0);
        assert(c.back() == 9);
        LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(c));
    }
#endif

  return 0;
}
