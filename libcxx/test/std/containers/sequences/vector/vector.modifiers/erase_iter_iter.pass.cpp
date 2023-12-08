//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// iterator erase(const_iterator first, const_iterator last);

#include <vector>
#include <iterator>
#include <cassert>

#include "asan_testing.h"
#include "min_allocator.h"
#include "MoveOnly.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Throws {
    Throws() : v_(0) {}
    Throws(int v) : v_(v) {}
    Throws(const Throws  &rhs) : v_(rhs.v_) { if (sThrows) throw 1; }
    Throws(      Throws &&rhs) : v_(rhs.v_) { if (sThrows) throw 1; }
    Throws& operator=(const Throws  &rhs) { v_ = rhs.v_; return *this; }
    Throws& operator=(      Throws &&rhs) { v_ = rhs.v_; return *this; }
    int v_;
    static bool sThrows;
    };

bool Throws::sThrows = false;
#endif

TEST_CONSTEXPR_CXX20 bool tests()
{
    int a1[] = {1, 2, 3};
    {
        std::vector<int> l1(a1, a1+3);
        assert(is_contiguous_container_asan_correct(l1));
        std::vector<int>::iterator i = l1.erase(l1.cbegin(), l1.cbegin());
        assert(l1.size() == 3);
        assert(std::distance(l1.cbegin(), l1.cend()) == 3);
        assert(i == l1.begin());
        assert(is_contiguous_container_asan_correct(l1));
    }
    {
        std::vector<int> l1(a1, a1+3);
        assert(is_contiguous_container_asan_correct(l1));
        std::vector<int>::iterator i = l1.erase(l1.cbegin(), std::next(l1.cbegin()));
        assert(l1.size() == 2);
        assert(std::distance(l1.cbegin(), l1.cend()) == 2);
        assert(i == l1.begin());
        assert(l1 == std::vector<int>(a1+1, a1+3));
        assert(is_contiguous_container_asan_correct(l1));
    }
    {
        std::vector<int> l1(a1, a1+3);
        assert(is_contiguous_container_asan_correct(l1));
        std::vector<int>::iterator i = l1.erase(l1.cbegin(), std::next(l1.cbegin(), 2));
        assert(l1.size() == 1);
        assert(std::distance(l1.cbegin(), l1.cend()) == 1);
        assert(i == l1.begin());
        assert(l1 == std::vector<int>(a1+2, a1+3));
        assert(is_contiguous_container_asan_correct(l1));
    }
    {
        std::vector<int> l1(a1, a1+3);
        assert(is_contiguous_container_asan_correct(l1));
        std::vector<int>::iterator i = l1.erase(l1.cbegin(), std::next(l1.cbegin(), 3));
        assert(l1.size() == 0);
        assert(std::distance(l1.cbegin(), l1.cend()) == 0);
        assert(i == l1.begin());
        assert(is_contiguous_container_asan_correct(l1));
    }
    {
        std::vector<std::vector<int> > outer(2, std::vector<int>(1));
        assert(is_contiguous_container_asan_correct(outer));
        assert(is_contiguous_container_asan_correct(outer[0]));
        assert(is_contiguous_container_asan_correct(outer[1]));
        outer.erase(outer.begin(), outer.begin());
        assert(outer.size() == 2);
        assert(outer[0].size() == 1);
        assert(outer[1].size() == 1);
        assert(is_contiguous_container_asan_correct(outer));
        assert(is_contiguous_container_asan_correct(outer[0]));
        assert(is_contiguous_container_asan_correct(outer[1]));
    }
    // Make sure vector::erase works with move-only types
    {
        // When non-trivial
        {
            std::vector<MoveOnly> v;
            v.emplace_back(1); v.emplace_back(2); v.emplace_back(3);
            v.erase(v.begin(), v.begin() + 2);
            assert(v.size() == 1);
            assert(v[0] == MoveOnly(3));
        }
        // When trivial
        {
            std::vector<TrivialMoveOnly> v;
            v.emplace_back(1); v.emplace_back(2); v.emplace_back(3);
            v.erase(v.begin(), v.begin() + 2);
            assert(v.size() == 1);
            assert(v[0] == TrivialMoveOnly(3));
        }
    }
#if TEST_STD_VER >= 11
    {
        std::vector<int, min_allocator<int>> l1(a1, a1+3);
        assert(is_contiguous_container_asan_correct(l1));
        std::vector<int, min_allocator<int>>::iterator i = l1.erase(l1.cbegin(), l1.cbegin());
        assert(l1.size() == 3);
        assert(std::distance(l1.cbegin(), l1.cend()) == 3);
        assert(i == l1.begin());
        assert(is_contiguous_container_asan_correct(l1));
    }
    {
        std::vector<int, min_allocator<int>> l1(a1, a1+3);
        assert(is_contiguous_container_asan_correct(l1));
        std::vector<int, min_allocator<int>>::iterator i = l1.erase(l1.cbegin(), std::next(l1.cbegin()));
        assert(l1.size() == 2);
        assert(std::distance(l1.cbegin(), l1.cend()) == 2);
        assert(i == l1.begin());
        assert((l1 == std::vector<int, min_allocator<int>>(a1+1, a1+3)));
        assert(is_contiguous_container_asan_correct(l1));
    }
    {
        std::vector<int, min_allocator<int>> l1(a1, a1+3);
        assert(is_contiguous_container_asan_correct(l1));
        std::vector<int, min_allocator<int>>::iterator i = l1.erase(l1.cbegin(), std::next(l1.cbegin(), 2));
        assert(l1.size() == 1);
        assert(std::distance(l1.cbegin(), l1.cend()) == 1);
        assert(i == l1.begin());
        assert((l1 == std::vector<int, min_allocator<int>>(a1+2, a1+3)));
        assert(is_contiguous_container_asan_correct(l1));
    }
    {
        std::vector<int, min_allocator<int>> l1(a1, a1+3);
        assert(is_contiguous_container_asan_correct(l1));
        std::vector<int, min_allocator<int>>::iterator i = l1.erase(l1.cbegin(), std::next(l1.cbegin(), 3));
        assert(l1.size() == 0);
        assert(std::distance(l1.cbegin(), l1.cend()) == 0);
        assert(i == l1.begin());
        assert(is_contiguous_container_asan_correct(l1));
    }
    {
        std::vector<std::vector<int, min_allocator<int>>, min_allocator<std::vector<int, min_allocator<int>>>> outer(2, std::vector<int, min_allocator<int>>(1));
        assert(is_contiguous_container_asan_correct(outer));
        assert(is_contiguous_container_asan_correct(outer[0]));
        assert(is_contiguous_container_asan_correct(outer[1]));
        outer.erase(outer.begin(), outer.begin());
        assert(outer.size() == 2);
        assert(outer[0].size() == 1);
        assert(outer[1].size() == 1);
        assert(is_contiguous_container_asan_correct(outer));
        assert(is_contiguous_container_asan_correct(outer[0]));
        assert(is_contiguous_container_asan_correct(outer[1]));
    }
#endif

    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER > 17
    static_assert(tests());
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
// Test for LWG2853:
// Throws: Nothing unless an exception is thrown by the assignment operator or move assignment operator of T.
    {
        Throws arr[] = {1, 2, 3};
        std::vector<Throws> v(arr, arr+3);
        Throws::sThrows = true;
        v.erase(v.begin(), --v.end());
        assert(v.size() == 1);
        v.erase(v.begin(), v.end());
        assert(v.size() == 0);
    }
#endif

    return 0;
}
