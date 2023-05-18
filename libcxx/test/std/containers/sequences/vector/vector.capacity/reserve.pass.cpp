//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void reserve(size_type n);

#include <vector>
#include <cassert>
#include <stdexcept>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

TEST_CONSTEXPR_CXX20 bool tests() {
    {
        std::vector<int> v;
        v.reserve(10);
        assert(v.capacity() >= 10);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int> v(100);
        assert(v.capacity() == 100);
        v.reserve(50);
        assert(v.size() == 100);
        assert(v.capacity() == 100);
        v.reserve(150);
        assert(v.size() == 100);
        assert(v.capacity() == 150);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        // Add 1 for implementations that dynamically allocate a container proxy.
        std::vector<int, limited_allocator<int, 250 + 1> > v(100);
        assert(v.capacity() == 100);
        v.reserve(50);
        assert(v.size() == 100);
        assert(v.capacity() == 100);
        v.reserve(150);
        assert(v.size() == 100);
        assert(v.capacity() == 150);
        assert(is_contiguous_container_asan_correct(v));
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    if (!TEST_IS_CONSTANT_EVALUATED) {
        std::vector<int> v;
        std::size_t sz = v.max_size() + 1;

        try {
            v.reserve(sz);
            assert(false);
        } catch (const std::length_error&) {
            assert(v.size() == 0);
            assert(v.capacity() == 0);
        }
    }
    if (!TEST_IS_CONSTANT_EVALUATED) {
        std::vector<int> v(10, 42);
        int* previous_data = v.data();
        std::size_t previous_capacity = v.capacity();
        std::size_t sz = v.max_size() + 1;

        try {
            v.reserve(sz);
            assert(false);
        } catch (std::length_error&) {
            assert(v.size() == 10);
            assert(v.capacity() == previous_capacity);
            assert(v.data() == previous_data);

            for (int i = 0; i < 10; ++i) {
                assert(v[i] == 42);
            }
        }
    }
#endif
#if TEST_STD_VER >= 11
    {
        std::vector<int, min_allocator<int>> v;
        v.reserve(10);
        assert(v.capacity() >= 10);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int, min_allocator<int>> v(100);
        assert(v.capacity() == 100);
        v.reserve(50);
        assert(v.size() == 100);
        assert(v.capacity() == 100);
        v.reserve(150);
        assert(v.size() == 100);
        assert(v.capacity() == 150);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
      std::vector<int, safe_allocator<int>> v;
      v.reserve(10);
      assert(v.capacity() >= 10);
      assert(is_contiguous_container_asan_correct(v));
    }
    {
      std::vector<int, safe_allocator<int>> v(100);
      assert(v.capacity() == 100);
      v.reserve(50);
      assert(v.size() == 100);
      assert(v.capacity() == 100);
      v.reserve(150);
      assert(v.size() == 100);
      assert(v.capacity() == 150);
      assert(is_contiguous_container_asan_correct(v));
    }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    if (!TEST_IS_CONSTANT_EVALUATED) {
        std::vector<int, limited_allocator<int, 100> > v;
        v.reserve(50);
        assert(v.capacity() == 50);
        assert(is_contiguous_container_asan_correct(v));
        try {
            v.reserve(101);
            assert(false);
        } catch (const std::length_error&) {
            // no-op
        }
        assert(v.capacity() == 50);
        assert(is_contiguous_container_asan_correct(v));
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

  return 0;
}
