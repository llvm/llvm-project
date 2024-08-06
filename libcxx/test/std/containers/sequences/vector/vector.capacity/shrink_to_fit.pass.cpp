//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void shrink_to_fit();

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

TEST_CONSTEXPR_CXX20 bool tests() {
    {
        std::vector<int> v(100);
        v.push_back(1);
        assert(is_contiguous_container_asan_correct(v));
        v.shrink_to_fit();
        assert(v.capacity() == 101);
        assert(v.size() == 101);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int, limited_allocator<int, 401> > v(100);
        v.push_back(1);
        assert(is_contiguous_container_asan_correct(v));
        v.shrink_to_fit();
        assert(v.capacity() == 101);
        assert(v.size() == 101);
        assert(is_contiguous_container_asan_correct(v));
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    if (!TEST_IS_CONSTANT_EVALUATED) {
        std::vector<int, limited_allocator<int, 400> > v(100);
        v.push_back(1);
        assert(is_contiguous_container_asan_correct(v));
        v.shrink_to_fit();
        LIBCPP_ASSERT(v.capacity() == 200); // assumes libc++'s 2x growth factor
        assert(v.size() == 101);
        assert(is_contiguous_container_asan_correct(v));
    }
#endif
#if TEST_STD_VER >= 11
    {
        std::vector<int, min_allocator<int>> v(100);
        v.push_back(1);
        assert(is_contiguous_container_asan_correct(v));
        v.shrink_to_fit();
        assert(v.capacity() == 101);
        assert(v.size() == 101);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
      std::vector<int, safe_allocator<int>> v(100);
      v.push_back(1);
      assert(is_contiguous_container_asan_correct(v));
      v.shrink_to_fit();
      assert(v.capacity() == 101);
      assert(v.size() == 101);
      assert(is_contiguous_container_asan_correct(v));
    }
#endif

    return true;
}

#if TEST_STD_VER >= 23
template <typename T>
struct increasing_allocator {
  using value_type         = T;
  std::size_t min_elements = 1000;
  increasing_allocator()   = default;

  template <typename U>
  constexpr increasing_allocator(const increasing_allocator<U>& other) noexcept : min_elements(other.min_elements) {}

  constexpr std::allocation_result<T*> allocate_at_least(std::size_t n) {
    if (n < min_elements)
      n = min_elements;
    min_elements += 1000;
    return std::allocator<T>{}.allocate_at_least(n);
  }
  constexpr T* allocate(std::size_t n) { return allocate_at_least(n).ptr; }
  constexpr void deallocate(T* p, std::size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }
};

template <typename T, typename U>
bool operator==(increasing_allocator<T>, increasing_allocator<U>) {
  return true;
}

// https://github.com/llvm/llvm-project/issues/95161
constexpr bool test_increasing_allocator() {
  std::vector<int, increasing_allocator<int>> v;
  v.push_back(1);
  assert(is_contiguous_container_asan_correct(v));
  std::size_t capacity = v.capacity();
  v.shrink_to_fit();
  assert(v.capacity() <= capacity);
  assert(v.size() == 1);
  assert(is_contiguous_container_asan_correct(v));

  return true;
}
#endif // TEST_STD_VER >= 23

int main(int, char**)
{
  tests();
#if TEST_STD_VER > 17
    static_assert(tests());
#endif
#if TEST_STD_VER >= 23
    test_increasing_allocator();
    static_assert(test_increasing_allocator());
#endif

    return 0;
}
