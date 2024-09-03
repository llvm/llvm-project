//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// void shrink_to_fit();

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 bool tests()
{
    {
        std::vector<bool> v(100);
        v.push_back(1);
        v.shrink_to_fit();
        assert(v.capacity() >= 101);
        assert(v.size() >= 101);
    }
#if TEST_STD_VER >= 11
    {
        std::vector<bool, min_allocator<bool>> v(100);
        v.push_back(1);
        v.shrink_to_fit();
        assert(v.capacity() >= 101);
        assert(v.size() >= 101);
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
  std::vector<bool, increasing_allocator<bool>> v;
  v.push_back(1);
  std::size_t capacity = v.capacity();
  v.shrink_to_fit();
  assert(v.capacity() <= capacity);
  assert(v.size() == 1);

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
#endif // TEST_STD_VER >= 23

    return 0;
}
