//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void shrink_to_fit(); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s) {
  typename S::size_type old_cap = s.capacity();
  S s0                          = s;
  s.shrink_to_fit();
  LIBCPP_ASSERT(s.__invariants());
  assert(s == s0);
  assert(s.capacity() <= old_cap);
  assert(s.capacity() >= s.size());
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  S s;
  test(s);

  s.assign(10, 'a');
  s.erase(5);
  test(s);

  s.assign(50, 'a');
  s.erase(5);
  test(s);

  s.assign(100, 'a');
  s.erase(50);
  test(s);

  s.assign(100, 'a');
  for (int i = 0; i <= 9; ++i) {
    s.erase(90 - 10 * i);
    test(s);
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char>>>();
#endif

  return true;
}

#if TEST_STD_VER >= 23
std::size_t min_bytes = 1000;

template <typename T>
struct increasing_allocator {
  using value_type       = T;
  increasing_allocator() = default;
  template <typename U>
  increasing_allocator(const increasing_allocator<U>&) noexcept {}
  std::allocation_result<T*> allocate_at_least(std::size_t n) {
    std::size_t allocation_amount = n * sizeof(T);
    if (allocation_amount < min_bytes)
      allocation_amount = min_bytes;
    min_bytes += 1000;
    return {static_cast<T*>(::operator new(allocation_amount)), allocation_amount / sizeof(T)};
  }
  T* allocate(std::size_t n) { return allocate_at_least(n).ptr; }
  void deallocate(T* p, std::size_t) noexcept { ::operator delete(static_cast<void*>(p)); }
};

template <typename T, typename U>
bool operator==(increasing_allocator<T>, increasing_allocator<U>) {
  return true;
}

// https://github.com/llvm/llvm-project/issues/95161
void test_increasing_allocator() {
  std::basic_string<char, std::char_traits<char>, increasing_allocator<char>> s{
      "String does not fit in the internal buffer"};
  std::size_t capacity = s.capacity();
  std::size_t size     = s.size();
  s.shrink_to_fit();
  assert(s.capacity() <= capacity);
  assert(s.size() == size);
  LIBCPP_ASSERT(is_string_asan_correct(s));
}
#endif // TEST_STD_VER >= 23

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  test_increasing_allocator();
#endif
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
