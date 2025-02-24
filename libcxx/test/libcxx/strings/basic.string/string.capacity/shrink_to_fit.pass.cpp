//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string>

// void shrink_to_fit(); // constexpr since C++20

#include <cassert>
#include <string>

#include "asan_testing.h"
#include "increasing_allocator.h"

template <typename T>
struct oversizing_allocator {
  using value_type       = T;
  oversizing_allocator() = default;
  template <typename U>
  oversizing_allocator(const oversizing_allocator<U>&) noexcept {}
  std::allocation_result<T*> allocate_at_least(std::size_t n) {
    ++n;
    return {static_cast<T*>(::operator new(n * sizeof(T))), n};
  }
  T* allocate(std::size_t n) { return allocate_at_least(n).ptr; }
  void deallocate(T* p, std::size_t) noexcept { ::operator delete(static_cast<void*>(p)); }
};

template <typename T, typename U>
bool operator==(oversizing_allocator<T>, oversizing_allocator<U>) {
  return true;
}

// Make sure we use an allocation returned by allocate_at_least if it is smaller than the current allocation
// even if it contains more bytes than we requested.
// Fix issue: https://github.com/llvm/llvm-project/pull/115659
void test_oversizing_allocator() {
  std::basic_string<char, std::char_traits<char>, oversizing_allocator<char>> s{
      "String does not fit in the internal buffer and is a bit longer"};
  s                    = "String does not fit in the internal buffer";
  std::size_t capacity = s.capacity();
  std::size_t size     = s.size();
  s.shrink_to_fit();
  assert(s.capacity() < capacity);
  assert(s.size() == size);
}

// Make sure libc++ shrink_to_fit does NOT swap buffer with equal allocation sizes
void test_no_swap_with_equal_allocation_size() {
  { // Test with custom allocator with a minimum allocation size
    std::basic_string<char, std::char_traits<char>, min_size_allocator<128, char> > s(
        "A long string exceeding SSO limit but within min alloc size");
    std::size_t capacity = s.capacity();
    std::size_t size     = s.size();
    auto data            = s.data();
    s.shrink_to_fit();
    assert(s.capacity() <= capacity);
    assert(s.size() == size);
    assert(is_string_asan_correct(s));
    assert(s.capacity() == capacity && s.data() == data);
  }
  { // Test with custom allocator with a minimum power of two allocation size
    std::basic_string<char, std::char_traits<char>, pow2_allocator<char> > s(
        "This is a long string that exceeds the SSO limit");
    std::size_t capacity = s.capacity();
    std::size_t size     = s.size();
    auto data            = s.data();
    s.shrink_to_fit();
    assert(s.capacity() <= capacity);
    assert(s.size() == size);
    assert(is_string_asan_correct(s));
    assert(s.capacity() == capacity && s.data() == data);
  }
}

int main(int, char**) {
  test_oversizing_allocator();
  test_no_swap_with_equal_allocation_size();

  return 0;
}
