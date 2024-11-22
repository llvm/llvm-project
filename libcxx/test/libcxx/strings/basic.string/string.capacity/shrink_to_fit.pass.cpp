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

// Make sure we use an allocation returned by allocate_at_least if it is smaller than the current allocation
// even if it contains more bytes than we requested

#include <cassert>
#include <string>

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

int main(int, char**) {
  test_oversizing_allocator();

  return 0;
}
