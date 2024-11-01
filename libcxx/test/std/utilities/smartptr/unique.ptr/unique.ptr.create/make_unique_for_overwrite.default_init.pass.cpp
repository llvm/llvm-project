//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: sanitizer-new-delete

// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(); // T is not array
//
// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(size_t n); // T is U[]

// Test the object is not value initialized

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <memory>

constexpr char pattern = 0xDE;

void* operator new(std::size_t count) {
  void* ptr = std::malloc(count);
  for (std::size_t i = 0; i < count; ++i) {
    *(reinterpret_cast<char*>(ptr) + i) = pattern;
  }
  return ptr;
}

void* operator new[](std::size_t count) { return ::operator new(count); }

void operator delete(void* ptr) noexcept { std::free(ptr); }

void operator delete[](void* ptr) noexcept { ::operator delete(ptr); }

void test() {
  {
    std::same_as<std::unique_ptr<int>> auto ptr = std::make_unique_for_overwrite<int>();
    assert(*(reinterpret_cast<char*>(ptr.get())) == pattern);
  }
  {
    std::same_as<std::unique_ptr<int[]>> auto ptr = std::make_unique_for_overwrite<int[]>(3);
    assert(*(reinterpret_cast<char*>(&ptr[0])) == pattern);
    assert(*(reinterpret_cast<char*>(&ptr[1])) == pattern);
    assert(*(reinterpret_cast<char*>(&ptr[2])) == pattern);
  }
}

int main(int, char**) {
  test();

  return 0;
}
