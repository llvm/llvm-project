//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asan
// UNSUPPORTED: c++03

// <string>

// Test based on: https://bugs.chromium.org/p/chromium/issues/detail?id=1419798#c5
// Some allocators during deallocation may not call destructors and just reuse memory.
// In those situations, one may want to deactivate annotations for a specific allocator.
// It's possible with __asan_annotate_container_with_allocator template class.
// This test confirms that those allocators work after turning off annotations.

#include <assert.h>
#include <stdlib.h>
#include <string>
#include <new>

struct reuse_allocator {
  static size_t const N = 100;
  reuse_allocator() {
    for (size_t i = 0; i < N; ++i)
      __buffers[i] = malloc(8 * 1024);
  }
  ~reuse_allocator() {
    for (size_t i = 0; i < N; ++i)
      free(__buffers[i]);
  }
  void* alloc() {
    assert(__next_id < N);
    return __buffers[__next_id++];
  }
  void reset() { __next_id = 0; }
  void* __buffers[N];
  size_t __next_id = 0;
} reuse_buffers;

template <typename T>
struct user_allocator {
  using value_type = T;
  user_allocator() = default;
  template <class U>
  user_allocator(user_allocator<U>) {}
  friend bool operator==(user_allocator, user_allocator) { return true; }
  friend bool operator!=(user_allocator x, user_allocator y) { return !(x == y); }

  T* allocate(size_t) { return (T*)reuse_buffers.alloc(); }
  void deallocate(T*, size_t) noexcept {}
};

template <class T>
struct std::__asan_annotate_container_with_allocator<user_allocator<T>> {
  static bool const value = false;
};

int main() {
  using S = std::basic_string<char, std::char_traits<char>, user_allocator<char>>;

  {
    S* s = new (reuse_buffers.alloc()) S();
    for (int i = 0; i < 100; i++)
      s->push_back('a');
  }
  reuse_buffers.reset();
  {
    S s;
    for (int i = 0; i < 1000; i++)
      s.push_back('b');
  }

  return 0;
}
