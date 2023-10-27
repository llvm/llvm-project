//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <deque>

// Test based on: https://bugs.chromium.org/p/chromium/issues/detail?id=1419798#c5
// Some allocators during deallocation may not call destructors and just reuse memory.
// In those situations, one may want to deactivate annotations for a specific allocator.
// It's possible with __asan_annotate_container_with_allocator template class.
// This test confirms that those allocators work after turning off annotations.

#include <cassert>
#include <deque>
#include <new>

struct reuse_allocator {
  static size_t const N = 100;
  reuse_allocator() {
    for (size_t i = 0; i < N; ++i)
      __buffers[i] = new char[8 * 1024];
  }
  ~reuse_allocator() {
    for (size_t i = 0; i < N; ++i)
      delete[] (char*)__buffers[i];
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

#ifdef _LIBCPP_HAS_ASAN_CONTAINER_ANNOTATIONS_FOR_ALL_ALLOCATORS
template <class T>
struct std::__asan_annotate_container_with_allocator<user_allocator<T>> : false_type {};
#endif

int main(int, char**) {
  using D = std::deque<int, user_allocator<int>>;

  {
    D* d = new (reuse_buffers.alloc()) D();
    for (int i = 0; i < 100; i++)
      d->push_back(i);
  }
  reuse_buffers.reset();
  {
    D d;
    for (int i = 0; i < 1000; i++)
      d.push_back(i);
  }

  return 0;
}
