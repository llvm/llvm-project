//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asan
// UNSUPPORTED: c++03

// Test based on: https://bugs.chromium.org/p/chromium/issues/detail?id=1419798#c5
// Some allocators during deallocation may not call destructors and just reuse memory.
// In those situations, one may want to deactivate annotations for a specific allocator.
// It's possible with __asan_annotate_container_with_allocator template class.
// This test confirms that those allocators work after turning off annotations.
//
// A context to this test is a situations when memory is repurposed and destructors are not called.
//   Related issue: https://llvm.org/PR60384
//
// That issue appeared in the past and was addressed here: https://reviews.llvm.org/D145628
//
// There was also a discussion, if it's UB.
//   Related discussion: https://reviews.llvm.org/D136765#4155262
//   Related notes: https://eel.is/c++draft/basic.life#6
// Probably it's no longer UB due a change in CWG2523.
//   https://cplusplus.github.io/CWG/issues/2523.html
//
// Therefore we make sure that it works that way, also because people rely on this behavior.
// Annotations are turned off only, if a user explicitly turns off annotations for a specific allocator.

#include <assert.h>
#include <stdlib.h>
#include <string>
#include <new>

// Allocator with pre-allocated (with malloc in constructor) buffers.
// Memory may be freed without calling destructors.
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

  T* allocate(size_t n) {
    if (n * sizeof(T) > 8 * 1024)
      throw std::bad_array_new_length();
    return (T*)reuse_buffers.alloc();
  }
  void deallocate(T*, size_t) noexcept {}
};

// Turn off annotations for user_allocator:
template <class T>
struct std::__asan_annotate_container_with_allocator<user_allocator<T>> {
  static bool const value = false;
};

int main(int, char**) {
  using S = std::basic_string<char, std::char_traits<char>, user_allocator<char>>;

  {
    // Create a string with a buffer from reuse allocator object:
    S* s = new (reuse_buffers.alloc()) S();
    // Use string, so it's poisoned, if container annotations for that allocator are not turned off:
    for (int i = 0; i < 40; i++)
      s->push_back('a');
  }
  // Reset the state of the allocator, don't call destructors, allow memory to be reused:
  reuse_buffers.reset();
  {
    // Create a next string with the same allocator, so the same buffer due to the reset:
    S s;
    // Use memory inside the string again, if it's poisoned, an error will be raised:
    for (int i = 0; i < 60; i++)
      s.push_back('a');
  }

  return 0;
}
