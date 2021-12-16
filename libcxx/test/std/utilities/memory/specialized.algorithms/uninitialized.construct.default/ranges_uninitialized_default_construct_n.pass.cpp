//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts, libcpp-has-no-incomplete-ranges

// <memory>

// template <nothrow-forward-iterator ForwardIterator>
//   requires default_initializable<iter_value_t<ForwardIterator>>
// ForwardIterator ranges::uninitialized_default_construct_n(ForwardIterator first,
//     iter_difference_t<ForwardIterator> n);

#include <cassert>
#include <memory>
#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"

struct Counted {
  static int current_objects;
  static int total_objects;
  static int throw_on;

  explicit Counted() {
    if (throw_on == total_objects) {
      TEST_THROW(1);
    }

    ++current_objects;
    ++total_objects;
  }

  ~Counted() { --current_objects; }

  static void reset() {
    current_objects = total_objects = 0;
    throw_on = -1;
  }

  Counted(Counted const&) = delete;
  friend void operator&(Counted) = delete;
};
int Counted::current_objects = 0;
int Counted::total_objects = 0;
int Counted::throw_on = -1;

template <typename T, int N>
struct Buffer {
  alignas(T) char buffer[sizeof(T) * N] = {};

  T* begin() { return reinterpret_cast<T*>(buffer); }
  T* end() { return begin() + N; }
  const T* cbegin() const { return reinterpret_cast<const T*>(buffer); }
  const T* cend() const { return cbegin() + N; }
};

// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::uninitialized_default_construct_n)>);

struct NotDefaultCtrable { NotDefaultCtrable() = delete; };
static_assert(!std::is_invocable_v<decltype(std::ranges::uninitialized_default_construct_n),
    NotDefaultCtrable*, int>);

int main(int, char**) {
  // An empty range -- no default constructors should be invoked.
  {
    Buffer<Counted, 1> buf;

    std::ranges::uninitialized_default_construct_n(buf.begin(), 0);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);
  }

  // A range containing several objects.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    std::ranges::uninitialized_default_construct_n(buf.begin(), N);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    Counted::throw_on = 3; // When constructing the fourth object (counting from one).
    try {
      std::ranges::uninitialized_default_construct_n(buf.begin(), N);
    } catch(...) {}
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 3);
    std::destroy(buf.begin(), buf.begin() + Counted::total_objects);
    Counted::reset();
  }
#endif  // TEST_HAS_NO_EXCEPTIONS

  // Works with const iterators.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    std::ranges::uninitialized_default_construct_n(buf.cbegin(), N);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  return 0;
}
