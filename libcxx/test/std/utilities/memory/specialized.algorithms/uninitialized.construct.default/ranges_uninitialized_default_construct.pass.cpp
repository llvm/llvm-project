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

// template <nothrow-forward-iterator ForwardIterator, nothrow-sentinel-for<ForwardIterator> Sentinel>
//   requires default_initializable<iter_value_t<ForwardIterator>>
// ForwardIterator ranges::uninitialized_default_construct(ForwardIterator first, Sentinel last);
//
// template <nothrow-forward-range ForwardRange>
//   requires default_initializable<range_value_t<ForwardRange>>
// borrowed_iterator_t<ForwardRange> ranges::uninitialized_default_construct(ForwardRange&& range);

#include <cassert>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>

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
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::uninitialized_default_construct)>);

struct NotDefaultCtrable { NotDefaultCtrable() = delete; };
static_assert(!std::is_invocable_v<decltype(std::ranges::uninitialized_default_construct),
    NotDefaultCtrable*, NotDefaultCtrable*>);

int main(int, char**) {
  // An empty range -- no default constructors should be invoked.
  {
    Buffer<Counted, 1> buf;

    std::ranges::uninitialized_default_construct(buf.begin(), buf.begin());
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);

    std::ranges::uninitialized_default_construct(std::ranges::empty_view<Counted>());
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);

    forward_iterator<Counted*> it(buf.begin());
    auto range = std::ranges::subrange(it, sentinel_wrapper<forward_iterator<Counted*>>(it));
    std::ranges::uninitialized_default_construct(range.begin(), range.end());
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);

    std::ranges::uninitialized_default_construct(range);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);
  }

  // A range containing several objects, (iter, sentinel) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, 5> buf;

    std::ranges::uninitialized_default_construct(buf.begin(), buf.end());
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // A range containing several objects, (range) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    auto range = std::ranges::subrange(buf.begin(), buf.end());
    std::ranges::uninitialized_default_construct(range);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // Using `counted_iterator`.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    std::ranges::uninitialized_default_construct(
        std::counted_iterator(buf.begin(), N), std::default_sentinel);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // Using `views::counted`.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    auto counted_range = std::views::counted(buf.begin(), N);
    std::ranges::uninitialized_default_construct(counted_range);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // Using `reverse_view`.
  {
    constexpr int N = 3;
    Buffer<Counted, 5> buf;

    auto range = std::ranges::subrange(buf.begin(), buf.begin() + N);
    std::ranges::uninitialized_default_construct(std::ranges::reverse_view(range));
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.begin() + N);
    Counted::reset();
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid. (iterator, sentinel) overload.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    Buffer<Counted, 5> buf;

    Counted::throw_on = 3; // When constructing the fourth object (counting from one).
    try {
      std::ranges::uninitialized_default_construct(buf.begin(), buf.end());
    } catch(...) {}
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 3);
    std::destroy(buf.begin(), buf.begin() + Counted::total_objects);
    Counted::reset();
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid. (range) overload.
  {
    Buffer<Counted, 5> buf;

    Counted::throw_on = 3; // When constructing the fourth object.
    try {
      auto range = std::ranges::subrange(buf.begin(), buf.end());
      std::ranges::uninitialized_default_construct(range);
    } catch(...) {}
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 3);
    std::destroy(buf.begin(), buf.begin() + Counted::total_objects);
    Counted::reset();
  }
#endif  // TEST_HAS_NO_EXCEPTIONS

  // Works with const iterators, (iter, sentinel) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    std::ranges::uninitialized_default_construct(buf.cbegin(), buf.cend());
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // Works with const iterators, (range) overload.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;
    auto range = std::ranges::subrange(buf.cbegin(), buf.cend());

    std::ranges::uninitialized_default_construct(range);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  return 0;
}
