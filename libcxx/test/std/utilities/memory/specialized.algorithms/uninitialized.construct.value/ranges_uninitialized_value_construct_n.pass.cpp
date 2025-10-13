//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// template <nothrow-forward-iterator ForwardIterator>
//   requires default_initializable<iter_value_t<ForwardIterator>>
// ForwardIterator ranges::uninitialized_value_construct_n(ForwardIterator first,
//     iter_difference_t<ForwardIterator> n);

#include <cassert>
#include <memory>
#include <ranges>

#include "../buffer.h"
#include "../counted.h"
#include "test_macros.h"
#include "test_iterators.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::uninitialized_value_construct_n)>);

struct NotDefaultCtrable {
  NotDefaultCtrable() = delete;
};
static_assert(!std::is_invocable_v<decltype(std::ranges::uninitialized_value_construct_n), NotDefaultCtrable*, int>);

int main(int, char**) {
  // An empty range -- no default constructors should be invoked.
  {
    Buffer<Counted, 1> buf;

    std::ranges::uninitialized_value_construct_n(buf.begin(), 0);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);
  }

  // A range containing several objects.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    std::ranges::uninitialized_value_construct_n(buf.begin(), N);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // Any existing values should be overwritten by value constructors.
  {
    constexpr int N = 5;
    int buffer[N] = {42, 42, 42, 42, 42};

    std::ranges::uninitialized_value_construct_n(buffer, 1);
    assert(buffer[0] == 0);
    assert(buffer[1] == 42);

    std::ranges::uninitialized_value_construct_n(buffer, N);
    assert(buffer[0] == 0);
    assert(buffer[1] == 0);
    assert(buffer[2] == 0);
    assert(buffer[3] == 0);
    assert(buffer[4] == 0);
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    Counted::throw_on = 3; // When constructing the fourth object (counting from one).
    try {
      std::ranges::uninitialized_value_construct_n(buf.begin(), N);
    } catch (...) {
    }
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 3);
    std::destroy(buf.begin(), buf.begin() + Counted::total_objects);
    Counted::reset();
  }
#endif // TEST_HAS_NO_EXCEPTIONS

  return 0;
}
