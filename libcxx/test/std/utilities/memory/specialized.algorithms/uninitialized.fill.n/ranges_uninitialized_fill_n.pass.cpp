//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// template <nothrow-forward-iterator ForwardIterator, class T>
//   requires constructible_from<iter_value_t<ForwardIterator>, const T&>
// ForwardIterator ranges::uninitialized_fill_n(ForwardIterator first, iter_difference_t<ForwardIterator> n);

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>

#include "../buffer.h"
#include "../counted.h"
#include "copy_move_types.h"
#include "test_macros.h"
#include "test_iterators.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::uninitialized_fill_n)>);

struct NotConvertibleFromInt {};
static_assert(!std::is_invocable_v<decltype(std::ranges::uninitialized_fill_n), NotConvertibleFromInt*, std::size_t, int>);

TEST_CONSTEXPR_CXX26 bool test() {
  constexpr int n = 3;
  ConstCopy value(42);
  std::allocator<ConstCopy> alloc;

  ConstCopy* out = alloc.allocate(n);
  auto result    = std::ranges::uninitialized_fill_n(out, n, value);
  assert(result == out + n);
  for (int i = 0; i != n; ++i)
    assert(out[i].val == value.val);

  std::destroy(out, out + n);
  alloc.deallocate(out, n);

  return true;
}

int main(int, char**) {
  constexpr int value = 42;
  Counted x(value);
  Counted::reset();
  auto pred = [](const Counted& e) { return e.value == value; };

  // An empty range -- no default constructors should be invoked.
  {
    Buffer<Counted, 1> buf;

    std::ranges::uninitialized_fill_n(buf.begin(), 0, x);
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 0);
  }

  // A range containing several objects.
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    std::ranges::uninitialized_fill_n(buf.begin(), N, x);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::all_of(buf.begin(), buf.end(), pred));

    std::destroy(buf.begin(), buf.end());
    Counted::reset();
  }

  // Any existing values should be overwritten by value constructors.
  {
    constexpr int N = 5;
    int buffer[N] = {value, value, value, value, value};

    std::ranges::uninitialized_fill_n(buffer, 1, 0);
    assert(buffer[0] == 0);
    assert(buffer[1] == value);

    std::ranges::uninitialized_fill_n(buffer, N, 0);
    assert(buffer[0] == 0);
    assert(buffer[1] == 0);
    assert(buffer[2] == 0);
    assert(buffer[3] == 0);
    assert(buffer[4] == 0);
  }

  // An exception is thrown while objects are being created -- the existing objects should stay
  // valid. (iterator, sentinel) overload.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    constexpr int N = 5;
    Buffer<Counted, N> buf;

    Counted::throw_on = 3; // When constructing the fourth object.
    try {
      std::ranges::uninitialized_fill_n(buf.begin(), N, x);
    } catch (...) {
    }
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == 3);

    std::destroy(buf.begin(), buf.begin() + 3);
    Counted::reset();
  }
#endif // TEST_HAS_NO_EXCEPTIONS

  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
