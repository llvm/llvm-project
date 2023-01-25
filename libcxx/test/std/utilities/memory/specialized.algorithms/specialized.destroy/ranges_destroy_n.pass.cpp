//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>
//
// namespace ranges {
//   template<nothrow-input-iterator InputIterator>
//     requires destructible<iter_value_t<InputIterator>>
//     constexpr InputIterator destroy_n(InputIterator first, iter_difference_t<InputIterator> n) noexcept; // since C++20
// }

#include <cassert>
#include <memory>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::destroy_n)>);

struct NotNothrowDtrable {
  ~NotNothrowDtrable() noexcept(false) {}
};
static_assert(!std::is_invocable_v<decltype(std::ranges::destroy_n), NotNothrowDtrable*, int>);

struct Counted {
  int& count;

  constexpr Counted(int& count_ref) : count(count_ref) { ++count; }
  constexpr Counted(const Counted& rhs) : count(rhs.count) { ++count; }
  constexpr ~Counted() { --count; }

  friend void operator&(Counted) = delete;
};

template <class Iterator>
constexpr void test() {
  {
    constexpr int N = 5;
    std::allocator<Counted> alloc;
    using Traits = std::allocator_traits<decltype(alloc)>;
    int counter = 0;

    Counted* out = Traits::allocate(alloc, N);
    for (int i = 0; i != N; ++i) {
      Traits::construct(alloc, out + i, counter);
    }
    assert(counter == N);

    std::ranges::destroy_n(Iterator(out), N);
    assert(counter == 0);

    Traits::deallocate(alloc, out, N);
  }
}

constexpr bool tests() {
  test<Counted*>();
  test<forward_iterator<Counted*>>();

  return true;
}

constexpr bool test_arrays() {
  // One-dimensional array.
  {
    constexpr int N = 5;
    constexpr int M = 3;

    using Array = Counted[M];
    std::allocator<Array> alloc;
    using Traits = std::allocator_traits<decltype(alloc)>;
    int counter = 0;

    Array* buffer = Traits::allocate(alloc, N);
    for (int i = 0; i != N; ++i) {
      Array& array_ref = *(buffer + i);
      for (int j = 0; j != M; ++j) {
        Traits::construct(alloc, std::addressof(array_ref[j]), counter);
      }
    }
    assert(counter == N * M);

    std::ranges::destroy_n(buffer, N);
    assert(counter == 0);

    Traits::deallocate(alloc, buffer, N);
  }

  // Multidimensional array.
  {
    constexpr int N = 5;
    constexpr int A = 3;
    constexpr int B = 3;

    using Array = Counted[A][B];
    std::allocator<Array> alloc;
    using Traits = std::allocator_traits<decltype(alloc)>;
    int counter = 0;

    Array* buffer = Traits::allocate(alloc, N);
    for (int i = 0; i != N; ++i) {
      Array& array_ref = *(buffer + i);
      for (int j = 0; j != A; ++j) {
        for (int k = 0; k != B; ++k) {
          Traits::construct(alloc, std::addressof(array_ref[j][k]), counter);
        }
      }
    }
    assert(counter == N * A * B);

    std::ranges::destroy_n(buffer, N);
    assert(counter == 0);

    Traits::deallocate(alloc, buffer, N);
  }

  return true;
}

int main(int, char**) {
  tests();
  test_arrays();

  static_assert(tests());
  // TODO: Until std::construct_at has support for arrays, it's impossible to test this
  //       in a constexpr context (see https://reviews.llvm.org/D114903).
  // static_assert(test_arrays());

  return 0;
}
