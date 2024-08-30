//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// reverse_iterator       rbegin();
// reverse_iterator       rend();
// const_reverse_iterator rbegin()  const;
// const_reverse_iterator rend()    const;
// const_reverse_iterator crbegin() const;
// const_reverse_iterator crend()   const;

#include <inplace_vector>
#include <cassert>
#include <iterator>

#include "MoveOnly.h"

template <class Vector, bool IsConstexpr>
constexpr void check_vector_reverse_iterators() {
  bool ConstexprTests = true;
  if consteval {
    ConstexprTests = IsConstexpr;
  }

  if (ConstexprTests) {
    Vector vec;
    assert(vec.rbegin() == vec.rend());
    assert(vec.crbegin() == vec.crend());
  }
  if (ConstexprTests && Vector::capacity() >= 10) {
    const int n = 10;
    Vector vec;
    const Vector& cvec = vec;
    vec.reserve(n);
    for (int i = 0; i < n; ++i)
      vec.push_back(i);
    {
      int iterations = 0;

      for (typename Vector::const_reverse_iterator it = vec.crbegin(); it != vec.crend(); ++it) {
        assert(*it == (n - iterations - 1));
        ++iterations;
      }
      assert(iterations == n);
    }
    {
      assert(cvec.rbegin() == vec.crbegin());
      assert(cvec.rend() == vec.crend());
    }
    {
      int iterations = 0;

      for (typename Vector::reverse_iterator it = vec.rbegin(); it != vec.rend(); ++it) {
        assert(*it == (n - iterations - 1));
        *it = 40;
        assert(*it == 40);
        ++iterations;
      }
      assert(iterations == n);
    }

    assert(std::distance(vec.rbegin(), vec.rend()) == n);
    assert(std::distance(cvec.rbegin(), cvec.rend()) == n);
    assert(std::distance(vec.crbegin(), vec.crend()) == n);
    assert(std::distance(cvec.crbegin(), cvec.crend()) == n);
  }
}

constexpr bool test() {
  check_vector_reverse_iterators<std::inplace_vector<int, 0>, true>();
  check_vector_reverse_iterators<std::inplace_vector<int, 10>, true>();
  check_vector_reverse_iterators<std::inplace_vector<int, 100>, true>();
  check_vector_reverse_iterators<std::inplace_vector<MoveOnly, 0>, true>();
  check_vector_reverse_iterators<std::inplace_vector<MoveOnly, 10>, false>();
  check_vector_reverse_iterators<std::inplace_vector<MoveOnly, 100>, false>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
