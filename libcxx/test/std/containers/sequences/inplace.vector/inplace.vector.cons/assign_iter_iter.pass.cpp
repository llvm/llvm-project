//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// template <class InputIter> void assign(InputIter first, InputIter last);

#include <inplace_vector>
#include <algorithm>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"
#include "emplace_constructible.h"
#include "container_test_types.h"

constexpr bool test() {
  int arr1[] = {42};
  int arr2[] = {1, 101, 42};
  if !consteval {
    using T  = EmplaceConstructibleMoveableAndAssignable<int>;
    using It = forward_iterator<int*>;
    {
      std::inplace_vector<T, 10> v;
      v.assign(It(arr1), It(std::end(arr1)));
      assert(v[0].value == 42);
    }
    {
      std::inplace_vector<T, 10> v;
      v.assign(It(arr2), It(std::end(arr2)));
      assert(v[0].value == 1);
      assert(v[1].value == 101);
      assert(v[2].value == 42);
    }
  }
  if !consteval {
    using T  = EmplaceConstructibleMoveableAndAssignable<int>;
    using It = cpp17_input_iterator<int*>;
    {
      std::inplace_vector<T, 10> v;
      v.assign(It(arr1), It(std::end(arr1)));
      assert(v[0].copied == 0);
      assert(v[0].value == 42);
    }
    {
      std::inplace_vector<T, 10> v;
      v.assign(It(arr2), It(std::end(arr2)));
      //assert(v[0].copied == 0);
      assert(v[0].value == 1);
      //assert(v[1].copied == 0);
      assert(v[1].value == 101);
      assert(v[2].copied == 0);
      assert(v[2].value == 42);
    }
  }
  {
    using T  = int;
    using It = forward_iterator<int*>;
    {
      std::inplace_vector<T, 10> v;
      v.assign(It(arr1), It(std::end(arr1)));
      assert(v[0] == 42);
    }
    {
      std::inplace_vector<T, 10> v;
      v.assign(It(arr2), It(std::end(arr2)));
      assert(v[0] == 1);
      assert(v[1] == 101);
      assert(v[2] == 42);
    }
  }
  {
    using T  = int;
    using It = cpp17_input_iterator<int*>;
    {
      std::inplace_vector<T, 10> v;
      v.assign(It(arr1), It(std::end(arr1)));
      assert(v[0] == 42);
    }
    {
      std::inplace_vector<T, 10> v;
      v.assign(It(arr2), It(std::end(arr2)));
      assert(v[0] == 1);
      assert(v[1] == 101);
      assert(v[2] == 42);
    }
  }

  if !consteval {
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
      using It = forward_iterator<int*>;

      std::inplace_vector<int, 10> dst(10);
      int src[20]{};

      dst.assign(It(src + 0), It(src + 20));
      assert(false);
    } catch (const std::bad_alloc& e) {
      // OK
    } catch (...) {
      assert(false);
    }
#endif
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
