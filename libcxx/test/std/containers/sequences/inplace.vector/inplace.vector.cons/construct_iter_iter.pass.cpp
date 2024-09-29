//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// template <class InputIter> inplace_vector(InputIter first, InputIter last);

#include <inplace_vector>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_iterators.h"
#include "emplace_constructible.h"
#include "container_test_types.h"

template <class C, class Iterator>
constexpr void test(Iterator first, Iterator last) {
  {
    C c(first, last);
    assert(c.size() == static_cast<std::size_t>(std::distance(first, last)));
    for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e; ++i, ++first)
      assert(*i == *first);
  }
  // Test with an empty range
  {
    C c(first, first);
    assert(c.empty());
  }
}

constexpr void basic_test_cases() {
  int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 1, 0};
  int* an = a + sizeof(a) / sizeof(a[0]);
  using V = std::inplace_vector<int, 20>;
  test<V>(cpp17_input_iterator<const int*>(a), cpp17_input_iterator<const int*>(an));
  test<V>(forward_iterator<const int*>(a), forward_iterator<const int*>(an));
  test<V>(bidirectional_iterator<const int*>(a), bidirectional_iterator<const int*>(an));
  test<V>(random_access_iterator<const int*>(a), random_access_iterator<const int*>(an));
  test<V>(a, an);

  // Regression test for https://github.com/llvm/llvm-project/issues/46841
  {
    V v1({}, forward_iterator<const int*>{});
    V v2(forward_iterator<const int*>{}, {});
  }

  if !consteval {
    volatile int src[] = {1, 2, 3};
    std::inplace_vector<int, 3> v(src, src + 3);
    assert(v[0] == 1);
    assert(v[1] == 2);
    assert(v[2] == 3);
  }
}

constexpr void emplaceable_concept_tests() {
  int arr1[] = {42};
  int arr2[] = {1, 101, 42};
  if !consteval {
    using T  = EmplaceConstructible<int>;
    using It = forward_iterator<int*>;
    {
      std::inplace_vector<T, 10> v(It(arr1), It(std::end(arr1)));
      assert(v[0].value == 42);
    }
    {
      std::inplace_vector<T, 10> v(It(arr2), It(std::end(arr2)));
      assert(v[0].value == 1);
      assert(v[1].value == 101);
      assert(v[2].value == 42);
    }
  }

  if !consteval {
    using T  = EmplaceConstructibleAndMoveInsertable<int>;
    using It = cpp17_input_iterator<int*>;
    {
      std::inplace_vector<T, 10> v(It(arr1), It(std::end(arr1)));
      assert(v[0].copied == 0);
      assert(v[0].value == 42);
    }
    {
      std::inplace_vector<T, 10> v(It(arr2), It(std::end(arr2)));
      //assert(v[0].copied == 0);
      assert(v[0].value == 1);
      //assert(v[1].copied == 0);
      assert(v[1].value == 101);
      assert(v[2].copied == 0);
      assert(v[2].value == 42);
    }
  }
}

constexpr void overcapacity_tests() {
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
}

struct B1 {
  int x;
};
struct B2 {
  int y;
};
struct Der : B1, B2 {
  int z;
};

// Initialize a vector with a different value type.
constexpr void test_ctor_with_different_value_type() {
  {
    // Make sure initialization is performed with each element value, not with
    // a memory blob.
    float array[3] = {0.0f, 1.0f, 2.0f};
    TEST_DIAGNOSTIC_PUSH
    TEST_MSVC_DIAGNOSTIC_IGNORED(4244) // conversion from 'float' to 'int', possible loss of data
    std::inplace_vector<int, 10> v(array, array + 3);
    TEST_DIAGNOSTIC_POP
    assert(v.size() == 3);
    assert(v[0] == 0);
    assert(v[1] == 1);
    assert(v[2] == 2);
  }
  {
    Der z;
    Der* array[1] = {&z};
    // Though the types Der* and B2* are very similar, initialization still cannot
    // be done with `memcpy`.
    std::inplace_vector<B2*, 10> v(array, array + 1);
    assert(v.size() == 1);
    assert(v[0] == &z);
  }
  {
    // Though the types are different, initialization can be done with `memcpy`.
    std::int32_t array[1] = {-1};
    std::inplace_vector<std::uint32_t, 10> v(array, array + 1);
    assert(v.size() == 1);
    assert(v[0] == 4294967295U);
  }
}

constexpr bool tests() {
  basic_test_cases();
  emplaceable_concept_tests(); // See PR34898
  test_ctor_with_different_value_type();
  overcapacity_tests();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
