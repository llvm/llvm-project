//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// template <class InputIter> vector(InputIter first, InputIter last);

#include <vector>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"
#if TEST_STD_VER >= 11
#include "emplace_constructible.h"
#include "container_test_types.h"
#endif

template <class C, class Iterator>
void test(Iterator first, Iterator last) {
  {
    C c(first, last);
    LIBCPP_ASSERT(c.__invariants());
    assert(c.size() == static_cast<std::size_t>(std::distance(first, last)));
    LIBCPP_ASSERT(is_contiguous_container_asan_correct(c));
    for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e;
      ++i, ++first)
    assert(*i == *first);
  }
  // Test with an empty range
  {
    C c(first, first);
    LIBCPP_ASSERT(c.__invariants());
    assert(c.empty());
    LIBCPP_ASSERT(is_contiguous_container_asan_correct(c));
  }
}

static void basic_test_cases() {
  int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 1, 0};
  int* an = a + sizeof(a) / sizeof(a[0]);
  test<std::vector<int> >(cpp17_input_iterator<const int*>(a),
                          cpp17_input_iterator<const int*>(an));
  test<std::vector<int> >(forward_iterator<const int*>(a),
                          forward_iterator<const int*>(an));
  test<std::vector<int> >(bidirectional_iterator<const int*>(a),
                          bidirectional_iterator<const int*>(an));
  test<std::vector<int> >(random_access_iterator<const int*>(a),
                          random_access_iterator<const int*>(an));
  test<std::vector<int> >(a, an);

  test<std::vector<int, limited_allocator<int, 63> > >(
      cpp17_input_iterator<const int*>(a), cpp17_input_iterator<const int*>(an));
  // Add 1 for implementations that dynamically allocate a container proxy.
  test<std::vector<int, limited_allocator<int, 18 + 1> > >(
      forward_iterator<const int*>(a), forward_iterator<const int*>(an));
  test<std::vector<int, limited_allocator<int, 18 + 1> > >(
      bidirectional_iterator<const int*>(a),
      bidirectional_iterator<const int*>(an));
  test<std::vector<int, limited_allocator<int, 18 + 1> > >(
      random_access_iterator<const int*>(a),
      random_access_iterator<const int*>(an));
  test<std::vector<int, limited_allocator<int, 18 + 1> > >(a, an);
#if TEST_STD_VER >= 11
  test<std::vector<int, min_allocator<int> > >(cpp17_input_iterator<const int*>(a),
                                               cpp17_input_iterator<const int*>(an));
  test<std::vector<int, min_allocator<int> > >(
      forward_iterator<const int*>(a), forward_iterator<const int*>(an));
  test<std::vector<int, min_allocator<int> > >(
      bidirectional_iterator<const int*>(a),
      bidirectional_iterator<const int*>(an));
  test<std::vector<int, min_allocator<int> > >(
      random_access_iterator<const int*>(a),
      random_access_iterator<const int*>(an));
  test<std::vector<int> >(a, an);
#endif
}

void emplaceable_concept_tests() {
#if TEST_STD_VER >= 11
  int arr1[] = {42};
  int arr2[] = {1, 101, 42};
  {
    using T = EmplaceConstructible<int>;
    using It = forward_iterator<int*>;
    {
      std::vector<T> v(It(arr1), It(std::end(arr1)));
      assert(v[0].value == 42);
    }
    {
      std::vector<T> v(It(arr2), It(std::end(arr2)));
      assert(v[0].value == 1);
      assert(v[1].value == 101);
      assert(v[2].value == 42);
    }
  }
  {
    using T = EmplaceConstructibleAndMoveInsertable<int>;
    using It = cpp17_input_iterator<int*>;
    {
      std::vector<T> v(It(arr1), It(std::end(arr1)));
      assert(v[0].copied == 0);
      assert(v[0].value == 42);
    }
    {
      std::vector<T> v(It(arr2), It(std::end(arr2)));
      //assert(v[0].copied == 0);
      assert(v[0].value == 1);
      //assert(v[1].copied == 0);
      assert(v[1].value == 101);
      assert(v[2].copied == 0);
      assert(v[2].value == 42);
    }
  }
#endif
}

void test_ctor_under_alloc() {
#if TEST_STD_VER >= 11
  int arr1[] = {42};
  int arr2[] = {1, 101, 42};
  {
    using C = TCT::vector<>;
    using It = forward_iterator<int*>;
    {
      ExpectConstructGuard<int&> G(1);
      C v(It(arr1), It(std::end(arr1)));
    }
    {
      ExpectConstructGuard<int&> G(3);
      C v(It(arr2), It(std::end(arr2)));
    }
  }
  {
    using C = TCT::vector<>;
    using It = cpp17_input_iterator<int*>;
    {
      ExpectConstructGuard<int&> G(1);
      C v(It(arr1), It(std::end(arr1)));
    }
    {
      //ExpectConstructGuard<int&> G(3);
      //C v(It(arr2), It(std::end(arr2)), a);
    }
  }
#endif
}

// In C++03, you can't instantiate a template with a local type.
struct B1 { int x; };
struct B2 { int y; };
struct Der : B1, B2 { int z; };

// Initialize a vector with a different value type.
void test_ctor_with_different_value_type() {
  {
    // Make sure initialization is performed with each element value, not with
    // a memory blob.
    float array[3] = {0.0f, 1.0f, 2.0f};
    TEST_DIAGNOSTIC_PUSH
    TEST_MSVC_DIAGNOSTIC_IGNORED(4244) // conversion from 'float' to 'int', possible loss of data
    std::vector<int> v(array, array + 3);
    TEST_DIAGNOSTIC_POP
    assert(v[0] == 0);
    assert(v[1] == 1);
    assert(v[2] == 2);
  }
  {
    Der z;
    Der *array[1] = { &z };
    // Though the types Der* and B2* are very similar, initialization still cannot
    // be done with `memcpy`.
    std::vector<B2*> v(array, array + 1);
    assert(v[0] == &z);
  }
  {
    // Though the types are different, initialization can be done with `memcpy`.
    int32_t array[1] = { -1 };
    std::vector<uint32_t> v(array, array + 1);
    assert(v[0] == 4294967295U);
  }
}


int main(int, char**) {
  basic_test_cases();
  emplaceable_concept_tests(); // See PR34898
  test_ctor_under_alloc();
  test_ctor_with_different_value_type();

  return 0;
}
