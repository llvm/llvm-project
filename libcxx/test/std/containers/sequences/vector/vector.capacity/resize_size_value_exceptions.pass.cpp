//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// Check that std::vector<T>::resize(size_type sz, const value_type& x) provides the strong exception guarantee
// if T is Cpp17CopyInsertable.

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "count_new.h"
#include "increasing_allocator.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"

template <typename T, typename Alloc>
void test_allocation_exception_for_strong_guarantee(
    std::vector<T, Alloc>& v, const std::vector<T>& values, std::size_t new_size) {
  assert(v.size() == values.size());
  T* old_data          = v.data();
  std::size_t old_size = v.size();
  std::size_t old_cap  = v.capacity();

  try {
    v.resize(new_size, values.empty() ? T() : values[0]);
  } catch (...) { // std::length_error, std::bad_alloc
    assert(v.data() == old_data);
    assert(v.size() == old_size);
    assert(v.capacity() == old_cap);
    for (std::size_t i = 0; i < v.size(); ++i)
      assert(v[i] == values[i]);
  }
}

template <typename T, typename Alloc>
void test_copy_ctor_exception_for_strong_guarantee(std::vector<throwing_data<T>, Alloc>& v,
                                                   const std::vector<T>& values) {
  assert(v.empty() && !values.empty());
  int throw_after = values.size() + values.size() / 2; // Trigger an exception halfway through reallocation
  v.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    v.emplace_back(values[i], throw_after);

  throwing_data<T>* old_data = v.data();
  std::size_t old_size       = v.size();
  std::size_t old_cap        = v.capacity();
  std::size_t new_size       = 2 * old_cap;

  try {
    int n = new_size - old_size + 1;
    throwing_data<T> t(T(), n);
    v.resize(new_size, t);
  } catch (...) {
    assert(v.data() == old_data);
    assert(v.size() == old_size);
    assert(v.capacity() == old_cap);
    for (std::size_t i = 0; i < v.size(); ++i)
      assert(v[i].data_ == values[i]);
  }
}

// Check the strong exception guarantee during reallocation failures
void test_allocation_exceptions() {
  //
  // Tests for std::length_error during reallocation failures
  //
  {
    std::vector<int> v;
    test_allocation_exception_for_strong_guarantee(v, std::vector<int>(), v.max_size() + 1);
  }
  check_new_delete_called();

  {
    int a[] = {1, 2, 3, 4, 5};
    std::vector<int> in(a, a + sizeof(a) / sizeof(a[0]));
    std::vector<int> v(in.begin(), in.end());
    test_allocation_exception_for_strong_guarantee(v, in, v.max_size() + 1);
  }
  check_new_delete_called();

  {
    int a[] = {1, 2, 3, 4, 5};
    std::vector<int> in(a, a + sizeof(a) / sizeof(a[0]));
    std::vector<int, min_allocator<int> > v(in.begin(), in.end());
    test_allocation_exception_for_strong_guarantee(v, in, v.max_size() + 1);
  }
  check_new_delete_called();

  {
    int a[] = {1, 2, 3, 4, 5};
    std::vector<int> in(a, a + sizeof(a) / sizeof(a[0]));
    std::vector<int, safe_allocator<int> > v(in.begin(), in.end());
    test_allocation_exception_for_strong_guarantee(v, in, v.max_size() + 1);
  }
  check_new_delete_called();

  {
    int a[] = {1, 2, 3, 4, 5};
    std::vector<int> in(a, a + sizeof(a) / sizeof(a[0]));
    std::vector<int, test_allocator<int> > v(in.begin(), in.end());
    test_allocation_exception_for_strong_guarantee(v, in, v.max_size() + 1);
  }
  check_new_delete_called();

  {
    std::vector<int> in(10, 42);
    std::vector<int, limited_allocator<int, 100> > v(in.begin(), in.end());
    test_allocation_exception_for_strong_guarantee(v, in, v.max_size() + 1);
  }
  check_new_delete_called();

#if TEST_STD_VER >= 23
  {
    std::vector<int> in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int, increasing_allocator<int>> v(in.begin(), in.end());
    test_allocation_exception_for_strong_guarantee(v, in, v.max_size() + 1);
  }
  check_new_delete_called();
#endif

  //
  // Tests for std::bad_alloc during reallocation failures
  //
  {
    std::vector<int> in(10, 42);
    std::vector<int, limited_allocator<int, 100> > v(in.begin(), in.end());
    test_allocation_exception_for_strong_guarantee(v, in, 91);
  }
  check_new_delete_called();

  {
    std::vector<int> in(10, 42);
    std::vector<int, limited_allocator<int, 100> > v(in.begin(), in.end());
    v.reserve(30);
    test_allocation_exception_for_strong_guarantee(v, in, 61);
  }
  check_new_delete_called();

  { // Practical example: Testing with 100 integers.
    auto in = getIntegerInputs(100);
    std::vector<int, limited_allocator<int, 299> > v(in.begin(), in.end());
    test_allocation_exception_for_strong_guarantee(v, in, 200);
  }
  check_new_delete_called();

  { // Practical example: Testing with 100 strings, each 256 characters long.
    std::vector<std::string> in = getStringInputsWithLength(100, 256);
    std::vector<std::string, limited_allocator<std::string, 299> > v(in.begin(), in.end());
    test_allocation_exception_for_strong_guarantee(v, in, 200);
  }
  check_new_delete_called();
}

// Check the strong exception guarantee during copy-constructor failures
void test_copy_ctor_exceptions() {
  {
    int a[] = {1, 2, 3, 4, 5};
    std::vector<int> in(a, a + sizeof(a) / sizeof(a[0]));
    std::vector<throwing_data<int> > v;
    test_copy_ctor_exception_for_strong_guarantee(v, in);
  }
  check_new_delete_called();

  {
    int a[] = {1, 2, 3, 4, 5};
    std::vector<int> in(a, a + sizeof(a) / sizeof(a[0]));
    std::vector<throwing_data<int>, min_allocator<throwing_data<int> > > v;
    test_copy_ctor_exception_for_strong_guarantee(v, in);
  }
  check_new_delete_called();

  {
    std::vector<int> in(10, 42);
    std::vector<throwing_data<int>, safe_allocator<throwing_data<int> > > v;
    test_copy_ctor_exception_for_strong_guarantee(v, in);
  }
  check_new_delete_called();

  {
    std::vector<int> in(10, 42);
    std::vector<throwing_data<int>, test_allocator<throwing_data<int> > > v;
    test_copy_ctor_exception_for_strong_guarantee(v, in);
  }
  check_new_delete_called();

  {
    std::vector<int> in(10, 42);
    std::vector<throwing_data<int>, limited_allocator<throwing_data<int>, 100> > v;
    test_copy_ctor_exception_for_strong_guarantee(v, in);
  }
  check_new_delete_called();

#if TEST_STD_VER >= 23
  {
    std::vector<int> in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<throwing_data<int>, increasing_allocator<throwing_data<int>>> v;
    test_copy_ctor_exception_for_strong_guarantee(v, in);
  }
  check_new_delete_called();
#endif

  { // Practical example: Testing with 100 integers.
    auto in = getIntegerInputs(100);
    std::vector<throwing_data<int> > v;
    test_copy_ctor_exception_for_strong_guarantee(v, in);
  }
  check_new_delete_called();

  { // Practical example: Testing with 100 strings, each 256 characters long.
    std::vector<std::string> in = getStringInputsWithLength(100, 256);
    std::vector<throwing_data<std::string> > v;
    test_copy_ctor_exception_for_strong_guarantee(v, in);
  }
  check_new_delete_called();
}

int main(int, char**) {
  test_allocation_exceptions();
  test_copy_ctor_exceptions();
}
