//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: no-exceptions

// <inplace_vector>

// (bug report: https://llvm.org/PR58392)
// Check that vector constructors don't leak memory when an operation inside the constructor throws an exception

#include <cstddef>
#include <memory>
#include <type_traits>
#include <inplace_vector>

#include "count_new.h"
#include "test_iterators.h"

struct ThrowingT {
  int* throw_after_n_ = nullptr;
  ThrowingT() { throw 0; }

  ThrowingT(int& throw_after_n) : throw_after_n_(&throw_after_n) {
    if (throw_after_n == 0)
      throw 0;
    --throw_after_n;
  }

  ThrowingT(const ThrowingT&) {
    if (throw_after_n_ == nullptr || *throw_after_n_ == 0)
      throw 1;
    --*throw_after_n_;
  }

  ThrowingT& operator=(const ThrowingT&) {
    if (throw_after_n_ == nullptr || *throw_after_n_ == 0)
      throw 1;
    --*throw_after_n_;
    return *this;
  }
};

template <class IterCat>
struct Iterator {
  using iterator_category = IterCat;
  using difference_type   = std::ptrdiff_t;
  using value_type        = int;
  using reference         = int&;
  using pointer           = int*;

  int i_;
  Iterator(int i = 0) : i_(i) {}
  int& operator*() {
    if (i_ == 1)
      throw 1;
    return i_;
  }

  friend bool operator==(const Iterator& lhs, const Iterator& rhs) { return lhs.i_ == rhs.i_; }

  friend bool operator!=(const Iterator& lhs, const Iterator& rhs) { return lhs.i_ != rhs.i_; }

  Iterator& operator++() {
    ++i_;
    return *this;
  }

  Iterator operator++(int) {
    auto tmp = *this;
    ++i_;
    return tmp;
  }
};

void check_new_delete_called() {
  assert(globalMemCounter.new_called == globalMemCounter.delete_called);
  assert(globalMemCounter.new_array_called == globalMemCounter.delete_array_called);
  assert(globalMemCounter.aligned_new_called == globalMemCounter.aligned_delete_called);
  assert(globalMemCounter.aligned_new_array_called == globalMemCounter.aligned_delete_array_called);
}

int main(int, char**) {
  try { // Throw in inplace_vector(size_type) from type
    std::inplace_vector<ThrowingT, 1> v(1);
    assert(false);
  } catch (int) {
  } catch (...) {
    assert(false);
  }
  check_new_delete_called();

  try { // Throw in inplace_vector(size_type) from lack of capacity
    std::inplace_vector<ThrowingT, 1> v(2);
    assert(false);
  } catch (const std::bad_alloc&) {
  } catch (...) {
    assert(false);
  }
  check_new_delete_called();

  try { // Do not throw when none are constructed
    [[maybe_unused]] std::inplace_vector<ThrowingT, 1> v1(0);
    [[maybe_unused]] std::inplace_vector<ThrowingT, 0> v2(0);
  } catch (...) {
    assert(false);
  }
  check_new_delete_called();

  try { // Throw in inplace_vector(size_type, value_type) from type
    int throw_after = 1;
    ThrowingT v(throw_after);
    std::inplace_vector<ThrowingT, 1> vec(1, v);
    assert(false);
  } catch (int) {
  } catch (...) {
    assert(false);
  }
  check_new_delete_called();

  try { // Throw in inplace_vector(InputIterator, InputIterator) from input iterator
    std::inplace_vector<int, 4> vec((Iterator<std::input_iterator_tag>()), Iterator<std::input_iterator_tag>(2));
    assert(false);
  } catch (int) {
  } catch (...) {
    assert(false);
  }
  check_new_delete_called();

  try { // Throw in inplace_vector(InputIterator, InputIterator) from forward iterator
    std::inplace_vector<int, 4> vec((Iterator<std::forward_iterator_tag>()), Iterator<std::forward_iterator_tag>(2));
    assert(false);
  } catch (int) {
  } catch (...) {
    assert(false);
  }
  check_new_delete_called();

  try { // Throw in vector(const vector&) from type
    std::inplace_vector<ThrowingT, 10> vec;
    int throw_after = 0;
    vec.emplace_back(throw_after);
    auto vec2 = vec;
    assert(false);
  } catch (int) {
  } catch (...) {
    assert(false);
  }
  check_new_delete_called();

  try { // Throw in vector(initializer_list<value_type>) from type
    int throw_after = 1;
    std::inplace_vector<ThrowingT, 10> vec({ThrowingT(throw_after)});
    assert(false);
  } catch (int) {
  } catch (...) {
    assert(false);
  }
  check_new_delete_called();

  return 0;
}
