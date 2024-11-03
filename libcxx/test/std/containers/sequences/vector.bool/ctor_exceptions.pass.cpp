//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// (bug report: https://llvm.org/PR58392)
// Check that vector<bool> constructors don't leak memory when an operation inside the constructor throws an exception

#include <cstddef>
#include <type_traits>
#include <vector>

#include "count_new.h"
#include "test_iterators.h"

template <class T>
struct Allocator {
  using value_type      = T;
  using is_always_equal = std::false_type;

  template <class U>
  Allocator(const Allocator<U>&) {}

  Allocator(bool should_throw = true) {
    if (should_throw)
      throw 0;
  }

  T* allocate(std::size_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* ptr, std::size_t n) { std::allocator<T>().deallocate(ptr, n); }

  template <class U>
  friend bool operator==(const Allocator&, const Allocator<U>&) { return true; }
};

template <class IterCat>
struct Iterator {
  using iterator_category = IterCat;
  using difference_type   = std::ptrdiff_t;
  using value_type        = bool;
  using reference         = bool&;
  using pointer           = bool*;

  int i_;
  bool b_ = true;
  Iterator(int i = 0) : i_(i) {}
  bool& operator*() {
    if (i_ == 1)
      throw 1;
    return b_;
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
  using AllocVec = std::vector<bool, Allocator<bool> >;

#if TEST_STD_VER >= 14
  try { // Throw in vector(size_type, const allocator_type&) from allocator
    Allocator<bool> alloc(false);
    AllocVec get_alloc(0, alloc);
  } catch (int) {
  }
  check_new_delete_called();
#endif  // TEST_STD_VER >= 14

  try { // Throw in vector(InputIterator, InputIterator) from input iterator
    std::vector<bool> vec((Iterator<std::input_iterator_tag>()), Iterator<std::input_iterator_tag>(2));
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator) from forward iterator
    std::vector<bool> vec((Iterator<std::forward_iterator_tag>()), Iterator<std::forward_iterator_tag>(2));
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator) from allocator
    int a[] = {1, 2};
    AllocVec vec(cpp17_input_iterator<int*>(a), cpp17_input_iterator<int*>(a + 2));
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from input iterator
    std::allocator<bool> alloc;
    std::vector<bool> vec(Iterator<std::input_iterator_tag>(), Iterator<std::input_iterator_tag>(2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from forward iterator
    std::allocator<bool> alloc;
    std::vector<bool> vec(Iterator<std::forward_iterator_tag>(), Iterator<std::forward_iterator_tag>(2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from allocator
    bool a[] = {true, false};
    Allocator<bool> alloc(false);
    AllocVec vec(cpp17_input_iterator<bool*>(a), cpp17_input_iterator<bool*>(a + 2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from allocator
    bool a[] = {true, false};
    Allocator<bool> alloc(false);
    AllocVec vec(forward_iterator<bool*>(a), forward_iterator<bool*>(a + 2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  return 0;
}
