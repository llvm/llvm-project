//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

#include <stacktrace>
#include <iostream>
#include <iterator>
#include <memory>

#include <cassert>
#include <stdexcept>

/*
  (19.6.4.3) Observers [stacktrace.basic.obs]

  // [stacktrace.basic.obs], observers
  allocator_type get_allocator() const noexcept;      [1]

  const_iterator begin() const noexcept;              [2]
  const_iterator end() const noexcept;                [3]
  const_reverse_iterator rbegin() const noexcept;     [4]
  const_reverse_iterator rend() const noexcept;       [5]

  const_iterator cbegin() const noexcept;             [6]
  const_iterator cend() const noexcept;               [7]
  const_reverse_iterator crbegin() const noexcept;    [8]
  const_reverse_iterator crend() const noexcept;      [9]

  bool empty() const noexcept;                        [10]
  size_type size() const noexcept;                    [11]
  size_type max_size() const noexcept;                [12]

  const_reference operator[](size_type) const;        [13]
  const_reference at(size_type) const;                [14]
*/

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test1() { return std::stacktrace::current(0, 4); }
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test2() { return test1(); }
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test3() { return test2(); }

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  std::stacktrace st;

  /*
    using const_iterator = implementation-defined;
    The type models random_access_iterator ([iterator.concept.random.access]) and meets the
    Cpp17RandomAccessIterator requirements ([random.access.iterators]).
  */
  static_assert(std::random_access_iterator<decltype(st.begin())>);
  static_assert(std::random_access_iterator<decltype(st.rbegin())>);
  static_assert(std::random_access_iterator<decltype(st.cbegin())>);
  static_assert(std::random_access_iterator<decltype(st.crbegin())>);

  /*
    allocator_type get_allocator() const noexcept;
    Returns: frames_.get_allocator().
  */
  static_assert(std::same_as<decltype(st.get_allocator()), std::stacktrace::allocator_type>);

  /*
    const_iterator begin() const noexcept;
    const_iterator cbegin() const noexcept;
    Returns: An iterator referring to the first element in frames_. If empty() is true,
    then it returns the same value as end().

    const_iterator end() const noexcept;
    const_iterator cend() const noexcept;
    Returns: The end iterator.

    const_reverse_iterator rbegin() const noexcept;
    const_reverse_iterator crbegin() const noexcept;
    Returns: reverse_iterator(cend()).

    const_reverse_iterator rend() const noexcept;
    const_reverse_iterator crend() const noexcept;
    Returns: reverse_iterator(cbegin()).
  */

  // st is initially empty:
  assert(st.begin() == st.end());
  assert(st.cbegin() == st.cend());
  assert(st.rbegin() == st.rend());
  assert(st.crbegin() == st.crend());

  /*
    bool empty() const noexcept;
    Returns: frames_.empty().
  */
  assert(st.empty());
  assert(st.size() == 0);

  // no longer empty:
  st = test3();
  assert(!st.empty());
  assert(st.begin() != st.end());
  assert(st.cbegin() != st.cend());

  /*
    size_type size() const noexcept;
    Returns: frames_.size().

    size_type max_size() const noexcept;
    Returns: frames_.max_size().
  */
  std::cout << st << std::endl;
  assert(st.size() >= 4);
  assert(st.max_size() == (std::vector<std::stacktrace_entry, std::allocator<std::stacktrace_entry>>().max_size()));

  /*
    const_reference operator[](size_type frame_no) const;
    Preconditions: frame_no < size() is true.
    Returns: frames_[frame_no].
    Throws: Nothing.
  */
  assert(st[0]);
  assert(st[1]);
  assert(st[2]);
  assert(st[3]);

  /*
    const_reference at(size_type frame_no) const;
    Returns: frames_[frame_no].
    Throws: out_of_range if frame_no >= size().
  */
  assert(st.at(0));
  assert(st.at(1));
  assert(st.at(2));
  assert(st.at(3));
  try {
    (void)st.at(42);
    assert(false && "'at' should have thrown range_error");
  } catch (std::out_of_range const&) {
    // ok
  }

  auto f0 = st[0];
  auto f1 = st[1];
  auto f2 = st[2];
  auto f3 = st[3];

  auto fit = st.begin();
  assert(*fit++ == f0);
  assert(fit != st.end());

  auto rit = st.rbegin();
  assert(rit != st.rend());

  return 0;
}
