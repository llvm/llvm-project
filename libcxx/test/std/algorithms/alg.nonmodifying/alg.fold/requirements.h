//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_FOLD_REQUIREMENTS_H
#define LIBCXX_TEST_FOLD_REQUIREMENTS_H

#include <cstddef>

// FIXME(cjdb): deduplicate
struct bad_iterator_category {
  using value_type        = int;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = void;

  value_type operator*() const;

  bad_iterator_category& operator++();
  void operator++(int);
};

struct non_movable {
  non_movable()              = default;
  non_movable(non_movable&&) = delete;
};

struct copyable_non_movable {
  copyable_non_movable(int);
  copyable_non_movable(copyable_non_movable&&) = delete;
  copyable_non_movable(copyable_non_movable const&);

  friend int operator+(copyable_non_movable const&, non_movable const&);
  friend int operator+(non_movable const&, copyable_non_movable const&);

  friend copyable_non_movable const& operator-(int, copyable_non_movable const&);
  friend copyable_non_movable const& operator-(copyable_non_movable const&, int);
  friend copyable_non_movable const& operator-(copyable_non_movable const&, copyable_non_movable const&);
};

struct non_copy_constructible_callable {
  non_copy_constructible_callable()                                       = default;
  non_copy_constructible_callable(non_copy_constructible_callable&&)      = default;
  non_copy_constructible_callable(non_copy_constructible_callable const&) = delete;

  int operator()(int, int) const;
};

struct not_invocable {
  int operator()(int, int&&);
};

struct S {};

struct non_decayable_result {
  S volatile& operator()(S, S) const;
};

struct not_convertible_to_int {
  friend int operator+(not_convertible_to_int, not_convertible_to_int);
  friend int operator+(not_convertible_to_int, int);
  friend int operator+(int, not_convertible_to_int);
};

struct not_invocable_with_decayed {
  not_invocable_with_decayed(int);
  friend not_invocable_with_decayed& operator+(int, not_invocable_with_decayed&);
  friend not_invocable_with_decayed& operator+(not_invocable_with_decayed&, int);
  friend not_invocable_with_decayed& operator+(not_invocable_with_decayed volatile&, not_invocable_with_decayed&);
};

struct not_assignable_to_decayed {
  not_assignable_to_decayed();
  not_assignable_to_decayed(not_assignable_to_decayed&);
  not_assignable_to_decayed(not_assignable_to_decayed const&);
  not_assignable_to_decayed(not_assignable_to_decayed volatile&);
  not_assignable_to_decayed(not_assignable_to_decayed const volatile&);
  friend not_assignable_to_decayed volatile& operator+(not_assignable_to_decayed, not_assignable_to_decayed);
};

#endif // LIBCXX_TEST_FOLD_REQUIREMENTS
