//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::sort

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

struct BadIter {
  struct Value {
    friend bool operator==(const Value& x, const Value& y);
    friend bool operator!=(const Value& x, const Value& y);
    friend bool operator< (const Value& x, const Value& y);
    friend bool operator<=(const Value& x, const Value& y);
    friend bool operator> (const Value& x, const Value& y);
    friend bool operator>=(const Value& x, const Value& y);
    friend void swap(Value, Value);
  };

  using iterator_category = std::random_access_iterator_tag;
  using value_type = Value;
  using reference = Value&;
  using difference_type = long;
  using pointer = Value*;

  Value operator*() const; // Not `Value&`.
  reference operator[](difference_type n) const;

  BadIter& operator++();
  BadIter& operator--();
  BadIter operator++(int);
  BadIter operator--(int);

  BadIter& operator+=(difference_type n);
  BadIter& operator-=(difference_type n);
  friend BadIter operator+(BadIter x, difference_type n);
  friend BadIter operator+(difference_type n, BadIter x);
  friend BadIter operator-(BadIter x, difference_type n);
  friend difference_type operator-(BadIter x, BadIter y);

  friend bool operator==(const BadIter& x, const BadIter& y);
  friend bool operator!=(const BadIter& x, const BadIter& y);
  friend bool operator< (const BadIter& x, const BadIter& y);
  friend bool operator<=(const BadIter& x, const BadIter& y);
  friend bool operator> (const BadIter& x, const BadIter& y);
  friend bool operator>=(const BadIter& x, const BadIter& y);
};

// Verify that iterators with incorrect `iterator_traits` are rejected. This protects against potential undefined
// behavior when these iterators are passed to standard algorithms.
void test() {
  std::sort(BadIter(), BadIter());
  //expected-error-re@*:* {{static assertion failed {{.*}}It looks like your iterator's `iterator_traits<It>::reference` does not match the return type of dereferencing the iterator}}
}
