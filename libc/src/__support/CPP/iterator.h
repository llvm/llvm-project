//===-- Standalone implementation of iterator -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_ITERATOR_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_ITERATOR_H

#include "src/__support/CPP/type_traits/enable_if.h"
#include "src/__support/CPP/type_traits/is_convertible.h"
#include "src/__support/CPP/type_traits/is_same.h"
#include "src/__support/macros/attributes.h"

namespace LIBC_NAMESPACE {
namespace cpp {

template <typename T> struct iterator_traits;
template <typename T> struct iterator_traits<T *> {
  using reference = T &;
};

template <typename Iter> class reverse_iterator {
  Iter current;

public:
  using reference = typename iterator_traits<Iter>::reference;

  reverse_iterator() : current() {}
  constexpr explicit reverse_iterator(Iter it) : current(it) {}

  template <typename Other,
            cpp::enable_if_t<!cpp::is_same_v<Iter, Other> &&
                                 cpp::is_convertible_v<const Other &, Iter>,
                             int> = 0>
  constexpr explicit reverse_iterator(const Other &it) : current(it) {}

  constexpr reference operator*() const {
    Iter tmp = current;
    return *--tmp;
  }
  constexpr reverse_iterator operator--() {
    ++current;
    return *this;
  }
  constexpr reverse_iterator &operator++() {
    --current;
    return *this;
  }
  constexpr reverse_iterator operator++(int) {
    reverse_iterator tmp(*this);
    --current;
    return tmp;
  }
};

} // namespace cpp
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_ITERATOR_H
