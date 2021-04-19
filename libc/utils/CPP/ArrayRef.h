//===-- Self contained ArrayRef type ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_CPP_ARRAYREF_H
#define LLVM_LIBC_UTILS_CPP_ARRAYREF_H

#include "Array.h"

#include <stddef.h> // For size_t.

namespace __llvm_libc {
namespace cpp {

// The implementations of ArrayRef and MutableArrayRef in this file are based
// on the implementations of the types with the same names in
// llvm/ADT/ArrayRef.h. The implementations in this file are of a limited
// functionality, but can be extended in an as needed basis.
namespace internal {
template <typename T> class ArrayRefBase {
public:
  using iterator = T *;
  using pointer = T *;
  using reference = T &;

  ArrayRefBase() = default;

  // From Array.
  template <size_t N>
  ArrayRefBase(Array<T, N> &Arr) : Data(Arr.Data), Length(N) {}

  // Construct an ArrayRefBase from a single element.
  explicit ArrayRefBase(T &OneElt) : Data(&OneElt), Length(1) {}

  // Construct an ArrayRefBase from a pointer and length.
  ArrayRefBase(pointer Data, size_t Length) : Data(Data), Length(Length) {}

  // Construct an ArrayRefBase from a range.
  ArrayRefBase(iterator Begin, iterator End)
      : Data(Begin), Length(End - Begin) {}

  // Construct an ArrayRefBase from a C array.
  template <size_t N>
  constexpr ArrayRefBase(T (&Arr)[N]) : Data(Arr), Length(N) {}

  iterator begin() const { return Data; }
  iterator end() const { return Data + Length; }

  bool empty() const { return Length == 0; }

  pointer data() const { return Data; }

  size_t size() const { return Length; }

  reference operator[](size_t Index) const { return Data[Index]; }

  // slice(n, m) - Chop off the first N elements of the array, and keep M
  // elements in the array.
  ArrayRefBase<T> slice(size_t N, size_t M) const {
    return ArrayRefBase<T>(data() + N, M);
  }
  // slice(n) - Chop off the first N elements of the array.
  ArrayRefBase<T> slice(size_t N) const { return slice(N, size() - N); }

  // Drop the first \p N elements of the array.
  ArrayRefBase<T> drop_front(size_t N = 1) const {
    return slice(N, size() - N);
  }

  // Drop the last \p N elements of the array.
  ArrayRefBase<T> drop_back(size_t N = 1) const { return slice(0, size() - N); }

  // Return a copy of *this with only the first \p N elements.
  ArrayRefBase<T> take_front(size_t N = 1) const {
    if (N >= size())
      return *this;
    return drop_back(size() - N);
  }

  // Return a copy of *this with only the last \p N elements.
  ArrayRefBase<T> take_back(size_t N = 1) const {
    if (N >= size())
      return *this;
    return drop_front(size() - N);
  }

private:
  pointer Data = nullptr;
  size_t Length = 0;
};
} // namespace internal

template <typename T> using ArrayRef = internal::ArrayRefBase<const T>;
template <typename T> using MutableArrayRef = internal::ArrayRefBase<T>;

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_CPP_ARRAYREF_H
