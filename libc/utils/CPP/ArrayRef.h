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

// The implementations of ArrayRef and MutualArrayRef in this file are based
// on the implementations of the types with the same names in
// llvm/ADT/ArrayRef.h. The implementations in this file are of a limited
// functionality, but can be extended in an as needed basis.

template <typename T> class ArrayRef {
public:
  using iterator = const T *;

private:
  const T *Data = nullptr;
  size_t Length = 0;

public:
  ArrayRef() = default;

  // From Array.
  template <size_t N>
  ArrayRef(const Array<T, N> &Arr) : Data(Arr.Data), Length(N) {}

  // Construct an ArrayRef from a single element.
  explicit ArrayRef(const T &OneElt) : Data(&OneElt), Length(1) {}

  // Construct an ArrayRef from a pointer and length.
  ArrayRef(const T *data, size_t length) : Data(data), Length(length) {}

  // Construct an ArrayRef from a range.
  ArrayRef(const T *begin, const T *end) : Data(begin), Length(end - begin) {}

  // Construct an ArrayRef from a C array.
  template <size_t N>
  constexpr ArrayRef(const T (&Arr)[N]) : Data(Arr), Length(N) {}

  iterator begin() const { return Data; }
  iterator end() const { return Data + Length; }

  bool empty() const { return Length == 0; }

  const T *data() const { return Data; }

  size_t size() const { return Length; }

  const T &operator[](size_t Index) const { return Data[Index]; }
};

template <typename T> class MutableArrayRef : public ArrayRef<T> {
public:
  using iterator = T *;

  // From Array.
  template <size_t N> MutableArrayRef(Array<T, N> &Arr) : ArrayRef<T>(Arr) {}

  // Construct from a single element.
  explicit MutableArrayRef(T &OneElt) : ArrayRef<T>(OneElt) {}

  // Construct from a pointer and length.
  MutableArrayRef(T *data, size_t length) : ArrayRef<T>(data, length) {}

  // Construct from a range.
  MutableArrayRef(T *begin, T *end) : ArrayRef<T>(begin, end) {}

  // Construct from a C array.
  template <size_t N>
  constexpr MutableArrayRef(T (&Arr)[N]) : ArrayRef<T>(Arr) {}

  T *data() const { return const_cast<T *>(ArrayRef<T>::data()); }

  iterator begin() const { return data(); }
  iterator end() const { return data() + this->size(); }

  T &operator[](size_t Index) const { return data()[Index]; }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_CPP_ARRAYREF_H
