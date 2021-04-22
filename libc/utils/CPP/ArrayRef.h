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
#include "TypeTraits.h" //RemoveCVType

#include <stddef.h> // For size_t.

namespace __llvm_libc {
namespace cpp {

// The implementations of ArrayRef and MutableArrayRef in this file are based
// on the implementations of the types with the same names in
// llvm/ADT/ArrayRef.h. The implementations in this file are of a limited
// functionality, but can be extended in an as needed basis.
namespace internal {
template <typename QualifiedT> class ArrayRefBase {
public:
  using value_type = RemoveCVType<QualifiedT>;
  using pointer = value_type *;
  using const_pointer = const value_type *;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator = const_pointer;
  using const_iterator = const_pointer;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  ArrayRefBase() = default;

  // Construct an ArrayRefBase from a single element.
  explicit ArrayRefBase(QualifiedT &OneElt) : Data(&OneElt), Length(1) {}

  // Construct an ArrayRefBase from a pointer and length.
  ArrayRefBase(QualifiedT *Data, size_t Length) : Data(Data), Length(Length) {}

  // Construct an ArrayRefBase from a range.
  ArrayRefBase(QualifiedT *Begin, QualifiedT *End)
      : Data(Begin), Length(End - Begin) {}

  // Construct an ArrayRefBase from a C array.
  template <size_t N>
  constexpr ArrayRefBase(QualifiedT (&Arr)[N]) : Data(Arr), Length(N) {}

  QualifiedT *data() const { return Data; }
  size_t size() const { return Length; }

  auto begin() const { return data(); }
  auto end() const { return data() + size(); }

  bool empty() const { return size() == 0; }

  auto operator[](size_t Index) const { return data()[Index]; }

  // slice(n, m) - Chop off the first N elements of the array, and keep M
  // elements in the array.
  auto slice(size_t N, size_t M) const { return ArrayRefBase(data() + N, M); }
  // slice(n) - Chop off the first N elements of the array.
  auto slice(size_t N) const { return slice(N, size() - N); }

  // Drop the first \p N elements of the array.
  auto drop_front(size_t N = 1) const { return slice(N, size() - N); }

  // Drop the last \p N elements of the array.
  auto drop_back(size_t N = 1) const { return slice(0, size() - N); }

  // Return a copy of *this with only the first \p N elements.
  auto take_front(size_t N = 1) const {
    if (N >= size())
      return *this;
    return drop_back(size() - N);
  }

  // Return a copy of *this with only the last \p N elements.
  auto take_back(size_t N = 1) const {
    if (N >= size())
      return *this;
    return drop_front(size() - N);
  }

  // equals - Check for element-wise equality.
  bool equals(ArrayRefBase<QualifiedT> RHS) const {
    if (Length != RHS.Length)
      return false;
    auto First1 = begin();
    auto Last1 = end();
    auto First2 = RHS.begin();
    for (; First1 != Last1; ++First1, ++First2) {
      if (!(*First1 == *First2)) {
        return false;
      }
    }
    return true;
  }

private:
  QualifiedT *Data = nullptr;
  size_t Length = 0;
};
} // namespace internal

template <typename T> struct ArrayRef : public internal::ArrayRefBase<const T> {
private:
  static_assert(IsSameV<T, RemoveCVType<T>>,
                "ArrayRef must have a non-const, non-volatile value_type");
  using Impl = internal::ArrayRefBase<const T>;
  using Impl::Impl;

public:
  // From Array.
  template <size_t N> ArrayRef(const Array<T, N> &Arr) : Impl(Arr.Data, N) {}
};

template <typename T>
struct MutableArrayRef : public internal::ArrayRefBase<T> {
private:
  static_assert(
      IsSameV<T, RemoveCVType<T>>,
      "MutableArrayRef must have a non-const, non-volatile value_type");
  using Impl = internal::ArrayRefBase<T>;
  using Impl::Impl;

public:
  // From Array.
  template <size_t N> MutableArrayRef(Array<T, N> &Arr) : Impl(Arr.Data, N) {}
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_CPP_ARRAYREF_H
