//===-- A data structure for a fixed capacity data store --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FIXEDVECTOR_H
#define LLVM_LIBC_SRC___SUPPORT_FIXEDVECTOR_H

#include "src/__support/CPP/array.h"

#include "src/__support/CPP/iterator.h"

namespace LIBC_NAMESPACE {

// A fixed size data store backed by an underlying cpp::array data structure. It
// supports vector like API but is not resizable like a vector.
template <typename T, size_t CAPACITY> class FixedVector {
  cpp::array<T, CAPACITY> store;
  size_t item_count = 0;

public:
  constexpr FixedVector() = default;

  using iterator = typename cpp::array<T, CAPACITY>::iterator;
  constexpr FixedVector(iterator begin, iterator end) : store{}, item_count{} {
    for (; begin != end; ++begin)
      push_back(*begin);
  }

  using const_iterator = typename cpp::array<T, CAPACITY>::const_iterator;
  constexpr FixedVector(const_iterator begin, const_iterator end)
      : store{}, item_count{} {
    for (; begin != end; ++begin)
      push_back(*begin);
  }

  constexpr FixedVector(size_t count, const T &value) : store{}, item_count{} {
    for (size_t i = 0; i < count; ++i)
      push_back(value);
  }

  constexpr bool push_back(const T &obj) {
    if (item_count == CAPACITY)
      return false;
    store[item_count] = obj;
    ++item_count;
    return true;
  }

  constexpr const T &back() const { return store[item_count - 1]; }

  constexpr T &back() { return store[item_count - 1]; }

  constexpr bool pop_back() {
    if (item_count == 0)
      return false;
    --item_count;
    return true;
  }

  constexpr T &operator[](size_t idx) { return store[idx]; }

  constexpr const T &operator[](size_t idx) const { return store[idx]; }

  constexpr bool empty() const { return item_count == 0; }

  constexpr size_t size() const { return item_count; }

  // Empties the store for all practical purposes.
  constexpr void reset() { item_count = 0; }

  // This static method does not free up the resources held by |store|,
  // say by calling `free` or something similar. It just does the equivalent
  // of the `reset` method. Considering that FixedVector is of fixed storage,
  // a `destroy` method like this should not be required. However, FixedVector
  // is used in a few places as an alternate for data structures which use
  // dynamically allocated storate. So, the `destroy` method like this
  // matches the `destroy` API of those other data structures so that users
  // can easily swap one data structure for the other.
  static void destroy(FixedVector<T, CAPACITY> *store) { store->reset(); }

  using reverse_iterator = typename cpp::array<T, CAPACITY>::reverse_iterator;
  LIBC_INLINE constexpr reverse_iterator rbegin() {
    return reverse_iterator{&store[item_count]};
  }
  LIBC_INLINE constexpr reverse_iterator rend() { return store.rend(); }

  LIBC_INLINE constexpr iterator begin() { return store.begin(); }
  LIBC_INLINE constexpr iterator end() { return iterator{&store[item_count]}; }

  LIBC_INLINE constexpr const_iterator begin() const { return store.begin(); }
  LIBC_INLINE constexpr const_iterator end() const {
    return const_iterator{&store[item_count]};
  }
};

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDVECTOR_H
