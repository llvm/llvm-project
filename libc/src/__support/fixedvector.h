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

namespace LIBC_NAMESPACE {

// A fixed size data store backed by an underlying cpp::array data structure. It
// supports vector like API but is not resizable like a vector.
template <typename T, size_t CAPACITY> class FixedVector {
  cpp::array<T, CAPACITY> store;
  size_t item_count = 0;

public:
  constexpr FixedVector() = default;

  bool push_back(const T &obj) {
    if (item_count == CAPACITY)
      return false;
    store[item_count] = obj;
    ++item_count;
    return true;
  }

  const T &back() const { return store[item_count - 1]; }

  T &back() { return store[item_count - 1]; }

  bool pop_back() {
    if (item_count == 0)
      return false;
    --item_count;
    return true;
  }

  bool empty() const { return item_count == 0; }

  // Empties the store for all practical purposes.
  void reset() { item_count = 0; }

  // This static method does not free up the resources held by |store|,
  // say by calling `free` or something similar. It just does the equivalent
  // of the `reset` method. Considering that FixedVector is of fixed storage,
  // a `destroy` method like this should not be required. However, FixedVector
  // is used in a few places as an alternate for data structures which use
  // dynamically allocated storate. So, the `destroy` method like this
  // matches the `destroy` API of those other data structures so that users
  // can easily swap one data structure for the other.
  static void destroy(FixedVector<T, CAPACITY> *store) { store->reset(); }
};

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDVECTOR_H
