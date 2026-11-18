//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A growable array, which reports allocation failures to the caller.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_VECTOR_H
#define LLVM_LIBC_SRC___SUPPORT_VECTOR_H

#include "hdr/func/free.h"
#include "hdr/func/realloc.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/checked.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stddef.h> // max_align_t, ptrdiff_t

namespace LIBC_NAMESPACE_DECL {

/// A heap-allocated array which grows as needed, similar to std::vector.
///
/// Operations which may need to allocate memory return a Checked object, which
/// the caller must check. If such an operation fails, the vector is left
/// unchanged. The other operations cannot fail, but they have the same
/// preconditions as in std::vector (e.g., pop_back() requires a non-empty
/// vector). Where practical, these are checked with LIBC_ASSERT.
///
/// Copying a vector requires an allocation, so Vector is not copy-constructible
/// or copy-assignable.
///
/// T must be trivially copyable, so that the elements can be moved with
/// realloc. Over-aligned types (whose alignment exceeds alignof(max_align_t))
/// are not supported either. Like with std::vector, T may be incomplete at the
/// point where Vector<T> is instantiated, as long as it is completed before the
/// vector is used.
template <typename T> class Vector {
  T *items = nullptr;
  size_t count = 0;
  size_t cap = 0;

  // Limiting the storage size to PTRDIFF_MAX bytes guarantees that differences
  // between pointers into it are representable. It also means that doubling
  // the capacity cannot overflow.
  LIBC_INLINE static constexpr size_t max_size() {
    return static_cast<size_t>(cpp::numeric_limits<ptrdiff_t>::max()) /
           sizeof(T);
  }

  // Moves the elements to new storage with room for `new_cap` elements.
  LIBC_INLINE bool reallocate(size_t new_cap) {
    // These are checked here rather than at class scope, so that T can be
    // incomplete when Vector<T> is instantiated.
    static_assert(cpp::is_trivially_copyable_v<T>,
                  "non-trivially-copyable types are not supported");
    static_assert(alignof(T) <= alignof(max_align_t),
                  "over-aligned types are not supported");
    // Together with trivial copyability, this guarantees that T is an
    // implicit-lifetime type, whose objects realloc implicitly creates in the
    // new block.
    static_assert(cpp::is_move_constructible_v<T>,
                  "T must be move-constructible");
    LIBC_ASSERT(count <= new_cap && new_cap <= max_size());
    // realloc(p, 0) may free p and return nullptr.
    LIBC_ASSERT(new_cap > 0);
    // realloc can avoid the copy if it manages to resize the block in place.
    T *new_items = static_cast<T *>(::realloc(items, new_cap * sizeof(T)));
    if (new_items == nullptr)
      return false;
    items = new_items;
    cap = new_cap;
    return true;
  }

  // Makes room for at least one more element in a full vector.
  LIBC_INLINE bool grow() {
    LIBC_ASSERT(count == cap);
    if (cap == max_size())
      return false;
    // Grow geometrically, so that appends take amortized constant time.
    return reallocate(cpp::max(cap + 1, cpp::min(2 * cap, max_size())));
  }

  // Returns the index of the element `ptr` points to, or `count` if it does not
  // point to an element of this vector.
  LIBC_INLINE size_t index_of(const T *ptr) const {
    // Compare the addresses as integers, as comparing pointers to unrelated
    // objects gives unspecified results.
    uintptr_t offset =
        reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(items);
    return offset < count * sizeof(T) ? offset / sizeof(T) : count;
  }

public:
  using iterator = T *;
  using const_iterator = const T *;

  LIBC_INLINE constexpr Vector() = default;
  LIBC_INLINE Vector(Vector &&other) { swap(other); }
  Vector(const Vector &) = delete;

  LIBC_INLINE Vector &operator=(Vector &&other) {
    Vector tmp(cpp::move(other));
    swap(tmp);
    return *this;
  }
  Vector &operator=(const Vector &) = delete;

  LIBC_INLINE ~Vector() { ::free(items); }

  LIBC_INLINE void swap(Vector &other) {
    cpp::swap(items, other.items);
    cpp::swap(count, other.count);
    cpp::swap(cap, other.cap);
  }

  LIBC_INLINE bool empty() const { return count == 0; }
  LIBC_INLINE size_t size() const { return count; }
  LIBC_INLINE size_t capacity() const { return cap; }

  LIBC_INLINE T *data() { return items; }
  LIBC_INLINE const T *data() const { return items; }

  LIBC_INLINE T &operator[](size_t i) {
    LIBC_ASSERT(i < count);
    return items[i];
  }
  LIBC_INLINE const T &operator[](size_t i) const {
    LIBC_ASSERT(i < count);
    return items[i];
  }

  LIBC_INLINE T &front() { return (*this)[0]; }
  LIBC_INLINE const T &front() const { return (*this)[0]; }
  LIBC_INLINE T &back() { return (*this)[count - 1]; }
  LIBC_INLINE const T &back() const { return (*this)[count - 1]; }

  LIBC_INLINE iterator begin() { return items; }
  LIBC_INLINE const_iterator begin() const { return items; }
  LIBC_INLINE iterator end() { return items + count; }
  LIBC_INLINE const_iterator end() const { return items + count; }

  LIBC_INLINE void clear() { count = 0; }

  /// Appends a copy of `value`, which may be an element of this vector.
  LIBC_INLINE Checked push_back(const T &value) {
    const T *ptr = &value;
    if (count == cap) {
      // Growing the storage moves `value` if it is an element of this vector,
      // so in that case we find it again by its index.
      size_t index = index_of(ptr);
      if (!grow())
        return false;
      if (index < count)
        ptr = items + index;
    }
    new (items + count) T(*ptr);
    ++count;
    return true;
  }

  LIBC_INLINE void pop_back() {
    LIBC_ASSERT(count > 0);
    --count;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_VECTOR_H
