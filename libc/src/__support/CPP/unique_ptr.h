//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Standalone implementation of std::unique_ptr.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_UNIQUE_PTR_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_UNIQUE_PTR_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stddef.h> // For size_t

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

/// A wrapper for deleting objects by default.
template <typename T> struct default_delete {
  LIBC_INLINE constexpr default_delete() = default;

  template <typename U, typename = enable_if_t<is_convertible_v<U *, T *>>>
  LIBC_INLINE constexpr default_delete(const default_delete<U> &) {}

  LIBC_INLINE constexpr void operator()(T *ptr) const {
    static_assert(sizeof(T) > 0, "cannot delete an incomplete type");
    delete ptr;
  }
};

/// A wrapper for deleting array objects by default.
template <typename T> struct default_delete<T[]> {
  LIBC_INLINE constexpr default_delete() = default;

  template <typename U,
            typename = enable_if_t<is_convertible_v<U (*)[], T (*)[]>>>
  LIBC_INLINE constexpr default_delete(const default_delete<U[]> &) {}

  template <typename U,
            typename = enable_if_t<is_convertible_v<U (*)[], T (*)[]>>>
  LIBC_INLINE constexpr void operator()(U *ptr) const {
    static_assert(sizeof(U) > 0, "cannot delete an incomplete type");
    delete[] ptr;
  }
};

/// A smart pointer that owns and manages another object through a pointer.
template <typename T, typename Deleter = default_delete<T>>
class LIBC_TRIVIAL_ABI unique_ptr {
  T *ptr = nullptr;
  LIBC_NO_UNIQUE_ADDRESS Deleter deleter{};

public:
  using element_type = T;
  using deleter_type = Deleter;
  using pointer = T *;

  LIBC_INLINE constexpr unique_ptr() = default;
  LIBC_INLINE constexpr unique_ptr(decltype(nullptr))
      : ptr(nullptr), deleter() {}

  template <typename D = Deleter, enable_if_t<!is_pointer_v<D>, int> = 0>
  LIBC_INLINE constexpr explicit unique_ptr(pointer p) : ptr(p), deleter() {}

  LIBC_INLINE constexpr unique_ptr(pointer p, const deleter_type &d)
      : ptr(p), deleter(d) {}
  LIBC_INLINE constexpr unique_ptr(pointer p, deleter_type &&d)
      : ptr(p), deleter(cpp::move(d)) {}

  // Move constructor
  LIBC_INLINE constexpr unique_ptr(unique_ptr &&other)
      : ptr(other.release()),
        deleter(cpp::forward<Deleter>(other.get_deleter())) {}

  // Move assignment
  LIBC_INLINE constexpr unique_ptr &operator=(unique_ptr &&other) {
    if (this == &other)
      return *this;
    reset(other.release());
    deleter = cpp::forward<Deleter>(other.get_deleter());
    return *this;
  }

  // Nullptr assignment
  LIBC_INLINE constexpr unique_ptr &operator=(decltype(nullptr)) {
    reset();
    return *this;
  }

  // Conversion move constructor
  template <typename U, typename E,
            typename = enable_if_t<
                is_convertible_v<typename unique_ptr<U, E>::pointer, pointer> &&
                !is_array_v<U> && is_convertible_v<E, Deleter>>>
  LIBC_INLINE constexpr unique_ptr(unique_ptr<U, E> &&other)
      : ptr(other.release()), deleter(cpp::forward<E>(other.get_deleter())) {}

  // Conversion move assignment
  template <typename U, typename E,
            typename = enable_if_t<
                is_convertible_v<typename unique_ptr<U, E>::pointer, pointer> &&
                !is_array_v<U> && is_assignable_v<Deleter &, E &&>>>
  LIBC_INLINE constexpr unique_ptr &operator=(unique_ptr<U, E> &&other) {
    reset(other.release());
    deleter = cpp::forward<E>(other.get_deleter());
    return *this;
  }

  // Disable copy
  unique_ptr(const unique_ptr &) = delete;
  unique_ptr &operator=(const unique_ptr &) = delete;

  LIBC_INLINE ~unique_ptr() { reset(); }

  LIBC_INLINE constexpr pointer get() const { return ptr; }
  LIBC_INLINE constexpr deleter_type &get_deleter() { return deleter; }
  LIBC_INLINE constexpr const deleter_type &get_deleter() const {
    return deleter;
  }

  LIBC_INLINE constexpr explicit operator bool() const {
    return ptr != nullptr;
  }

  [[nodiscard]] LIBC_INLINE constexpr pointer release() {
    pointer temp = ptr;
    ptr = nullptr;
    return temp;
  }

  LIBC_INLINE constexpr void reset(pointer p = pointer()) {
    pointer old_ptr = ptr;
    ptr = p;
    if (old_ptr)
      deleter(old_ptr);
  }

  LIBC_INLINE constexpr void swap(unique_ptr &other) {
    cpp::swap(ptr, other.ptr);
    cpp::swap(deleter, other.deleter);
  }

  LIBC_INLINE constexpr add_lvalue_reference_t<T> operator*() const {
    return *ptr;
  }
  LIBC_INLINE constexpr pointer operator->() const { return ptr; }
};

/// A smart pointer that owns and manages an array of objects through a pointer.
template <typename T, typename Deleter>
class LIBC_TRIVIAL_ABI unique_ptr<T[], Deleter> {
  T *ptr = nullptr;
  LIBC_NO_UNIQUE_ADDRESS Deleter deleter{};

public:
  using element_type = T;
  using deleter_type = Deleter;
  using pointer = T *;

  LIBC_INLINE constexpr unique_ptr() = default;
  LIBC_INLINE constexpr unique_ptr(decltype(nullptr))
      : ptr(nullptr), deleter() {}

  template <typename D = Deleter, enable_if_t<!is_pointer_v<D>, int> = 0>
  LIBC_INLINE constexpr explicit unique_ptr(pointer p) : ptr(p), deleter() {}

  LIBC_INLINE constexpr unique_ptr(pointer p, const deleter_type &d)
      : ptr(p), deleter(d) {}
  LIBC_INLINE constexpr unique_ptr(pointer p, deleter_type &&d)
      : ptr(p), deleter(cpp::move(d)) {}

  // Move constructor
  LIBC_INLINE constexpr unique_ptr(unique_ptr &&other)
      : ptr(other.release()),
        deleter(cpp::forward<Deleter>(other.get_deleter())) {}

  // Move assignment
  LIBC_INLINE constexpr unique_ptr &operator=(unique_ptr &&other) {
    if (this == &other)
      return *this;
    reset(other.release());
    deleter = cpp::forward<Deleter>(other.get_deleter());
    return *this;
  }

  // Nullptr assignment
  LIBC_INLINE constexpr unique_ptr &operator=(decltype(nullptr)) {
    reset();
    return *this;
  }

  // Disable copy
  unique_ptr(const unique_ptr &) = delete;
  unique_ptr &operator=(const unique_ptr &) = delete;

  LIBC_INLINE ~unique_ptr() { reset(); }

  LIBC_INLINE constexpr pointer get() const { return ptr; }
  LIBC_INLINE constexpr deleter_type &get_deleter() { return deleter; }
  LIBC_INLINE constexpr const deleter_type &get_deleter() const {
    return deleter;
  }

  LIBC_INLINE constexpr explicit operator bool() const {
    return ptr != nullptr;
  }

  [[nodiscard]] LIBC_INLINE constexpr pointer release() {
    pointer temp = ptr;
    ptr = nullptr;
    return temp;
  }

  LIBC_INLINE constexpr void reset(pointer p = pointer()) {
    pointer old_ptr = ptr;
    ptr = p;
    if (old_ptr)
      deleter(old_ptr);
  }

  LIBC_INLINE constexpr void swap(unique_ptr &other) {
    cpp::swap(ptr, other.ptr);
    cpp::swap(deleter, other.deleter);
  }

  LIBC_INLINE constexpr T &operator[](size_t i) const { return ptr[i]; }
};

template <typename T, typename D>
LIBC_INLINE constexpr void swap(unique_ptr<T, D> &x, unique_ptr<T, D> &y) {
  x.swap(y);
}

template <typename T1, typename D1, typename T2, typename D2>
LIBC_INLINE constexpr bool operator==(const unique_ptr<T1, D1> &x,
                                      const unique_ptr<T2, D2> &y) {
  return x.get() == y.get();
}

template <typename T1, typename D1, typename T2, typename D2>
LIBC_INLINE constexpr bool operator!=(const unique_ptr<T1, D1> &x,
                                      const unique_ptr<T2, D2> &y) {
  return x.get() != y.get();
}

template <typename T, typename D>
LIBC_INLINE constexpr bool operator==(const unique_ptr<T, D> &x,
                                      decltype(nullptr)) {
  return !x;
}

template <typename T, typename D>
LIBC_INLINE constexpr bool operator==(decltype(nullptr),
                                      const unique_ptr<T, D> &x) {
  return !x;
}

template <typename T, typename D>
LIBC_INLINE constexpr bool operator!=(const unique_ptr<T, D> &x,
                                      decltype(nullptr)) {
  return static_cast<bool>(x);
}

template <typename T, typename D>
LIBC_INLINE constexpr bool operator!=(decltype(nullptr),
                                      const unique_ptr<T, D> &x) {
  return static_cast<bool>(x);
}

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_UNIQUE_PTR_H
