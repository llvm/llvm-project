//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utilities for enforcing that results of fallible operations get checked.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CHECKED_H
#define LLVM_LIBC_SRC___SUPPORT_CHECKED_H

#include "src/__support/libc_assert.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace detail {

class LIBC_TRIVIAL_ABI CheckedState {
#ifndef NDEBUG
  bool checked = false;

public:
  LIBC_INLINE constexpr CheckedState() = default;

  // Moving disarms `other` and starts `this` unchecked so the new owner must
  // explicitly check its value.
  LIBC_INLINE constexpr CheckedState(CheckedState &&other) {
    other.checked = true;
  }

  // Overwriting an existing CheckedState requires it to have been checked,
  // disarms `other`, and resets `this` to unchecked.
  LIBC_INLINE constexpr CheckedState &operator=(CheckedState &&other) {
    assert_checked();
    // Self-move-assignment leaves the (checked) state unchanged.
    if (this == &other)
      return *this;
    other.checked = true;
    checked = false;
    return *this;
  }

  LIBC_INLINE ~CheckedState() { assert_checked(); }

  LIBC_INLINE constexpr void mark_checked() { checked = true; }
  LIBC_INLINE constexpr void assert_checked() const { LIBC_ASSERT(checked); }
#else
public:
  LIBC_INLINE constexpr CheckedState() = default;
  LIBC_INLINE constexpr CheckedState(CheckedState &&) = default;
  LIBC_INLINE constexpr CheckedState &operator=(CheckedState &&) = default;

  LIBC_INLINE constexpr void mark_checked() {}
  LIBC_INLINE constexpr void assert_checked() const {}
#endif
};

} // namespace detail

/// The result of an operation that can fail (typically, because it needs to
/// allocate memory). In !NDEBUG builds, it asserts that the result has been
/// checked (by converting it to bool) before it is destroyed or overwritten by
/// a move assignment.
///
/// Moving from a Checked object disarms the source and leaves the destination
/// unchecked, forcing the new owner to check it.
///
/// In NDEBUG builds, nothing is tracked and Checked is just a wrapper for bool.
class [[nodiscard]] LIBC_TRIVIAL_ABI Checked {
  LIBC_NO_UNIQUE_ADDRESS detail::CheckedState state;
  bool success;

public:
  LIBC_INLINE constexpr Checked(bool success) : success(success) {}

  // Disallow (implicit) conversions from other types, such as integers or
  // pointers, as their truthiness need not correspond to success.
  template <typename U> Checked(U) = delete;

  LIBC_INLINE Checked(Checked &&other) = default;
  LIBC_INLINE Checked &operator=(Checked &&other) = default;

  Checked(const Checked &) = delete;
  Checked &operator=(const Checked &) = delete;

  /// Returns true if the operation succeeded, and marks the result as checked.
  ///
  /// This is intentionally non-const: recording the check in a const member
  /// function would require a mutable flag, and concurrent checks through const
  /// references would then race on it.
  LIBC_INLINE constexpr explicit operator bool() {
    state.mark_checked();
    return success;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CHECKED_H
