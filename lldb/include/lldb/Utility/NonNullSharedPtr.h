//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NONNULLSHAREDPTR_H
#define LLDB_UTILITY_NONNULLSHAREDPTR_H

#include <cassert>
#include <memory>
#include <utility>

namespace lldb_private {

/// A non-nullable shared pointer that always holds a valid object.
///
/// NonNullSharedPtr is a smart pointer wrapper around std::shared_ptr that
/// guarantees the pointer is never null.
///
/// This class is used for enforcing invariants at the type level and
/// eliminating entire classes of null pointer bugs.
///
/// @tparam T The type of object to manage. Must be default-constructible.
template <typename T> class NonNullSharedPtr : private std::shared_ptr<T> {
  using Base = std::shared_ptr<T>;

public:
  NonNullSharedPtr(const std::shared_ptr<T> &t)
      : Base(t ? t : std::make_shared<T>()) {
    assert(t && "NonNullSharedPtr constructed from nullptr");
  }

  NonNullSharedPtr(std::shared_ptr<T> &&t) : Base(std::move(t)) {
    const auto b = static_cast<bool>(*this);
    assert(b && "NonNullSharedPtr constructed from nullptr");
    if (!b)
      Base::operator=(std::make_shared<T>());
  }

  NonNullSharedPtr(const NonNullSharedPtr &other) : Base(other) {}

  NonNullSharedPtr(NonNullSharedPtr &&other) : Base(std::move(other)) {}

  NonNullSharedPtr &operator=(const NonNullSharedPtr &other) {
    Base::operator=(other);
    return *this;
  }

  NonNullSharedPtr &operator=(NonNullSharedPtr &&other) {
    Base::operator=(std::move(other));
    return *this;
  }

  using Base::operator*;
  using Base::operator->;
  using Base::get;
  using Base::use_count;
  using Base::operator bool;

  void swap(NonNullSharedPtr &other) { Base::swap(other); }

  /// Explicitly deleted operations that could introduce nullptr.
  /// @{
  void reset() = delete;
  void reset(T *ptr) = delete;
  /// @}
};

} // namespace lldb_private

/// Specialized swap function for NonNullSharedPtr to enable argument-dependent
/// lookup (ADL) and efficient swapping.
template <typename T>
void swap(lldb_private::NonNullSharedPtr<T> &lhs,
          lldb_private::NonNullSharedPtr<T> &rhs) {
  lhs.swap(rhs);
}

#endif
