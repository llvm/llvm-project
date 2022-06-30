//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBCXX_TEST_STD_UTILITIES_TUPLE_ASSIGN_TYPES_H
#define LIBCXX_TEST_STD_UTILITIES_TUPLE_ASSIGN_TYPES_H

#include "test_allocator.h"
#include <type_traits>

struct CopyAssign {
  int val;

  constexpr CopyAssign() = default;
  constexpr CopyAssign(int v) : val(v) {}

  constexpr CopyAssign& operator=(const CopyAssign&) = default;

  constexpr const CopyAssign& operator=(const CopyAssign&) const = delete;
  constexpr CopyAssign& operator=(CopyAssign&&) = delete;
  constexpr const CopyAssign& operator=(CopyAssign&&) const = delete;
};

struct ConstCopyAssign {
  mutable int val;

  constexpr ConstCopyAssign() = default;
  constexpr ConstCopyAssign(int v) : val(v) {}

  constexpr const ConstCopyAssign& operator=(const ConstCopyAssign& other) const {
    val = other.val;
    return *this;
  }

  constexpr ConstCopyAssign& operator=(const ConstCopyAssign&) = delete;
  constexpr ConstCopyAssign& operator=(ConstCopyAssign&&) = delete;
  constexpr const ConstCopyAssign& operator=(ConstCopyAssign&&) const = delete;
};

struct MoveAssign {
  int val;

  constexpr MoveAssign() = default;
  constexpr MoveAssign(int v) : val(v) {}

  constexpr MoveAssign& operator=(MoveAssign&&) = default;

  constexpr MoveAssign& operator=(const MoveAssign&) = delete;
  constexpr const MoveAssign& operator=(const MoveAssign&) const = delete;
  constexpr const MoveAssign& operator=(MoveAssign&&) const = delete;
};

struct ConstMoveAssign {
  mutable int val;

  constexpr ConstMoveAssign() = default;
  constexpr ConstMoveAssign(int v) : val(v) {}

  constexpr const ConstMoveAssign& operator=(ConstMoveAssign&& other) const {
    val = other.val;
    return *this;
  }

  constexpr ConstMoveAssign& operator=(const ConstMoveAssign&) = delete;
  constexpr const ConstMoveAssign& operator=(const ConstMoveAssign&) const = delete;
  constexpr ConstMoveAssign& operator=(ConstMoveAssign&&) = delete;
};

template <class T>
struct AssignableFrom {
  T v;

  constexpr AssignableFrom() = default;

  template <class U>
  constexpr AssignableFrom(U&& u)
    requires std::is_constructible_v<T, U&&>
  : v(std::forward<U>(u)) {}

  constexpr AssignableFrom& operator=(const T& t)
    requires std::is_copy_assignable_v<T>
  {
    v = t;
    return *this;
  }

  constexpr AssignableFrom& operator=(T&& t)
    requires std::is_move_assignable_v<T>
  {
    v = std::move(t);
    return *this;
  }

  constexpr const AssignableFrom& operator=(const T& t) const
    requires std::is_assignable_v<const T&, const T&>
  {
    v = t;
    return *this;
  }

  constexpr const AssignableFrom& operator=(T&& t) const
    requires std::is_assignable_v<const T&, T&&>
  {
    v = std::move(t);
    return *this;
  }
};

struct TracedAssignment {
  int copyAssign = 0;
  mutable int constCopyAssign = 0;
  int moveAssign = 0;
  mutable int constMoveAssign = 0;

  constexpr TracedAssignment() = default;

  constexpr TracedAssignment& operator=(const TracedAssignment&) {
    copyAssign++;
    return *this;
  }
  constexpr const TracedAssignment& operator=(const TracedAssignment&) const {
    constCopyAssign++;
    return *this;
  }
  constexpr TracedAssignment& operator=(TracedAssignment&&) {
    moveAssign++;
    return *this;
  }
  constexpr const TracedAssignment& operator=(TracedAssignment&&) const {
    constMoveAssign++;
    return *this;
  }
};

#endif
