//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_STD_UTILITIES_TUPLE_CNSTR_TYPES_H
#define LIBCXX_TEST_STD_UTILITIES_TUPLE_CNSTR_TYPES_H

#include "test_allocator.h"
#include "test_macros.h"
#include <type_traits>
#include <tuple>
#include <utility>

// Types that can be used to test copy/move operations

struct MutableCopy {
  int val;
  bool alloc_constructed = false;

  MutableCopy() = default;
  TEST_CONSTEXPR MutableCopy(int _val) : val(_val) {}
  MutableCopy(MutableCopy&)       = default;
  MutableCopy(const MutableCopy&) = delete;

  TEST_CONSTEXPR MutableCopy(std::allocator_arg_t, const test_allocator<int>&, MutableCopy& o)
      : val(o.val), alloc_constructed(true) {}
};

template <>
struct std::uses_allocator<MutableCopy, test_allocator<int> > : std::true_type {};

struct ConstCopy {
  int val;
  bool alloc_constructed = false;

  ConstCopy() = default;
  TEST_CONSTEXPR ConstCopy(int _val) : val(_val) {}
  ConstCopy(const ConstCopy&) = default;
  ConstCopy(ConstCopy&)       = delete;

  TEST_CONSTEXPR ConstCopy(std::allocator_arg_t, const test_allocator<int>&, const ConstCopy& o)
      : val(o.val), alloc_constructed(true) {}
};

template <>
struct std::uses_allocator<ConstCopy, test_allocator<int> > : std::true_type {};

struct MutableMove {
  int val;
  bool alloc_constructed = false;

  MutableMove() = default;
  TEST_CONSTEXPR MutableMove(int _val) : val(_val) {}
  MutableMove(MutableMove&&)       = default;
  MutableMove(const MutableMove&&) = delete;

  TEST_CONSTEXPR MutableMove(std::allocator_arg_t, const test_allocator<int>&, MutableMove&& o)
      : val(o.val), alloc_constructed(true) {}
};

template <>
struct std::uses_allocator<MutableMove, test_allocator<int> > : std::true_type {};

struct ConstMove {
  int val;
  bool alloc_constructed = false;

  ConstMove() = default;
  TEST_CONSTEXPR ConstMove(int _val) : val(_val) {}
  TEST_CONSTEXPR ConstMove(const ConstMove&& o) : val(o.val) {}
  ConstMove(ConstMove&&) = delete;

  TEST_CONSTEXPR ConstMove(std::allocator_arg_t, const test_allocator<int>&, const ConstMove&& o)
      : val(o.val), alloc_constructed(true) {}
};

template <>
struct std::uses_allocator<ConstMove, test_allocator<int> > : std::true_type {};

template <class T>
struct ConvertibleFrom {
  T v;
  bool alloc_constructed = false;

  ConvertibleFrom() = default;
  template <class U = T, typename std::enable_if<std::is_constructible<U, U&>::value, int>::type = 0>
  TEST_CONSTEXPR ConvertibleFrom(T& _v) : v(_v) {}
  template <
      class U                                                                                                   = T,
      typename std::enable_if<std::is_constructible<U, const U&>::value && !std::is_const<U>::value, int>::type = 0>
  TEST_CONSTEXPR ConvertibleFrom(const T& _v) : v(_v) {}
  template <class U = T, typename std::enable_if<std::is_constructible<U, U&&>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 ConvertibleFrom(T&& _v) : v(std::move(_v)) {}
  template <
      class U                                                                                                    = T,
      typename std::enable_if<std::is_constructible<U, const U&&>::value && !std::is_const<U>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 ConvertibleFrom(const T&& _v) : v(std::move(_v)) {}

  template <class U, typename std::enable_if<std::is_constructible<ConvertibleFrom, U&&>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 ConvertibleFrom(std::allocator_arg_t, const test_allocator<int>&, U&& _u)
      : ConvertibleFrom(std::forward<U>(_u)) {
    alloc_constructed = true;
  }
};

template <class T>
struct std::uses_allocator<ConvertibleFrom<T>, test_allocator<int> > : std::true_type {};

template <class T>
struct ExplicitConstructibleFrom {
  T v;
  bool alloc_constructed = false;

  explicit ExplicitConstructibleFrom() = default;
  template <class U = T, typename std::enable_if<std::is_constructible<U, U&>::value, int>::type = 0>
  TEST_CONSTEXPR explicit ExplicitConstructibleFrom(T& _v) : v(_v) {}
  template <
      class U                                                                                                   = T,
      typename std::enable_if<std::is_constructible<U, const U&>::value && !std::is_const<U>::value, int>::type = 0>
  TEST_CONSTEXPR explicit ExplicitConstructibleFrom(const T& _v) : v(_v) {}
  template <class U = T, typename std::enable_if<std::is_constructible<U, U&&>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 explicit ExplicitConstructibleFrom(T&& _v) : v(std::move(_v)) {}
  template <
      class U                                                                                                    = T,
      typename std::enable_if<std::is_constructible<U, const U&&>::value && !std::is_const<U>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 explicit ExplicitConstructibleFrom(const T&& _v) : v(std::move(_v)) {}

  template <class U,
            typename std::enable_if<std::is_constructible<ExplicitConstructibleFrom, U&&>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 ExplicitConstructibleFrom(std::allocator_arg_t, const test_allocator<int>&, U&& _u)
      : ExplicitConstructibleFrom(std::forward<U>(_u)) {
    alloc_constructed = true;
  }
};

template <class T>
struct std::uses_allocator<ExplicitConstructibleFrom<T>, test_allocator<int> > : std::true_type {};

struct TracedCopyMove {
  int nonConstCopy = 0;
  int constCopy = 0;
  int nonConstMove = 0;
  int constMove = 0;
  bool alloc_constructed = false;

  TracedCopyMove() = default;
  TEST_CONSTEXPR TracedCopyMove(const TracedCopyMove& other)
      : nonConstCopy(other.nonConstCopy),
        constCopy(other.constCopy + 1),
        nonConstMove(other.nonConstMove),
        constMove(other.constMove) {}
  TEST_CONSTEXPR TracedCopyMove(TracedCopyMove& other)
      : nonConstCopy(other.nonConstCopy + 1),
        constCopy(other.constCopy),
        nonConstMove(other.nonConstMove),
        constMove(other.constMove) {}

  TEST_CONSTEXPR TracedCopyMove(TracedCopyMove&& other)
      : nonConstCopy(other.nonConstCopy),
        constCopy(other.constCopy),
        nonConstMove(other.nonConstMove + 1),
        constMove(other.constMove) {}

  TEST_CONSTEXPR TracedCopyMove(const TracedCopyMove&& other)
      : nonConstCopy(other.nonConstCopy),
        constCopy(other.constCopy),
        nonConstMove(other.nonConstMove),
        constMove(other.constMove + 1) {}

  template <class U, typename std::enable_if<std::is_constructible<TracedCopyMove, U&&>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 TracedCopyMove(std::allocator_arg_t, const test_allocator<int>&, U&& _u)
      : TracedCopyMove(std::forward<U>(_u)) {
    alloc_constructed = true;
  }
};

template <>
struct std::uses_allocator<TracedCopyMove, test_allocator<int> > : std::true_type {};

// If the constructor tuple(tuple<UTypes...>&) is not available,
// the fallback call to `tuple(const tuple&) = default;` or any other
// constructor that takes const ref would increment the constCopy.
inline TEST_CONSTEXPR bool nonConstCopyCtrCalled(const TracedCopyMove& obj) {
  return obj.nonConstCopy == 1 && obj.constCopy == 0 && obj.constMove == 0 && obj.nonConstMove == 0;
}

// If the constructor tuple(const tuple<UTypes...>&&) is not available,
// the fallback call to `tuple(const tuple&) = default;` or any other
// constructor that takes const ref would increment the constCopy.
inline TEST_CONSTEXPR bool constMoveCtrCalled(const TracedCopyMove& obj) {
  return obj.nonConstMove == 0 && obj.constMove == 1 && obj.constCopy == 0 && obj.nonConstCopy == 0;
}

struct NoConstructorFromInt {};

#if TEST_STD_VER >= 11
struct CvtFromTupleRef : TracedCopyMove {
  CvtFromTupleRef() = default;
  TEST_CONSTEXPR_CXX14 CvtFromTupleRef(std::tuple<CvtFromTupleRef>& other)
      : TracedCopyMove(static_cast<TracedCopyMove&>(std::get<0>(other))) {}
};

struct ExplicitCtrFromTupleRef : TracedCopyMove {
  explicit ExplicitCtrFromTupleRef() = default;
  TEST_CONSTEXPR_CXX14 explicit ExplicitCtrFromTupleRef(std::tuple<ExplicitCtrFromTupleRef>& other)
      : TracedCopyMove(static_cast<TracedCopyMove&>(std::get<0>(other))) {}
};

struct CvtFromConstTupleRefRef : TracedCopyMove {
  CvtFromConstTupleRefRef() = default;
  TEST_CONSTEXPR_CXX14 CvtFromConstTupleRefRef(const std::tuple<CvtFromConstTupleRefRef>&& other)
      : TracedCopyMove(static_cast<const TracedCopyMove&&>(std::get<0>(other))) {}
};

struct ExplicitCtrFromConstTupleRefRef : TracedCopyMove {
  explicit ExplicitCtrFromConstTupleRefRef() = default;
  TEST_CONSTEXPR_CXX14 explicit ExplicitCtrFromConstTupleRefRef(
      std::tuple<const ExplicitCtrFromConstTupleRefRef>&& other)
      : TracedCopyMove(static_cast<const TracedCopyMove&&>(std::get<0>(other))) {}
};
#endif

#if TEST_STD_VER >= 20
template <class T>
void conversion_test(T);

template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({std::forward<Args>(args)...}); };
#endif

struct CopyAssign {
  int val;

  CopyAssign() = default;
  TEST_CONSTEXPR CopyAssign(int v) : val(v) {}

  CopyAssign& operator=(const CopyAssign&) = default;

  const CopyAssign& operator=(const CopyAssign&) const = delete;
  CopyAssign& operator=(CopyAssign&&)                  = delete;
  const CopyAssign& operator=(CopyAssign&&) const      = delete;
};

struct ConstCopyAssign {
  mutable int val;

  ConstCopyAssign() = default;
  TEST_CONSTEXPR ConstCopyAssign(int v) : val(v) {}

  TEST_CONSTEXPR_CXX14 const ConstCopyAssign& operator=(const ConstCopyAssign& other) const {
    val = other.val;
    return *this;
  }

  ConstCopyAssign& operator=(const ConstCopyAssign&)        = delete;
  ConstCopyAssign& operator=(ConstCopyAssign&&)             = delete;
  const ConstCopyAssign& operator=(ConstCopyAssign&&) const = delete;
};

struct MoveAssign {
  int val;

  MoveAssign() = default;
  TEST_CONSTEXPR MoveAssign(int v) : val(v) {}

  MoveAssign& operator=(MoveAssign&&) = default;

  MoveAssign& operator=(const MoveAssign&)             = delete;
  const MoveAssign& operator=(const MoveAssign&) const = delete;
  const MoveAssign& operator=(MoveAssign&&) const      = delete;
};

struct ConstMoveAssign {
  mutable int val;

  ConstMoveAssign() = default;
  TEST_CONSTEXPR ConstMoveAssign(int v) : val(v) {}

  TEST_CONSTEXPR_CXX14 const ConstMoveAssign& operator=(ConstMoveAssign&& other) const {
    val = other.val;
    return *this;
  }

  ConstMoveAssign& operator=(const ConstMoveAssign&)             = delete;
  const ConstMoveAssign& operator=(const ConstMoveAssign&) const = delete;
  ConstMoveAssign& operator=(ConstMoveAssign&&)                  = delete;
};

template <class T>
struct AssignableFrom {
  T v;

  AssignableFrom() = default;

  template <class U, typename std::enable_if<std::is_constructible<T, U&&>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 AssignableFrom(U&& u) : v(std::forward<U>(u)) {}

  template <class U = T, typename std::enable_if<std::is_copy_assignable<U>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 AssignableFrom& operator=(const T& t) {
    v = t;
    return *this;
  }

  template <class U = T, typename std::enable_if<std::is_move_assignable<U>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 AssignableFrom& operator=(T&& t) {
    v = std::move(t);
    return *this;
  }

  template <class U = T, typename std::enable_if<std::is_assignable<const U&, const U&>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 const AssignableFrom& operator=(const T& t) const {
    v = t;
    return *this;
  }

  template <class U = T, typename std::enable_if<std::is_assignable<const U&, U&&>::value, int>::type = 0>
  TEST_CONSTEXPR_CXX14 const AssignableFrom& operator=(T&& t) const {
    v = std::move(t);
    return *this;
  }
};

struct TracedAssignment {
  int copyAssign = 0;
  mutable int constCopyAssign = 0;
  int moveAssign = 0;
  mutable int constMoveAssign = 0;

  TracedAssignment() = default;

  TEST_CONSTEXPR_CXX14 TracedAssignment& operator=(const TracedAssignment&) {
    copyAssign++;
    return *this;
  }
  TEST_CONSTEXPR_CXX14 const TracedAssignment& operator=(const TracedAssignment&) const {
    constCopyAssign++;
    return *this;
  }
  TEST_CONSTEXPR_CXX14 TracedAssignment& operator=(TracedAssignment&&) {
    moveAssign++;
    return *this;
  }
  TEST_CONSTEXPR_CXX14 const TracedAssignment& operator=(TracedAssignment&&) const {
    constMoveAssign++;
    return *this;
  }
};
#endif
