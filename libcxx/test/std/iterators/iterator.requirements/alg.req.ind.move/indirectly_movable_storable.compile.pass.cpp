//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class In, class Out>
// concept indirectly_movable_storable;

#include <iterator>

#include "test_macros.h"

struct Empty {};

struct MoveOnlyConvertible;
struct AssignableToMoveOnly;

struct MoveOnly {
  MoveOnly(MoveOnly&&) = default;
  MoveOnly(MoveOnly const&) = delete;
  MoveOnly& operator=(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly const&) = delete;
  MoveOnly() = default;

  MoveOnly& operator=(MoveOnlyConvertible const&) = delete;
  MoveOnly& operator=(AssignableToMoveOnly const&);
};

template<class T, class ValueType = T>
struct PointerTo {
  using value_type = ValueType;
  T& operator*() const;
};

// MoveOnlyConvertible is convertible to MoveOnly, but not assignable to it. This is
// implemented by explicitly deleting "operator=(MoveOnlyConvertible)" in MoveOnly.
struct MoveOnlyConvertible {
  operator MoveOnly&() const;
};

// This type can be constructed with a MoveOnly and assigned to a MoveOnly, so it does
// model indirectly_movable_storable.
struct AssignableToMoveOnly {
  AssignableToMoveOnly() = default;
  AssignableToMoveOnly(const MoveOnly&);
};

struct DeletedMoveCtor {
  DeletedMoveCtor(DeletedMoveCtor&&) = delete;
  DeletedMoveCtor& operator=(DeletedMoveCtor&&) = default;
};

struct CommonType { };

struct NotConstructibleFromRefIn {
  struct ValueType {
    operator CommonType&() const;
  };

  struct ReferenceType {
    operator CommonType&() const;
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<NotConstructibleFromRefIn::ValueType,
                                   NotConstructibleFromRefIn::ReferenceType, X, Y> {
  using type = CommonType&;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<NotConstructibleFromRefIn::ReferenceType,
                                   NotConstructibleFromRefIn::ValueType, X, Y> {
  using type = CommonType&;
};

struct NotAssignableFromRefIn {
  struct ReferenceType;

  struct ValueType {
    ValueType(ReferenceType);
    ValueType& operator=(ReferenceType) = delete;
    operator CommonType&() const;
  };

  struct ReferenceType {
    operator CommonType&() const;
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<NotAssignableFromRefIn::ValueType,
                                   NotAssignableFromRefIn::ReferenceType, X, Y> {
  using type = CommonType&;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<NotAssignableFromRefIn::ReferenceType,
                                   NotAssignableFromRefIn::ValueType, X, Y> {
  using type = CommonType&;
};

struct AnyWritable {
  template<class T>
  AnyWritable& operator=(T&&);
};

struct AnyOutput {
  using value_type = AnyWritable;
  AnyWritable& operator*() const;
};

static_assert( std::indirectly_movable_storable<int*, int*>);
static_assert( std::indirectly_movable_storable<const int*, int *>);
static_assert( std::indirectly_movable_storable<int*, int[2]>);
static_assert( std::indirectly_movable_storable<Empty*, Empty*>);
static_assert( std::indirectly_movable_storable<MoveOnly*, MoveOnly*>);
static_assert( std::indirectly_movable_storable<PointerTo<MoveOnly>, PointerTo<MoveOnly>>);
// The case when indirectly_writable<iter_rvalue_reference> but not indirectly_writable<iter_value>.
static_assert( std::indirectly_writable<
                 PointerTo<MoveOnly, MoveOnlyConvertible>,
                 std::iter_rvalue_reference_t<
                    PointerTo<MoveOnly, MoveOnlyConvertible>>>);
static_assert(!std::indirectly_movable_storable<PointerTo<MoveOnly, MoveOnlyConvertible>,
                                                PointerTo<MoveOnly, MoveOnlyConvertible>>);
static_assert(!std::indirectly_movable_storable<DeletedMoveCtor*, DeletedMoveCtor*>);
static_assert( std::indirectly_movable_storable<PointerTo<MoveOnly, AssignableToMoveOnly>,
                                                PointerTo<MoveOnly, AssignableToMoveOnly>>);
static_assert(!std::indirectly_movable_storable<NotConstructibleFromRefIn, AnyOutput>);
static_assert(!std::indirectly_movable_storable<NotAssignableFromRefIn, AnyOutput>);
