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

#include "MoveOnly.h"
#include "test_macros.h"

template <class T>
struct PointerTo {
  using value_type = T;
  T& operator*() const;
};

// Copying the underlying object between pointers (or dereferenceable classes) works. This is a non-exhaustive check
// because this functionality comes from `indirectly_movable`.
static_assert( std::indirectly_movable_storable<int*, int*>);
static_assert( std::indirectly_movable_storable<const int*, int*>);
static_assert(!std::indirectly_movable_storable<int*, const int*>);
static_assert(!std::indirectly_movable_storable<const int*, const int*>);
static_assert( std::indirectly_movable_storable<int*, int[2]>);
static_assert(!std::indirectly_movable_storable<int[2], int*>);
static_assert( std::indirectly_movable_storable<MoveOnly*, MoveOnly*>);
static_assert( std::indirectly_movable_storable<PointerTo<MoveOnly>, PointerTo<MoveOnly>>);

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// `ValueType`.
struct NoAssignment {
  struct ValueType;

  struct ReferenceType {
    ReferenceType& operator=(ValueType) = delete;
  };

  // `ValueType` is convertible to `ReferenceType` but not assignable to it. This is implemented by explicitly deleting
  // `operator=(ValueType)` in `ReferenceType`.
  struct ValueType {
    operator ReferenceType&() const;
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

// The case when `indirectly_writable<iter_rvalue_reference>` but not `indirectly_writable<iter_value>` (you can
// do `ReferenceType r = ValueType();` but not `r = ValueType();`).
static_assert( std::indirectly_writable<NoAssignment, std::iter_rvalue_reference_t<NoAssignment>>);
static_assert(!std::indirectly_writable<NoAssignment, std::iter_value_t<NoAssignment>>);
static_assert(!std::indirectly_movable_storable<NoAssignment, NoAssignment>);

struct DeletedMoveCtor {
  DeletedMoveCtor(DeletedMoveCtor&&) = delete;
  DeletedMoveCtor& operator=(DeletedMoveCtor&&) = default;
};

struct DeletedMoveAssignment {
  DeletedMoveAssignment(DeletedMoveAssignment&&) = default;
  DeletedMoveAssignment& operator=(DeletedMoveAssignment&&) = delete;
};

static_assert(!std::indirectly_movable_storable<DeletedMoveCtor*, DeletedMoveCtor*>);
static_assert(!std::indirectly_movable_storable<DeletedMoveAssignment*, DeletedMoveAssignment*>);

struct InconsistentIterator {
  struct ValueType;

  struct ReferenceType {
    ReferenceType& operator=(ValueType const&);
  };

  struct ValueType {
    ValueType() = default;
    ValueType(const ReferenceType&);
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

// `ValueType` can be constructed with a `ReferenceType` and assigned to a `ReferenceType`, so it does model
// `indirectly_movable_storable`.
static_assert( std::indirectly_movable_storable<InconsistentIterator, InconsistentIterator>);

// ReferenceType is a (proxy) reference for ValueType, but ValueType is not constructible from ReferenceType.
struct NotConstructibleFromRefIn {
  struct CommonType { };

  struct ReferenceType {
    operator CommonType&() const;
  };

  struct ValueType {
    ValueType(ReferenceType) = delete;
    operator CommonType&() const;
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<NotConstructibleFromRefIn::ValueType,
                                   NotConstructibleFromRefIn::ReferenceType, X, Y> {
  using type = NotConstructibleFromRefIn::CommonType&;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<NotConstructibleFromRefIn::ReferenceType,
                                   NotConstructibleFromRefIn::ValueType, X, Y> {
  using type = NotConstructibleFromRefIn::CommonType&;
};

static_assert(std::common_reference_with<NotConstructibleFromRefIn::ValueType&,
    NotConstructibleFromRefIn::ReferenceType&>);

struct AssignableFromAnything {
  template<class T>
  AssignableFromAnything& operator=(T&&);
};

// A type that can't be constructed from its own reference isn't `indirectly_movable_storable`, even when assigning it
// to a type that can be assigned from anything.
static_assert( std::indirectly_movable_storable<int*, AssignableFromAnything*>);
static_assert(!std::indirectly_movable_storable<NotConstructibleFromRefIn, AssignableFromAnything*>);

// ReferenceType is a (proxy) reference for ValueType, but ValueType is not assignable from ReferenceType.
struct NotAssignableFromRefIn {
  struct CommonType { };

  struct ReferenceType {
    operator CommonType&() const;
  };

  struct ValueType {
    ValueType(ReferenceType);
    ValueType& operator=(ReferenceType) = delete;
    operator CommonType&() const;
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<NotAssignableFromRefIn::ValueType,
                                   NotAssignableFromRefIn::ReferenceType, X, Y> {
  using type = NotAssignableFromRefIn::CommonType&;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<NotAssignableFromRefIn::ReferenceType,
                                   NotAssignableFromRefIn::ValueType, X, Y> {
  using type = NotAssignableFromRefIn::CommonType&;
};

static_assert(std::common_reference_with<NotAssignableFromRefIn::ValueType&, NotAssignableFromRefIn::ReferenceType&>);

// A type that can't be assigned from its own reference isn't `indirectly_movable_storable`, even when assigning it
// to a type that can be assigned from anything.
static_assert(!std::indirectly_movable_storable<NotAssignableFromRefIn, AssignableFromAnything*>);
