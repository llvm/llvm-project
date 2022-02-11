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
// concept indirectly_copyable_storable;

#include <iterator>

#include "MoveOnly.h"
#include "test_macros.h"

struct CopyOnly {
  CopyOnly(CopyOnly&&) = delete;
  CopyOnly(CopyOnly const&) = default;
  CopyOnly& operator=(CopyOnly&&) = delete;
  CopyOnly& operator=(CopyOnly const&) = default;
  CopyOnly() = default;
};

template<class T>
struct PointerTo {
  using value_type = T;
  T& operator*() const;
};

// Copying the underlying object between pointers (or dereferenceable classes) works. This is a non-exhaustive check
// because this functionality comes from `indirectly_copyable`.
static_assert( std::indirectly_copyable_storable<int*, int*>);
static_assert( std::indirectly_copyable_storable<const int*, int*>);
static_assert(!std::indirectly_copyable_storable<int*, const int*>);
static_assert(!std::indirectly_copyable_storable<const int*, const int*>);
static_assert( std::indirectly_copyable_storable<int*, int[2]>);
static_assert(!std::indirectly_copyable_storable<int[2], int*>);
static_assert(!std::indirectly_copyable_storable<MoveOnly*, MoveOnly*>);
static_assert(!std::indirectly_copyable_storable<PointerTo<MoveOnly>, PointerTo<MoveOnly>>);
// `indirectly_copyable_storable` requires the type to be `copyable`, which in turns requires it to be `movable`.
static_assert(!std::indirectly_copyable_storable<CopyOnly*, CopyOnly*>);
static_assert(!std::indirectly_copyable_storable<PointerTo<CopyOnly>, PointerTo<CopyOnly>>);

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// non-const lvalue of `ValueType` (but all other forms of assignment from `ValueType` work).
struct NoLvalueAssignment {
  struct ValueType;

  struct ReferenceType {
    ReferenceType& operator=(ValueType const&);
    ReferenceType& operator=(ValueType&) = delete;
    ReferenceType& operator=(ValueType&&);
    ReferenceType& operator=(ValueType const&&);
  };

  struct ValueType {
    operator ReferenceType&() const;
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

static_assert( std::indirectly_writable<NoLvalueAssignment, std::iter_reference_t<NoLvalueAssignment>>);
static_assert(!std::indirectly_writable<NoLvalueAssignment, std::iter_value_t<NoLvalueAssignment>&>);
static_assert( std::indirectly_writable<NoLvalueAssignment, const std::iter_value_t<NoLvalueAssignment>&>);
static_assert( std::indirectly_writable<NoLvalueAssignment, std::iter_value_t<NoLvalueAssignment>&&>);
static_assert( std::indirectly_writable<NoLvalueAssignment, const std::iter_value_t<NoLvalueAssignment>&&>);
static_assert(!std::indirectly_copyable_storable<NoLvalueAssignment, NoLvalueAssignment>);

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// const lvalue of `ValueType` (but all other forms of assignment from `ValueType` work).
struct NoConstLvalueAssignment {
  struct ValueType;

  struct ReferenceType {
    ReferenceType& operator=(ValueType const&) = delete;
    ReferenceType& operator=(ValueType&);
    ReferenceType& operator=(ValueType&&);
    ReferenceType& operator=(ValueType const&&);
  };

  struct ValueType {
    operator ReferenceType&() const;
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

static_assert( std::indirectly_writable<NoConstLvalueAssignment, std::iter_reference_t<NoConstLvalueAssignment>>);
static_assert( std::indirectly_writable<NoConstLvalueAssignment, std::iter_value_t<NoConstLvalueAssignment>&>);
static_assert(!std::indirectly_writable<NoConstLvalueAssignment, const std::iter_value_t<NoConstLvalueAssignment>&>);
static_assert( std::indirectly_writable<NoConstLvalueAssignment, std::iter_value_t<NoConstLvalueAssignment>&&>);
static_assert( std::indirectly_writable<NoConstLvalueAssignment, const std::iter_value_t<NoConstLvalueAssignment>&&>);
static_assert(!std::indirectly_copyable_storable<NoConstLvalueAssignment, NoConstLvalueAssignment>);

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// non-const rvalue of `ValueType` (but all other forms of assignment from `ValueType` work).
struct NoRvalueAssignment {
  struct ValueType;

  struct ReferenceType {
    ReferenceType& operator=(ValueType const&);
    ReferenceType& operator=(ValueType&);
    ReferenceType& operator=(ValueType&&) = delete;
    ReferenceType& operator=(ValueType const&&);
  };

  struct ValueType {
    operator ReferenceType&() const;
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

static_assert( std::indirectly_writable<NoRvalueAssignment, std::iter_reference_t<NoRvalueAssignment>>);
static_assert( std::indirectly_writable<NoRvalueAssignment, std::iter_value_t<NoRvalueAssignment>&>);
static_assert( std::indirectly_writable<NoRvalueAssignment, const std::iter_value_t<NoRvalueAssignment>&>);
static_assert(!std::indirectly_writable<NoRvalueAssignment, std::iter_value_t<NoRvalueAssignment>&&>);
static_assert( std::indirectly_writable<NoRvalueAssignment, const std::iter_value_t<NoRvalueAssignment>&&>);
static_assert(!std::indirectly_copyable_storable<NoRvalueAssignment, NoRvalueAssignment>);

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// const rvalue of `ValueType` (but all other forms of assignment from `ValueType` work).
struct NoConstRvalueAssignment {
  struct ValueType;

  struct ReferenceType {
    ReferenceType& operator=(ValueType const&);
    ReferenceType& operator=(ValueType&);
    ReferenceType& operator=(ValueType&&);
    ReferenceType& operator=(ValueType const&&) = delete;
  };

  struct ValueType {
    operator ReferenceType&() const;
  };

  using value_type = ValueType;
  ReferenceType& operator*() const;
};

static_assert( std::indirectly_writable<NoConstRvalueAssignment, std::iter_reference_t<NoConstRvalueAssignment>>);
static_assert( std::indirectly_writable<NoConstRvalueAssignment, std::iter_value_t<NoConstRvalueAssignment>&>);
static_assert( std::indirectly_writable<NoConstRvalueAssignment, const std::iter_value_t<NoConstRvalueAssignment>&>);
static_assert( std::indirectly_writable<NoConstRvalueAssignment, std::iter_value_t<NoConstRvalueAssignment>&&>);
static_assert(!std::indirectly_writable<NoConstRvalueAssignment, const std::iter_value_t<NoConstRvalueAssignment>&&>);
static_assert(!std::indirectly_copyable_storable<NoConstRvalueAssignment, NoConstRvalueAssignment>);

struct DeletedCopyCtor {
  DeletedCopyCtor(DeletedCopyCtor const&) = delete;
  DeletedCopyCtor& operator=(DeletedCopyCtor const&) = default;
};

struct DeletedNonconstCopyCtor {
  DeletedNonconstCopyCtor(DeletedNonconstCopyCtor const&) = default;
  DeletedNonconstCopyCtor(DeletedNonconstCopyCtor&) = delete;
  DeletedNonconstCopyCtor& operator=(DeletedNonconstCopyCtor const&) = default;
};

struct DeletedMoveCtor {
  DeletedMoveCtor(DeletedMoveCtor&&) = delete;
  DeletedMoveCtor& operator=(DeletedMoveCtor&&) = default;
};

struct DeletedConstMoveCtor {
  DeletedConstMoveCtor(DeletedConstMoveCtor&&) = default;
  DeletedConstMoveCtor(DeletedConstMoveCtor const&&) = delete;
  DeletedConstMoveCtor& operator=(DeletedConstMoveCtor&&) = default;
};

struct DeletedCopyAssignment {
  DeletedCopyAssignment(DeletedCopyAssignment const&) = default;
  DeletedCopyAssignment& operator=(DeletedCopyAssignment const&) = delete;
};

struct DeletedNonconstCopyAssignment {
  DeletedNonconstCopyAssignment(DeletedNonconstCopyAssignment const&) = default;
  DeletedNonconstCopyAssignment& operator=(DeletedNonconstCopyAssignment const&) = default;
  DeletedNonconstCopyAssignment& operator=(DeletedNonconstCopyAssignment&) = delete;
};

struct DeletedMoveAssignment {
  DeletedMoveAssignment(DeletedMoveAssignment&&) = default;
  DeletedMoveAssignment& operator=(DeletedMoveAssignment&&) = delete;
};

struct DeletedConstMoveAssignment {
  DeletedConstMoveAssignment(DeletedConstMoveAssignment&&) = default;
  DeletedConstMoveAssignment& operator=(DeletedConstMoveAssignment&&) = delete;
};

static_assert(!std::indirectly_copyable_storable<DeletedCopyCtor*, DeletedCopyCtor*>);
static_assert(!std::indirectly_copyable_storable<DeletedNonconstCopyCtor*, DeletedNonconstCopyCtor*>);
static_assert(!std::indirectly_copyable_storable<DeletedMoveCtor*, DeletedMoveCtor*>);
static_assert(!std::indirectly_copyable_storable<DeletedConstMoveCtor*, DeletedConstMoveCtor*>);
static_assert(!std::indirectly_copyable_storable<DeletedCopyAssignment*, DeletedCopyAssignment*>);
static_assert(!std::indirectly_copyable_storable<DeletedNonconstCopyAssignment*, DeletedNonconstCopyAssignment*>);
static_assert(!std::indirectly_copyable_storable<DeletedMoveAssignment*, DeletedMoveAssignment*>);
static_assert(!std::indirectly_copyable_storable<DeletedConstMoveAssignment*, DeletedConstMoveAssignment*>);

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
// `indirectly_copyable_storable`.
static_assert( std::indirectly_copyable_storable<InconsistentIterator, InconsistentIterator>);

struct CommonType { };

// ReferenceType is a (proxy) reference for ValueType, but ValueType is not constructible from ReferenceType.
struct NotConstructibleFromRefIn {
  struct ReferenceType;

  struct ValueType {
    ValueType(ReferenceType) = delete;
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

static_assert(std::common_reference_with<NotConstructibleFromRefIn::ValueType&,
    NotConstructibleFromRefIn::ReferenceType&>);

struct AssignableFromAnything {
  template<class T>
  AssignableFromAnything& operator=(T&&);
};

// A type that can't be constructed from its own reference isn't `indirectly_copyable_storable`, even when assigning it
// to a type that can be assigned from anything.
static_assert(!std::indirectly_copyable_storable<NotConstructibleFromRefIn, AssignableFromAnything*>);

// ReferenceType is a (proxy) reference for ValueType, but ValueType is not assignable from ReferenceType.
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

static_assert(std::common_reference_with<NotAssignableFromRefIn::ValueType&, NotAssignableFromRefIn::ReferenceType&>);

// A type that can't be assigned from its own reference isn't `indirectly_copyable_storable`, even when assigning it
// to a type that can be assigned from anything.
static_assert(!std::indirectly_copyable_storable<NotAssignableFromRefIn, AssignableFromAnything*>);
