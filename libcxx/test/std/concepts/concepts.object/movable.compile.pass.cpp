//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept movable = see below;

#include <concepts>

#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#ifndef _LIBCPP_HAS_NO_THREADS
#   include <mutex>
#endif

#include "type_classification/moveconstructible.h"
#include "type_classification/movable.h"

// Movable types
static_assert(std::movable<int>);
static_assert(std::movable<int volatile>);
static_assert(std::movable<int*>);
static_assert(std::movable<int const*>);
static_assert(std::movable<int volatile*>);
static_assert(std::movable<int const volatile*>);
static_assert(std::movable<int (*)()>);

struct S {};
static_assert(std::movable<S>);
static_assert(std::movable<int S::*>);
static_assert(std::movable<int (S::*)()>);
static_assert(std::movable<int (S::*)() noexcept>);
static_assert(std::movable<int (S::*)() &>);
static_assert(std::movable<int (S::*)() & noexcept>);
static_assert(std::movable<int (S::*)() &&>);
static_assert(std::movable<int (S::*)() && noexcept>);
static_assert(std::movable<int (S::*)() const>);
static_assert(std::movable<int (S::*)() const noexcept>);
static_assert(std::movable<int (S::*)() const&>);
static_assert(std::movable<int (S::*)() const & noexcept>);
static_assert(std::movable<int (S::*)() const&&>);
static_assert(std::movable<int (S::*)() const && noexcept>);
static_assert(std::movable<int (S::*)() volatile>);
static_assert(std::movable<int (S::*)() volatile noexcept>);
static_assert(std::movable<int (S::*)() volatile&>);
static_assert(std::movable<int (S::*)() volatile & noexcept>);
static_assert(std::movable<int (S::*)() volatile&&>);
static_assert(std::movable<int (S::*)() volatile && noexcept>);
static_assert(std::movable<int (S::*)() const volatile>);
static_assert(std::movable<int (S::*)() const volatile noexcept>);
static_assert(std::movable<int (S::*)() const volatile&>);
static_assert(std::movable<int (S::*)() const volatile & noexcept>);
static_assert(std::movable<int (S::*)() const volatile&&>);
static_assert(std::movable<int (S::*)() const volatile && noexcept>);

static_assert(std::movable<std::deque<int> >);
static_assert(std::movable<std::forward_list<int> >);
static_assert(std::movable<std::list<int> >);
static_assert(std::movable<std::optional<std::vector<int> > >);
static_assert(std::movable<std::vector<int> >);

static_assert(std::movable<traditional_copy_assignment_only>);
static_assert(std::movable<has_volatile_member>);
static_assert(std::movable<has_array_member>);

// Not objects
static_assert(!std::movable<int&>);
static_assert(!std::movable<int const&>);
static_assert(!std::movable<int volatile&>);
static_assert(!std::movable<int const volatile&>);
static_assert(!std::movable<int&&>);
static_assert(!std::movable<int const&&>);
static_assert(!std::movable<int volatile&&>);
static_assert(!std::movable<int const volatile&&>);
static_assert(!std::movable<int()>);
static_assert(!std::movable<int (&)()>);
static_assert(!std::movable<int[5]>);

// Core non-move assignable.
static_assert(!std::movable<int const>);
static_assert(!std::movable<int const volatile>);

static_assert(!std::movable<DeletedMoveCtor>);
static_assert(!std::movable<ImplicitlyDeletedMoveCtor>);
static_assert(!std::movable<DeletedMoveAssign>);
static_assert(!std::movable<ImplicitlyDeletedMoveAssign>);
static_assert(!std::movable<NonMovable>);
static_assert(!std::movable<DerivedFromNonMovable>);
static_assert(!std::movable<HasANonMovable>);

static_assert(std::movable<cpp03_friendly>);
static_assert(std::movable<const_move_ctor>);
static_assert(std::movable<volatile_move_ctor>);
static_assert(std::movable<cv_move_ctor>);
static_assert(std::movable<multi_param_move_ctor>);
static_assert(!std::movable<not_quite_multi_param_move_ctor>);

static_assert(!std::assignable_from<copy_assign_with_mutable_parameter&,
                                    copy_assign_with_mutable_parameter>);
static_assert(!std::movable<copy_assign_with_mutable_parameter>);

static_assert(!std::movable<const_move_assignment>);
static_assert(std::movable<volatile_move_assignment>);
static_assert(!std::movable<cv_move_assignment>);

static_assert(!std::movable<const_move_assign_and_traditional_move_assign>);
static_assert(!std::movable<volatile_move_assign_and_traditional_move_assign>);
static_assert(!std::movable<cv_move_assign_and_traditional_move_assign>);
static_assert(std::movable<const_move_assign_and_default_ops>);
static_assert(std::movable<volatile_move_assign_and_default_ops>);
static_assert(std::movable<cv_move_assign_and_default_ops>);

static_assert(!std::movable<has_const_member>);
static_assert(!std::movable<has_cv_member>);
static_assert(!std::movable<has_lvalue_reference_member>);
static_assert(!std::movable<has_rvalue_reference_member>);
static_assert(!std::movable<has_function_ref_member>);

static_assert(std::movable<deleted_assignment_from_const_rvalue>);

// `move_constructible and assignable_from<T&, T>` implies `swappable<T>`,
// so there's nothing to test for the case of non-swappable.

int main(int, char**) { return 0; }
