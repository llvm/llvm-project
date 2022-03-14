//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// concept copyable = see below;

#include <concepts>

#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "type_classification/copyable.h"

static_assert(std::copyable<int>);
static_assert(std::copyable<int volatile>);
static_assert(std::copyable<int*>);
static_assert(std::copyable<int const*>);
static_assert(std::copyable<int volatile*>);
static_assert(std::copyable<int volatile const*>);
static_assert(std::copyable<int (*)()>);

struct S {};
static_assert(std::copyable<S>);
static_assert(std::copyable<int S::*>);
static_assert(std::copyable<int (S::*)()>);
static_assert(std::copyable<int (S::*)() noexcept>);
static_assert(std::copyable<int (S::*)() &>);
static_assert(std::copyable<int (S::*)() & noexcept>);
static_assert(std::copyable<int (S::*)() &&>);
static_assert(std::copyable<int (S::*)() && noexcept>);
static_assert(std::copyable<int (S::*)() const>);
static_assert(std::copyable<int (S::*)() const noexcept>);
static_assert(std::copyable<int (S::*)() const&>);
static_assert(std::copyable<int (S::*)() const & noexcept>);
static_assert(std::copyable<int (S::*)() const&&>);
static_assert(std::copyable<int (S::*)() const && noexcept>);
static_assert(std::copyable<int (S::*)() volatile>);
static_assert(std::copyable<int (S::*)() volatile noexcept>);
static_assert(std::copyable<int (S::*)() volatile&>);
static_assert(std::copyable<int (S::*)() volatile & noexcept>);
static_assert(std::copyable<int (S::*)() volatile&&>);
static_assert(std::copyable<int (S::*)() volatile && noexcept>);
static_assert(std::copyable<int (S::*)() const volatile>);
static_assert(std::copyable<int (S::*)() const volatile noexcept>);
static_assert(std::copyable<int (S::*)() const volatile&>);
static_assert(std::copyable<int (S::*)() const volatile & noexcept>);
static_assert(std::copyable<int (S::*)() const volatile&&>);
static_assert(std::copyable<int (S::*)() const volatile && noexcept>);

static_assert(std::copyable<std::vector<int> >);
static_assert(std::copyable<std::deque<int> >);
static_assert(std::copyable<std::forward_list<int> >);
static_assert(std::copyable<std::list<int> >);
static_assert(std::copyable<std::shared_ptr<std::unique_ptr<int> > >);
static_assert(std::copyable<std::optional<std::vector<int> > >);
static_assert(std::copyable<std::vector<int> >);
static_assert(std::copyable<std::vector<std::unique_ptr<int> > >);

static_assert(std::copyable<has_volatile_member>);
static_assert(std::copyable<has_array_member>);

// Not objects
static_assert(!std::copyable<void>);
static_assert(!std::copyable<int&>);
static_assert(!std::copyable<int const&>);
static_assert(!std::copyable<int volatile&>);
static_assert(!std::copyable<int const volatile&>);
static_assert(!std::copyable<int&&>);
static_assert(!std::copyable<int const&&>);
static_assert(!std::copyable<int volatile&&>);
static_assert(!std::copyable<int const volatile&&>);
static_assert(!std::copyable<int()>);
static_assert(!std::copyable<int (&)()>);
static_assert(!std::copyable<int[5]>);

// Not copy constructible or copy assignable
static_assert(!std::copyable<std::unique_ptr<int> >);

// Not assignable
static_assert(!std::copyable<int const>);
static_assert(!std::copyable<int const volatile>);
static_assert(std::copyable<const_copy_assignment const>);
static_assert(!std::copyable<volatile_copy_assignment volatile>);
static_assert(std::copyable<cv_copy_assignment const volatile>);

static_assert(!std::copyable<no_copy_constructor>);
static_assert(!std::copyable<no_copy_assignment>);

static_assert(std::is_copy_assignable_v<no_copy_assignment_mutable>);
static_assert(!std::copyable<no_copy_assignment_mutable>);
static_assert(!std::copyable<derived_from_noncopyable>);
static_assert(!std::copyable<has_noncopyable>);
static_assert(!std::copyable<has_const_member>);
static_assert(!std::copyable<has_cv_member>);
static_assert(!std::copyable<has_lvalue_reference_member>);
static_assert(!std::copyable<has_rvalue_reference_member>);
static_assert(!std::copyable<has_function_ref_member>);

static_assert(
    !std::assignable_from<deleted_assignment_from_const_rvalue&,
                          deleted_assignment_from_const_rvalue const>);
static_assert(!std::copyable<deleted_assignment_from_const_rvalue>);

int main(int, char**) { return 0; }
