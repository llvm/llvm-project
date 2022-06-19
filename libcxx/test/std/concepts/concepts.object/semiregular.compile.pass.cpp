//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// concept semiregular = see below;

#include <concepts>

#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "type_classification/semiregular.h"

static_assert(std::semiregular<int>);
static_assert(std::semiregular<int volatile>);
static_assert(std::semiregular<int*>);
static_assert(std::semiregular<int const*>);
static_assert(std::semiregular<int volatile*>);
static_assert(std::semiregular<int volatile const*>);
static_assert(std::semiregular<int (*)()>);

struct S {};
static_assert(std::semiregular<S>);
static_assert(std::semiregular<int S::*>);
static_assert(std::semiregular<int (S::*)()>);
static_assert(std::semiregular<int (S::*)() noexcept>);
static_assert(std::semiregular<int (S::*)() &>);
static_assert(std::semiregular<int (S::*)() & noexcept>);
static_assert(std::semiregular<int (S::*)() &&>);
static_assert(std::semiregular<int (S::*)() && noexcept>);
static_assert(std::semiregular<int (S::*)() const>);
static_assert(std::semiregular<int (S::*)() const noexcept>);
static_assert(std::semiregular<int (S::*)() const&>);
static_assert(std::semiregular<int (S::*)() const & noexcept>);
static_assert(std::semiregular<int (S::*)() const&&>);
static_assert(std::semiregular<int (S::*)() const && noexcept>);
static_assert(std::semiregular<int (S::*)() volatile>);
static_assert(std::semiregular<int (S::*)() volatile noexcept>);
static_assert(std::semiregular<int (S::*)() volatile&>);
static_assert(std::semiregular<int (S::*)() volatile & noexcept>);
static_assert(std::semiregular<int (S::*)() volatile&&>);
static_assert(std::semiregular<int (S::*)() volatile && noexcept>);
static_assert(std::semiregular<int (S::*)() const volatile>);
static_assert(std::semiregular<int (S::*)() const volatile noexcept>);
static_assert(std::semiregular<int (S::*)() const volatile&>);
static_assert(std::semiregular<int (S::*)() const volatile & noexcept>);
static_assert(std::semiregular<int (S::*)() const volatile&&>);
static_assert(std::semiregular<int (S::*)() const volatile && noexcept>);

static_assert(std::semiregular<std::vector<int> >);
static_assert(std::semiregular<std::deque<int> >);
static_assert(std::semiregular<std::forward_list<int> >);
static_assert(std::semiregular<std::list<int> >);
static_assert(std::semiregular<std::shared_ptr<std::unique_ptr<int> > >);
static_assert(std::semiregular<std::optional<std::vector<int> > >);
static_assert(std::semiregular<std::vector<int> >);
static_assert(std::semiregular<std::vector<std::unique_ptr<int> > >);

static_assert(std::semiregular<has_volatile_member>);
static_assert(std::semiregular<has_array_member>);

// Not objects
static_assert(!std::semiregular<void>);
static_assert(!std::semiregular<int&>);
static_assert(!std::semiregular<int const&>);
static_assert(!std::semiregular<int volatile&>);
static_assert(!std::semiregular<int const volatile&>);
static_assert(!std::semiregular<int&&>);
static_assert(!std::semiregular<int const&&>);
static_assert(!std::semiregular<int volatile&&>);
static_assert(!std::semiregular<int const volatile&&>);
static_assert(!std::semiregular<int()>);
static_assert(!std::semiregular<int (&)()>);
static_assert(!std::semiregular<int[5]>);

// Not copyable
static_assert(!std::semiregular<std::unique_ptr<int> >);
static_assert(!std::semiregular<int const>);
static_assert(!std::semiregular<int const volatile>);
static_assert(std::semiregular<const_copy_assignment const>);
static_assert(!std::semiregular<volatile_copy_assignment volatile>);
static_assert(std::semiregular<cv_copy_assignment const volatile>);
static_assert(!std::semiregular<no_copy_constructor>);
static_assert(!std::semiregular<no_copy_assignment>);
static_assert(!std::semiregular<no_copy_assignment_mutable>);
static_assert(!std::semiregular<derived_from_noncopyable>);
static_assert(!std::semiregular<has_noncopyable>);
static_assert(!std::semiregular<has_const_member>);
static_assert(!std::semiregular<has_cv_member>);
static_assert(!std::semiregular<has_lvalue_reference_member>);
static_assert(!std::semiregular<has_rvalue_reference_member>);
static_assert(!std::semiregular<has_function_ref_member>);
static_assert(!std::semiregular<deleted_assignment_from_const_rvalue>);

// Not default_initialzable
static_assert(!std::semiregular<std::runtime_error>);
static_assert(
    !std::semiregular<std::tuple<std::runtime_error, std::overflow_error> >);
static_assert(!std::semiregular<std::nullopt_t>);
static_assert(!std::semiregular<no_copy_constructor>);
static_assert(!std::semiregular<no_copy_assignment>);
static_assert(std::is_copy_assignable_v<no_copy_assignment_mutable>);
static_assert(!std::semiregular<no_copy_assignment_mutable>);
static_assert(!std::semiregular<derived_from_noncopyable>);
static_assert(!std::semiregular<has_noncopyable>);

static_assert(!std::semiregular<no_default_ctor>);
static_assert(!std::semiregular<derived_from_non_default_initializable>);
static_assert(!std::semiregular<has_non_default_initializable>);

static_assert(!std::semiregular<deleted_default_ctor>);
static_assert(!std::semiregular<derived_from_deleted_default_ctor>);
static_assert(!std::semiregular<has_deleted_default_ctor>);

int main(int, char**) { return 0; }
