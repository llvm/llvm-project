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
// concept regular = see below;

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

#include "type_classification/moveconstructible.h"
#include "type_classification/semiregular.h"

static_assert(std::regular<int>);
static_assert(std::regular<float>);
static_assert(std::regular<double>);
static_assert(std::regular<long double>);
static_assert(std::regular<int volatile>);
static_assert(std::regular<void*>);
static_assert(std::regular<int*>);
static_assert(std::regular<int const*>);
static_assert(std::regular<int volatile*>);
static_assert(std::regular<int volatile const*>);
static_assert(std::regular<int (*)()>);

struct S {};
static_assert(!std::regular<S>);
static_assert(std::regular<int S::*>);
static_assert(std::regular<int (S::*)()>);
static_assert(std::regular<int (S::*)() noexcept>);
static_assert(std::regular<int (S::*)() &>);
static_assert(std::regular<int (S::*)() & noexcept>);
static_assert(std::regular<int (S::*)() &&>);
static_assert(std::regular<int (S::*)() && noexcept>);
static_assert(std::regular<int (S::*)() const>);
static_assert(std::regular<int (S::*)() const noexcept>);
static_assert(std::regular<int (S::*)() const&>);
static_assert(std::regular<int (S::*)() const & noexcept>);
static_assert(std::regular<int (S::*)() const&&>);
static_assert(std::regular<int (S::*)() const && noexcept>);
static_assert(std::regular<int (S::*)() volatile>);
static_assert(std::regular<int (S::*)() volatile noexcept>);
static_assert(std::regular<int (S::*)() volatile&>);
static_assert(std::regular<int (S::*)() volatile & noexcept>);
static_assert(std::regular<int (S::*)() volatile&&>);
static_assert(std::regular<int (S::*)() volatile && noexcept>);
static_assert(std::regular<int (S::*)() const volatile>);
static_assert(std::regular<int (S::*)() const volatile noexcept>);
static_assert(std::regular<int (S::*)() const volatile&>);
static_assert(std::regular<int (S::*)() const volatile & noexcept>);
static_assert(std::regular<int (S::*)() const volatile&&>);
static_assert(std::regular<int (S::*)() const volatile && noexcept>);

union U {};
static_assert(!std::regular<U>);
static_assert(std::regular<int U::*>);
static_assert(std::regular<int (U::*)()>);
static_assert(std::regular<int (U::*)() noexcept>);
static_assert(std::regular<int (U::*)() &>);
static_assert(std::regular<int (U::*)() & noexcept>);
static_assert(std::regular<int (U::*)() &&>);
static_assert(std::regular<int (U::*)() && noexcept>);
static_assert(std::regular<int (U::*)() const>);
static_assert(std::regular<int (U::*)() const noexcept>);
static_assert(std::regular<int (U::*)() const&>);
static_assert(std::regular<int (U::*)() const & noexcept>);
static_assert(std::regular<int (U::*)() const&&>);
static_assert(std::regular<int (U::*)() const && noexcept>);
static_assert(std::regular<int (U::*)() volatile>);
static_assert(std::regular<int (U::*)() volatile noexcept>);
static_assert(std::regular<int (U::*)() volatile&>);
static_assert(std::regular<int (U::*)() volatile & noexcept>);
static_assert(std::regular<int (U::*)() volatile&&>);
static_assert(std::regular<int (U::*)() volatile && noexcept>);
static_assert(std::regular<int (U::*)() const volatile>);
static_assert(std::regular<int (U::*)() const volatile noexcept>);
static_assert(std::regular<int (U::*)() const volatile&>);
static_assert(std::regular<int (U::*)() const volatile & noexcept>);
static_assert(std::regular<int (U::*)() const volatile&&>);
static_assert(std::regular<int (U::*)() const volatile && noexcept>);

static_assert(std::regular<std::vector<int> >);
static_assert(std::regular<std::deque<int> >);
static_assert(std::regular<std::forward_list<int> >);
static_assert(std::regular<std::list<int> >);
static_assert(std::regular<std::shared_ptr<std::unique_ptr<int> > >);
static_assert(std::regular<std::optional<std::vector<int> > >);
static_assert(std::regular<std::vector<int> >);
static_assert(std::regular<std::vector<std::unique_ptr<int> > >);
static_assert(std::semiregular<std::in_place_t> &&
              !std::regular<std::in_place_t>);

static_assert(!std::regular<has_volatile_member>);
static_assert(!std::regular<has_array_member>);

// Not objects
static_assert(!std::regular<void>);
static_assert(!std::regular<int&>);
static_assert(!std::regular<int const&>);
static_assert(!std::regular<int volatile&>);
static_assert(!std::regular<int const volatile&>);
static_assert(!std::regular<int&&>);
static_assert(!std::regular<int const&&>);
static_assert(!std::regular<int volatile&&>);
static_assert(!std::regular<int const volatile&&>);
static_assert(!std::regular<int()>);
static_assert(!std::regular<int (&)()>);
static_assert(!std::regular<int[5]>);

// not copyable
static_assert(!std::regular<std::unique_ptr<int> >);
static_assert(!std::regular<int const>);
static_assert(!std::regular<int const volatile>);
static_assert(!std::regular<volatile_copy_assignment volatile>);
static_assert(!std::regular<no_copy_constructor>);
static_assert(!std::regular<no_copy_assignment>);
static_assert(!std::regular<no_copy_assignment_mutable>);
static_assert(!std::regular<derived_from_noncopyable>);
static_assert(!std::regular<has_noncopyable>);
static_assert(!std::regular<has_const_member>);
static_assert(!std::regular<has_cv_member>);
static_assert(!std::regular<has_lvalue_reference_member>);
static_assert(!std::regular<has_rvalue_reference_member>);
static_assert(!std::regular<has_function_ref_member>);
static_assert(!std::regular<deleted_assignment_from_const_rvalue>);

// not default_initializable
static_assert(!std::regular<std::runtime_error>);
static_assert(
    !std::regular<std::tuple<std::runtime_error, std::overflow_error> >);
static_assert(!std::regular<std::nullopt_t>);
static_assert(!std::regular<no_copy_constructor>);
static_assert(!std::regular<no_copy_assignment>);
static_assert(std::is_copy_assignable_v<no_copy_assignment_mutable> &&
              !std::regular<no_copy_assignment_mutable>);
static_assert(!std::regular<derived_from_noncopyable>);
static_assert(!std::regular<has_noncopyable>);

static_assert(!std::regular<derived_from_non_default_initializable>);
static_assert(!std::regular<has_non_default_initializable>);

// not equality_comparable
static_assert(!std::regular<const_copy_assignment const>);
static_assert(!std::regular<cv_copy_assignment const volatile>);

struct is_equality_comparable {
  bool operator==(is_equality_comparable const&) const = default;
};
static_assert(std::regular<is_equality_comparable>);

int main(int, char**) { return 0; }
