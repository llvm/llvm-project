//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s \
// RUN:   -Xclang -verify-ignore-unexpected=note,warning
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s \
// RUN:   -Xclang -verify-ignore-unexpected=note,warning

// expected-no-diagnostics

// Checks the member types of sycl::multi_ptr and the invariants that SYCL 2020
// 4.7.7.1. requires of them, for every combination of address space and
// decoration. Also checks that the void specializations omit the members that
// the specification says they do not provide.

#include <sycl/sycl.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace as = sycl::access;

template <typename DataT, as::address_space Space, as::decorated Decorated>
void checkMemberTypes() {
  using multi_ptr_t = sycl::multi_ptr<DataT, Space, Decorated>;

  static_assert(multi_ptr_t::address_space == Space);
  static_assert(multi_ptr_t::is_decorated == (Decorated == as::decorated::yes));

  static_assert(std::is_same_v<typename multi_ptr_t::value_type, DataT>);
  static_assert(
      std::is_same_v<typename multi_ptr_t::difference_type, std::ptrdiff_t>);
  static_assert(std::is_same_v<typename multi_ptr_t::iterator_category,
                               std::random_access_iterator_tag>);

  // The decorated and the undecorated types may only differ in the address
  // space.
  static_assert(
      std::is_same_v<sycl::remove_decoration_t<typename multi_ptr_t::pointer>,
                     std::add_pointer_t<DataT>>);
  static_assert(
      std::is_same_v<sycl::remove_decoration_t<typename multi_ptr_t::reference>,
                     std::add_lvalue_reference_t<DataT>>);

  // The non-decorated interface exposes plain pointers and references.
  if constexpr (Decorated == as::decorated::no) {
    static_assert(std::is_same_v<typename multi_ptr_t::pointer,
                                 std::add_pointer_t<DataT>>);
    static_assert(std::is_same_v<typename multi_ptr_t::reference,
                                 std::add_lvalue_reference_t<DataT>>);
  }
}

template <typename DataT, as::address_space Space> void checkDecorations() {
  checkMemberTypes<DataT, Space, as::decorated::yes>();
  checkMemberTypes<DataT, Space, as::decorated::no>();
}

template <typename DataT> void checkAddressSpaces() {
  checkDecorations<DataT, as::address_space::global_space>();
  checkDecorations<DataT, as::address_space::local_space>();
  checkDecorations<DataT, as::address_space::private_space>();
  checkDecorations<DataT, as::address_space::generic_space>();
}

template void checkAddressSpaces<int>();
template void checkAddressSpaces<const int>();
template void checkAddressSpaces<double>();

// sycl::remove_decoration is safe to use with a non-decorated type and returns
// the same type.
static_assert(std::is_same_v<sycl::remove_decoration_t<int>, int>);
static_assert(std::is_same_v<sycl::remove_decoration_t<int *>, int *>);
static_assert(
    std::is_same_v<sycl::remove_decoration_t<const int &>, const int &>);

// SYCL 2020 4.7.7.1.: multi_ptr<void> does not provide the reference or
// const_reference types, the access operators, the arithmetic operators or the
// prefetch member function.
template <typename T, typename = void> struct HasReference : std::false_type {};
template <typename T>
struct HasReference<T, std::void_t<typename T::reference>> : std::true_type {};

template <typename T, typename = void>
struct HasDereference : std::false_type {};
template <typename T>
struct HasDereference<T, std::void_t<decltype(*std::declval<T &>())>>
    : std::true_type {};

template <typename T, typename = void> struct HasPrefetch : std::false_type {};
template <typename T>
struct HasPrefetch<T, std::void_t<decltype(std::declval<T &>().prefetch(1))>>
    : std::true_type {};

template <typename T, typename = void> struct HasIncrement : std::false_type {};
template <typename T>
struct HasIncrement<T, std::void_t<decltype(++std::declval<T &>())>>
    : std::true_type {};

template <typename VoidT, as::address_space Space, as::decorated Decorated>
void checkVoidRestrictions() {
  using void_ptr_t = sycl::multi_ptr<VoidT, Space, Decorated>;

  static_assert(void_ptr_t::address_space == Space);
  static_assert(std::is_same_v<typename void_ptr_t::value_type, VoidT>);
  static_assert(
      std::is_same_v<sycl::remove_decoration_t<typename void_ptr_t::pointer>,
                     std::add_pointer_t<VoidT>>);

  static_assert(!HasReference<void_ptr_t>::value);
  static_assert(!HasDereference<void_ptr_t>::value);
  static_assert(!HasPrefetch<void_ptr_t>::value);
  static_assert(!HasIncrement<void_ptr_t>::value);
}

template <typename VoidT> void checkVoid() {
  checkVoidRestrictions<VoidT, as::address_space::global_space,
                        as::decorated::yes>();
  checkVoidRestrictions<VoidT, as::address_space::global_space,
                        as::decorated::no>();
  checkVoidRestrictions<VoidT, as::address_space::local_space,
                        as::decorated::yes>();
  checkVoidRestrictions<VoidT, as::address_space::generic_space,
                        as::decorated::no>();
}

template void checkVoid<void>();
template void checkVoid<const void>();

// A non-void multi_ptr does provide all of them.
using int_ptr_t =
    sycl::multi_ptr<int, as::address_space::global_space, as::decorated::no>;
static_assert(HasReference<int_ptr_t>::value);
static_assert(HasDereference<int_ptr_t>::value);
static_assert(HasIncrement<int_ptr_t>::value);

// prefetch is only available in the global address space.
static_assert(HasPrefetch<int_ptr_t>::value);
static_assert(!HasPrefetch<sycl::multi_ptr<int, as::address_space::local_space,
                                           as::decorated::no>>::value);
static_assert(
    !HasPrefetch<sycl::multi_ptr<int, as::address_space::private_space,
                                 as::decorated::no>>::value);
