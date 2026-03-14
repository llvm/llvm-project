//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Test that `std::pointer_traits` is empty when the pointer type has
// no element_type typedef. See http://wg21.link/LWG3545 for details.

#include <memory>
#include <type_traits>
#include <utility>

#include "test_macros.h"

template <typename... Ts>
struct VoidifyImpl {
  using type = void;
};

template <typename... Ts>
using Voidify = typename VoidifyImpl<Ts...>::type;

template <class T, class = void>
struct HasElementType : std::false_type {};

template <class T>
struct HasElementType<T, Voidify<typename std::pointer_traits<T>::element_type> > : std::true_type {};

template <class T, class = void>
struct HasPointerType : std::false_type {};

template <class T>
struct HasPointerType<T, Voidify<typename std::pointer_traits<T>::pointer> > : std::true_type {};

template <class T, class = void>
struct HasDifferenceType : std::false_type {};

template <class T>
struct HasDifferenceType<T, Voidify<typename std::pointer_traits<T>::difference_type> > : std::true_type {};

template <class T, class U, class = void>
struct HasRebind : std::false_type {};

#if TEST_STD_VER >= 11
template <class T, class U>
struct HasRebind<T, U, Voidify<typename std::pointer_traits<T>::template rebind<U> > > : std::true_type {};
#else
template <class T, class U>
struct HasRebind<T, U, Voidify<typename std::pointer_traits<T>::template rebind<U>::other> > : std::true_type {};
#endif

template <class T, class = void>
struct HasPointerTo : std::false_type {};

template <class T>
struct HasPointerTo<
    T,
    Voidify<decltype(std::pointer_traits<T>::pointer_to(
        std::declval<typename std::add_lvalue_reference<typename std::pointer_traits<T>::element_type>::type>()))> >
    : std::true_type {};

struct NotAPtr {};

static_assert(!HasElementType<NotAPtr>::value, "");
static_assert(!HasPointerType<NotAPtr>::value, "");
static_assert(!HasDifferenceType<NotAPtr>::value, "");
static_assert(!HasRebind<NotAPtr, long>::value, "");
static_assert(!HasPointerTo<NotAPtr>::value, "");
