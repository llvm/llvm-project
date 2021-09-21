// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_USES_ALLOCATOR_UTILS_H
#define _LIBCPP___MEMORY_USES_ALLOCATOR_UTILS_H

#include <__config>
#include <__memory/uses_allocator.h>
#include <__memory/allocator_arg_t.h>
#include <__memory/construct_at.h>
#include <utility>
#include <type_traits>
#include <tuple>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCPP_STD_VER > 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_pair_specialization {
    static constexpr bool value = false;
};

template <class _Tp, class _Up>
struct __is_pair_specialization<pair<_Tp, _Up>> {
    static constexpr bool value = true;
};

template <class _Tp, class _Alloc, class... _Args>
    requires (!__is_pair_specialization<_Tp>::value)
constexpr auto uses_allocator_construction_args(const _Alloc& __alloc, _Args&&... __args) noexcept {
    if constexpr (uses_allocator_v<_Tp, _Alloc>) {
        static_assert(is_constructible_v<_Tp, allocator_arg_t, const _Alloc&, _Args...> ||
                      is_constructible_v<_Tp, _Args..., const _Alloc&>);
        if constexpr (is_constructible_v<_Tp, allocator_arg_t, const _Alloc&, _Args...>) {
            return tuple<allocator_arg_t, const _Alloc&, _Args&&...>{allocator_arg, __alloc,
                                                                     _VSTD::forward<_Args>(__args)...};
        } else { // is_constructible_v<_Tp, _Args..., const _Alloc&>
            return forward_as_tuple(_VSTD::forward<_Args>(__args)..., __alloc);
        }
    } else {
        static_assert(is_constructible_v<_Tp, _Args...>);
        return forward_as_tuple(_VSTD::forward<_Args>(__args)...);
    }
}

template <class _Tp, class _Alloc, class _Tuple1, class _Tuple2>
    requires __is_pair_specialization<_Tp>::value
constexpr auto uses_allocator_construction_args(const _Alloc& __alloc, piecewise_construct_t, _Tuple1&& __x, _Tuple2&& __y) noexcept {
    return tuple{piecewise_construct,
                 apply([&__alloc](auto&&... __args1) {
                     return uses_allocator_construction_args<_Tp::first_type>(__alloc, _VSTD::forward<decltype(__args1)>(__args1)...);
                 }, _VSTD::forward<_Tuple1>(__x)),
                 apply([&__alloc](auto&&... __args2) {
                     return uses_allocator_construction_args<_Tp::second_type>(__alloc, _VSTD::forward<decltype(__args2)>(__args2)...);
                 }, _VSTD::forward<_Tuple2>(__y))};
}

template <class _Tp, class _Alloc>
    requires __is_pair_specialization<_Tp>::value
constexpr auto uses_allocator_construction_args(const _Alloc& __alloc) noexcept {
    return uses_allocator_construction_args<_Tp>(__alloc, piecewise_construct, tuple{}, tuple{});
}

template <class _Tp, class _Alloc, class _Up, class _Vp>
    requires __is_pair_specialization<_Tp>::value
constexpr auto uses_allocator_construction_args(const _Alloc& __alloc, _Up&& __u, _Vp&& __v) noexcept {
    return uses_allocator_construction_args<_Tp>(__alloc,
                                                 piecewise_construct,
                                                 forward_as_tuple(_VSTD::forward<_Up>(__u)),
                                                 forward_as_tuple(_VSTD::forward<_Vp>(__v)));
}

template <class _Tp, class _Alloc, class _Up, class _Vp>
    requires __is_pair_specialization<_Tp>::value
constexpr auto uses_allocator_construction_args(const _Alloc& __alloc, const pair<_Up, _Vp>& __pr) noexcept {
    return uses_allocator_construction_args<_Tp>(__alloc,
                                                 piecewise_construct,
                                                 forward_as_tuple(__pr.first),
                                                 forward_as_tuple(__pr.second));
}

template <class _Tp, class _Alloc, class _Up, class _Vp>
    requires __is_pair_specialization<_Tp>::value
constexpr auto uses_allocator_construction_args(const _Alloc& __alloc, pair<_Up, _Vp>&& __pr) noexcept {
    return uses_allocator_construction_args<_Tp>(__alloc,
                                                 piecewise_construct,
                                                 forward_as_tuple(get<0>(_VSTD::move(__pr))),
                                                 forward_as_tuple(get<1>(_VSTD::move(__pr))));
}

template <class _Tp, class _Alloc, class... _Args>
constexpr _Tp make_obj_using_allocator(const _Alloc& __alloc, _Args&&... __args) {
    return make_from_tuple<_Tp>(uses_allocator_construction_args<_Tp>(__alloc, _VSTD::forward<_Args>(__args)...));
}

template <class _Tp, class _Alloc, class... _Args>
constexpr _Tp* uninitialized_construct_using_allocator(_Tp* __p, const _Alloc& __alloc, _Args&&... __args) {
    return apply([&]<class... _Xs>(_Xs&&... __xs) {
        return construct_at(__p, _VSTD::forward<_Xs>(__xs)...);
    }, uses_allocator_construction_args<_Tp>(__alloc, _VSTD::forward<_Args>(__args)...));
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER > 20

#endif // _LIBCPP___MEMORY_USES_ALLOCATOR_UTILS_H
