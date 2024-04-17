//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKEND_FWD_H
#define _LIBCPP___PSTL_BACKEND_FWD_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <class _Backend, class _ExecutionPolicy>
struct __find_if;

template <class _Backend, class _ExecutionPolicy>
struct __find;

template <class _Backend, class _ExecutionPolicy>
struct __find_if_not;

template <class _Backend, class _ExecutionPolicy>
struct __any_of;

template <class _Backend, class _ExecutionPolicy>
struct __all_of;

template <class _Backend, class _ExecutionPolicy>
struct __none_of;

template <class _Backend, class _ExecutionPolicy>
struct __is_partitioned;

template <class _Backend, class _ExecutionPolicy>
struct __for_each;

template <class _Backend, class _ExecutionPolicy>
struct __for_each_n;

template <class _Backend, class _ExecutionPolicy>
struct __fill;

template <class _Backend, class _ExecutionPolicy>
struct __fill_n;

template <class _Backend, class _ExecutionPolicy>
struct __replace;

template <class _Backend, class _ExecutionPolicy>
struct __replace_if;

template <class _Backend, class _ExecutionPolicy>
struct __generate;

template <class _Backend, class _ExecutionPolicy>
struct __generate_n;

template <class _Backend, class _ExecutionPolicy>
struct __merge;

template <class _Backend, class _ExecutionPolicy>
struct __stable_sort;

template <class _Backend, class _ExecutionPolicy>
struct __sort;

template <class _Backend, class _ExecutionPolicy>
struct __transform;

template <class _Backend, class _ExecutionPolicy>
struct __transform_binary;

template <class _Backend, class _ExecutionPolicy>
struct __replace_copy_if;

template <class _Backend, class _ExecutionPolicy>
struct __replace_copy;

template <class _Backend, class _ExecutionPolicy>
struct __move;

template <class _Backend, class _ExecutionPolicy>
struct __copy;

template <class _Backend, class _ExecutionPolicy>
struct __copy_n;

template <class _Backend, class _ExecutionPolicy>
struct __rotate_copy;

template <class _Backend, class _ExecutionPolicy>
struct __transform_reduce;

template <class _Backend, class _ExecutionPolicy>
struct __transform_reduce_binary;

template <class _Backend, class _ExecutionPolicy>
struct __count_if;

template <class _Backend, class _ExecutionPolicy>
struct __count;

template <class _Backend, class _ExecutionPolicy>
struct __equal_3leg;

template <class _Backend, class _ExecutionPolicy>
struct __equal;

template <class _Backend, class _ExecutionPolicy>
struct __reduce;

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKEND_FWD_H
