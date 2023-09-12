// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_INTERNAL_DECLARATION_H
#define _LIBCPP_EXPERIMENTAL___SIMD_INTERNAL_DECLARATION_H

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {

template <class _Tp, class _Abi>
struct __simd_storage;

template <class _Tp, class _Abi>
struct __mask_storage;

template <class _Tp, class _Abi>
struct __simd_operations;

template <class _Tp, class _Abi>
struct __mask_operations;

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#endif // _LIBCPP_EXPERIMENTAL___SIMD_INTERNAL_DECLARATION_H
