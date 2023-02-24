// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_SIMD_MASK_H
#define _LIBCPP_EXPERIMENTAL___SIMD_SIMD_MASK_H

#include <experimental/__simd/abi_tag.h>

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {

// class template simd_mask [simd.mask.class]
// TODO: implement simd_mask class
template <class _Tp, class _Abi = simd_abi::compatible<_Tp>>
class simd_mask;

template <class _Tp>
using native_simd_mask = simd_mask<_Tp, simd_abi::native<_Tp>>;

template <class _Tp, int _Np>
using fixed_size_simd_mask = simd_mask<_Tp, simd_abi::fixed_size<_Np>>;

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#endif // _LIBCPP_EXPERIMENTAL___SIMD_SIMD_MASK_H
