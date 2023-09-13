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
#include <experimental/__simd/declaration.h>
#include <experimental/__simd/reference.h>
#include <experimental/__simd/scalar.h>
#include <experimental/__simd/vec_ext.h>

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {

// class template simd_mask [simd.mask.class]
// TODO: implement simd_mask class
template <class _Tp, class _Abi>
class simd_mask {
  using _Impl    = __mask_operations<_Tp, _Abi>;
  using _Storage = typename _Impl::_MaskStorage;

  _Storage __s_;

public:
  using value_type = bool;
  using reference  = __simd_reference<_Tp, _Storage, value_type>;
  using simd_type  = simd<_Tp, _Abi>;
  using abi_type   = _Abi;

  static _LIBCPP_HIDE_FROM_ABI constexpr size_t size() noexcept { return simd_type::size(); }

  // broadcast constructor
  _LIBCPP_HIDE_FROM_ABI explicit simd_mask(value_type __v) noexcept : __s_(_Impl::__broadcast(__v)) {}

  // scalar access [simd.mask.subscr]
  // Add operator[] temporarily to test braodcast. Add test for it in later patch.
  _LIBCPP_HIDE_FROM_ABI value_type operator[](size_t __i) const { return __s_.__get(__i); }
};

template <class _Tp>
using native_simd_mask = simd_mask<_Tp, simd_abi::native<_Tp>>;

template <class _Tp, int _Np>
using fixed_size_simd_mask = simd_mask<_Tp, simd_abi::fixed_size<_Np>>;

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#endif // _LIBCPP_EXPERIMENTAL___SIMD_SIMD_MASK_H
