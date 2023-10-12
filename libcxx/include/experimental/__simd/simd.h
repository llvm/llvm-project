// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_SIMD_H
#define _LIBCPP_EXPERIMENTAL___SIMD_SIMD_H

#include <__type_traits/remove_cvref.h>
#include <experimental/__simd/abi_tag.h>
#include <experimental/__simd/declaration.h>
#include <experimental/__simd/reference.h>
#include <experimental/__simd/scalar.h>
#include <experimental/__simd/traits.h>
#include <experimental/__simd/vec_ext.h>

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {

// class template simd [simd.class]
// TODO: implement simd class
template <class _Tp, class _Abi>
class simd {
  using _Impl    = __simd_operations<_Tp, _Abi>;
  using _Storage = typename _Impl::_SimdStorage;

  _Storage __s_;

public:
  using value_type = _Tp;
  using reference  = __simd_reference<_Tp, _Storage, value_type>;
  using mask_type  = simd_mask<_Tp, _Abi>;
  using abi_type   = _Abi;

  static _LIBCPP_HIDE_FROM_ABI constexpr size_t size() noexcept { return simd_size_v<value_type, abi_type>; }

  // broadcast constructor
  template <class _Up, enable_if_t<__can_broadcast_v<value_type, __remove_cvref_t<_Up>>, int> = 0>
  _LIBCPP_HIDE_FROM_ABI simd(_Up&& __v) noexcept : __s_(_Impl::__broadcast(static_cast<value_type>(__v))) {}

  // generator constructor
  template <class _Generator, enable_if_t<__can_generate_v<value_type, _Generator, size()>, int> = 0>
  explicit _LIBCPP_HIDE_FROM_ABI simd(_Generator&& __g) : __s_(_Impl::__generate(std::forward<_Generator>(__g))) {}

  // scalar access [simd.subscr]
  // Add operator[] temporarily to test braodcast. Add test for it in later patch.
  _LIBCPP_HIDE_FROM_ABI value_type operator[](size_t __i) const { return __s_.__get(__i); }
};

template <class _Tp>
using native_simd = simd<_Tp, simd_abi::native<_Tp>>;

template <class _Tp, int _Np>
using fixed_size_simd = simd<_Tp, simd_abi::fixed_size<_Np>>;

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#endif // _LIBCPP_EXPERIMENTAL___SIMD_SIMD_H
