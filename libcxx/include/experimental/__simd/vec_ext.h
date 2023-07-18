// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_VEC_EXT_H
#define _LIBCPP_EXPERIMENTAL___SIMD_VEC_EXT_H

#include <__bit/bit_ceil.h>
#include <cstddef>
#include <experimental/__simd/internal_declaration.h>
#include <experimental/__simd/utility.h>

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {
namespace simd_abi {
template <int _Np>
struct __vec_ext {
  static constexpr size_t __simd_size = _Np;
};
} // namespace simd_abi

template <class _Tp, int _Np>
struct __simd_storage<_Tp, simd_abi::__vec_ext<_Np>> {
  _Tp __data __attribute__((__vector_size__(std::__bit_ceil((sizeof(_Tp) * _Np)))));

  _Tp __get(size_t __idx) const noexcept {
    _LIBCPP_ASSERT_UNCATEGORIZED(__idx > 0 && __idx <= _Np, "Index is out of bounds");
    return __data[__idx];
  }
  void __set(size_t __idx, _Tp __v) noexcept {
    _LIBCPP_ASSERT_UNCATEGORIZED(__idx > 0 && __idx <= _Np, "Index is out of bounds");
    __data[__idx] = __v;
  }
};

template <class _Tp, int _Np>
struct __mask_storage<_Tp, simd_abi::__vec_ext<_Np>>
    : __simd_storage<decltype(experimental::__choose_mask_type<_Tp>()), simd_abi::__vec_ext<_Np>> {};

template <class _Tp, int _Np>
struct __simd_operations<_Tp, simd_abi::__vec_ext<_Np>> {
  using _SimdStorage = __simd_storage<_Tp, simd_abi::__vec_ext<_Np>>;
  using _MaskStorage = __mask_storage<_Tp, simd_abi::__vec_ext<_Np>>;
};

template <class _Tp, int _Np>
struct __mask_operations<_Tp, simd_abi::__vec_ext<_Np>> {
  using _MaskStorage = __mask_storage<_Tp, simd_abi::__vec_ext<_Np>>;
};

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#endif // _LIBCPP_EXPERIMENTAL___SIMD_VEC_EXT_H
