// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_ALIGNED_TAG_H
#define _LIBCPP_EXPERIMENTAL___SIMD_ALIGNED_TAG_H

#include <__bit/bit_ceil.h>
#include <__memory/assume_aligned.h>
#include <cstdint>

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {
// memory alignment
struct element_aligned_tag {
  template <class _Tp, class _Up = typename _Tp::value_type>
  static constexpr size_t __alignment = alignof(_Up);

  template <class _Tp, class _Up>
  static _LIBCPP_HIDE_FROM_ABI constexpr _Up* __apply(_Up* __ptr) {
    return __ptr;
  }
};

struct vector_aligned_tag {
  template <class _Tp, class _Up = typename _Tp::value_type>
  static constexpr size_t __alignment = std::__bit_ceil(sizeof(_Up) * _Tp::size());

  template <class _Tp, class _Up>
  static _LIBCPP_HIDE_FROM_ABI constexpr _Up* __apply(_Up* __ptr) {
    return std::__assume_aligned<__alignment<_Tp, _Up>, _Up>(__ptr);
  }
};

template <size_t _Np>
struct overaligned_tag {
  template <class _Tp, class _Up = typename _Tp::value_type>
  static constexpr size_t __alignment = _Np;

  template <class _Tp, class _Up>
  static _LIBCPP_HIDE_FROM_ABI constexpr _Up* __apply(_Up* __ptr) {
    return std::__assume_aligned<__alignment<_Tp, _Up>, _Up>(__ptr);
  }
};

inline constexpr element_aligned_tag element_aligned{};

inline constexpr vector_aligned_tag vector_aligned{};

template <size_t _Np>
inline constexpr overaligned_tag<_Np> overaligned{};

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#endif // _LIBCPP_EXPERIMENTAL___SIMD_ALIGNED_TAG_H
