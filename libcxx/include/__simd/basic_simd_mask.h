//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___SIMD_BASIC_SIMD_MASK_H
#define _LIBCPP___SIMD_BASIC_SIMD_MASK_H

#include <__assert>
#include <__config>
#include <__cstddef/size_t.h>
#include <__simd/abi.h>
#include <__utility/integer_sequence.h>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD
namespace datapar {

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wpsabi")

struct __from_data_tag {};
inline constexpr __from_data_tag __from_data;

template <size_t _Bytes, class _Abi = __native_abi<__integer_from<_Bytes>>>
class basic_simd_mask {
public:
  using value_type = bool;
  using abi_type   = _Abi;

  static constexpr integral_constant<__simd_size_type, __simd_size_v<__integer_from<_Bytes>, abi_type>> size{};

private:
  using __data_t = abi_type::_MaskT;
  __data_t __data_;

  _LIBCPP_ALWAYS_INLINE static constexpr __data_t __broadcast(value_type __value) {
    return [&]<size_t... _Indices>(index_sequence<_Indices...>) _LIBCPP_ALWAYS_INLINE {
      return __data_t{((void)_Indices, __value)...};
    }(make_index_sequence<size()>{});
  }

public:
  // [simd.mask.ctor]
  _LIBCPP_HIDE_FROM_ABI constexpr explicit basic_simd_mask(value_type __value) noexcept
      : __data_(__broadcast(__value)) {}

  // TODO: converting constructor

  // TODO: generating constructor

  // libc++ extension
  _LIBCPP_ALWAYS_INLINE constexpr explicit basic_simd_mask(__data_t __data) noexcept : __data_(__data) {}
  _LIBCPP_ALWAYS_INLINE constexpr explicit operator __data_t() noexcept { return __data_; }

  // [simd.mask.subscr]
  _LIBCPP_HIDE_FROM_ABI constexpr value_type operator[](__simd_size_type __i) const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(__i >= 0 && __i < size(), "simd_mask::operator[] out of bounds");
    return __data_[__i];
  }

  // TODO: [simd.mask.unary]

  // TODO: [simd.mask.conv]

  // TODO: [simd.mask.binary]

  // TODO: [simd.mask.cassign]

  // TODO: [simd.mask.comparison]

  // TODO: [simd.mask.cond]

  template <size_t _Bytes2, class _Abi2>
  friend constexpr bool none_of(const basic_simd_mask<_Bytes2, _Abi2>&) noexcept;

  template <size_t _Bytes2, class _Abi2>
  friend constexpr bool any_of(const basic_simd_mask<_Bytes2, _Abi2>&) noexcept;

  template <size_t _Bytes2, class _Abi2>
  friend constexpr bool all_of(const basic_simd_mask<_Bytes2, _Abi2>&) noexcept;

  template <size_t _Bytes2, class _Abi2>
  friend constexpr __simd_size_type reduce_count(const basic_simd_mask<_Bytes2, _Abi2>&) noexcept;

  template <size_t _Bytes2, class _Abi2>
  friend constexpr __simd_size_type reduce_min_index(const basic_simd_mask<_Bytes2, _Abi2>&) noexcept;

  template <size_t _Bytes2, class _Abi2>
  friend constexpr __simd_size_type reduce_max_index(const basic_simd_mask<_Bytes2, _Abi2>&) noexcept;
};

template <class _Tp, __simd_size_type _Np = __simd_size_v<_Tp, __native_abi<_Tp>>>
using simd_mask = basic_simd_mask<sizeof(_Tp), __deduce_abi_t<_Tp, _Np>>;

// [simd.mask.reductions]

template <size_t _Bytes, class _Abi>
_LIBCPP_HIDE_FROM_ABI constexpr bool none_of(const basic_simd_mask<_Bytes, _Abi>& __mask) noexcept {
  return !_Abi::__any_of(__mask.__data_);
}

template <size_t _Bytes, class _Abi>
_LIBCPP_HIDE_FROM_ABI constexpr bool any_of(const basic_simd_mask<_Bytes, _Abi>& __mask) noexcept {
  return _Abi::__any_of(__mask.__data_);
}

template <size_t _Bytes, class _Abi>
_LIBCPP_HIDE_FROM_ABI constexpr bool all_of(const basic_simd_mask<_Bytes, _Abi>& __mask) noexcept {
  return _Abi::__all_of(__mask.__data_);
}

template <size_t _Bytes, class _Abi>
_LIBCPP_HIDE_FROM_ABI constexpr __simd_size_type reduce_count(const basic_simd_mask<_Bytes, _Abi>& __mask) noexcept {
  return _Abi::__reduce_count(__mask.__data_);
}

template <size_t _Bytes, class _Abi>
_LIBCPP_HIDE_FROM_ABI constexpr __simd_size_type
reduce_min_index(const basic_simd_mask<_Bytes, _Abi>& __mask) noexcept {
  return _Abi::__reduce_min_index(__mask.__data_);
}

template <size_t _Bytes, class _Abi>
_LIBCPP_HIDE_FROM_ABI constexpr __simd_size_type
reduce_max_index(const basic_simd_mask<_Bytes, _Abi>& __mask) noexcept {
  return _Abi::__reduce_max_index(__mask.__data_);
}

_LIBCPP_DIAGNOSTIC_POP

} // namespace datapar
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

#endif // _LIBCPP___SIMD_BASIC_SIMD_MASK_H
