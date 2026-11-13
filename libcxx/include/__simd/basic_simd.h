//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___SIMD_BASIC_SIMD_H
#define _LIBCPP___SIMD_BASIC_SIMD_H

#include <__assert>
#include <__concepts/convertible_to.h>
#include <__config>
#include <__memory/assume_aligned.h>
#include <__ranges/concepts.h>
#include <__simd/abi.h>
#include <__simd/basic_simd_mask.h>
#include <__simd/simd_flags.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/pack_utils.h>
#include <__type_traits/remove_const.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/integer_sequence.h>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

namespace datapar {

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wpsabi")
template <class _Tp, class _Abi = __native_abi<_Tp>>
class basic_simd {
public:
  using value_type = _Tp;
  using mask_type  = basic_simd_mask<sizeof(_Tp), _Abi>;
  using abi_type   = _Abi;

private:
  using __data_t = abi_type::_SimdT;

  __data_t __data_;

  _LIBCPP_ALWAYS_INLINE static constexpr __data_t __broadcast(value_type __value) {
    return [&]<size_t... _Indices>(index_sequence<_Indices...>) _LIBCPP_ALWAYS_INLINE noexcept {
      return __data_t{((void)_Indices, __value)...};
    }(make_index_sequence<size()>{});
  }

  template <class _Up>
  _LIBCPP_ALWAYS_INLINE static constexpr __data_t __load_from_pointer(const _Up* __ptr) {
    return [&]<size_t... _Indices>(index_sequence<_Indices...>) _LIBCPP_ALWAYS_INLINE noexcept {
      return __data_t{__ptr[_Indices]...};
    }(make_index_sequence<size()>{});
  }

public:
  static constexpr integral_constant<__simd_size_type, __simd_size_v<value_type, abi_type>> size{};

  constexpr basic_simd() noexcept = default;

  // [simd.ctor]
  template <convertible_to<value_type> _Up, class _From = remove_cvref_t<_Up>>
    requires(__value_preserving_convertible<_From, value_type> ||
             (!is_arithmetic_v<_From> && !__constexpr_wrapper_like<_From>) ||
             (__constexpr_wrapper_like<_From> && is_arithmetic_v<remove_const_t<decltype(_From::value)>> &&
              bool_constant<(static_cast<value_type>(_From::value) == _From::value)>::value))
  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd(_Up&& __value) noexcept : __data_{__broadcast(__value)} {}

  // TODO: converting constructor
  // TODO: generator constructor
  // TODO: flag constructor
  // TODO: mask flag constructortrue

  template <ranges::contiguous_range _Range, class... _Flags>
  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd(_Range&& __range, simd_flags<_Flags...> = {}) noexcept
    requires(ranges::size(__range) == size())
  {
    static_assert(__is_vectorizable_type_v<ranges::range_value_t<_Range>>, "Range has to be of a vectorizable type");
    static_assert(__contains_type_v<__type_list<_Flags...>, __convert_flag> ||
                      __value_preserving_convertible<ranges::range_value_t<_Range>, value_type>,
                  "implicit conversion is not value preserving - consider using std::datapar::simd_flag_convert");
    auto* __ptr = std::assume_aligned<__get_align_for<value_type, _Flags...>>(std::to_address(ranges::begin(__range)));
    __data_     = __load_from_pointer(__ptr);
  }

  template <ranges::contiguous_range _Range, class... _Flags>
  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd(
      _Range&& __range, const mask_type& __mask, simd_flags<_Flags...> = {}) noexcept
    requires(ranges::size(__range) == size())
  {
    static_assert(__is_vectorizable_type_v<ranges::range_value_t<_Range>>, "Range has to be of a vectorizable type");
    static_assert(__contains_type_v<__type_list<_Flags...>, __convert_flag> ||
                      __value_preserving_convertible<ranges::range_value_t<_Range>, value_type>,
                  "implicit conversion is not value preserving - consider using std::datapar::simd_flag_convert");
    auto* __ptr = std::assume_aligned<__get_align_for<value_type, _Flags...>>(std::to_address(ranges::begin(__range)));
    __data_     = abi_type::__select(__mask.__data_, __load_from_pointer(__ptr), __broadcast(0));
  }

  // libc++ extensions
  _LIBCPP_ALWAYS_INLINE constexpr explicit basic_simd(__data_t __data) noexcept : __data_(__data) {}

  // [simd.subscr]
  _LIBCPP_HIDE_FROM_ABI constexpr value_type operator[](__simd_size_type __index) const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(__index >= 0 && __index < size(), "simd::operator[] out of bounds");
    return __data_[__index];
  }

  // [simd.unary]

  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd& operator++() noexcept
    requires requires(value_type __v) { ++__v; }
  {
    __data_ += 1;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd operator++(int) noexcept
    requires requires(value_type __v) { __v++; }
  {
    auto __ret = *this;
    ++*this;
    return __ret;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd& operator--() noexcept
    requires requires(value_type __v) { --__v; }
  {
    __data_ -= 1;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd operator--(int) noexcept
    requires requires(value_type __v) { __v--; }
  {
    auto __ret = *this;
    --*this;
    return __ret;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr mask_type operator!() const noexcept
    requires requires(value_type __v) { !__v; }
  {
    return mask_type(!__data_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd operator~() const noexcept
    requires requires(value_type __v) { ~__v; }
  {
    return basic_simd(~__data_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd operator+() const noexcept
    requires requires(value_type __v) { +__v; }
  {
    return basic_simd(+__data_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_simd operator-() const noexcept
    requires requires(value_type __v) { -__v; }
  {
    return basic_simd(-__data_);
  }

  // [simd.binary]

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd operator+(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v + __v; }
  {
    return basic_simd(__lhs.__data_ + __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd operator-(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v - __v; }
  {
    return basic_simd(__lhs.__data_ - __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd operator*(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v * __v; }
  {
    return basic_simd(__lhs.__data_ * __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd operator/(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v / __v; }
  {
    return basic_simd(__lhs.__data_ / __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd operator%(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v % __v; }
  {
    return basic_simd(__lhs.__data_ % __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd operator&(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v & __v; }
  {
    return basic_simd(__lhs.__data_ & __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd operator|(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v | __v; }
  {
    return basic_simd(__lhs.__data_ | __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd operator^(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v ^ __v; }
  {
    return basic_simd(__lhs.__data_ ^ __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd
  operator<<(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v << __v; }
  {
    return basic_simd(__lhs.__data_ << __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd
  operator>>(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v << __v; }
  {
    return basic_simd(__lhs.__data_ >> __rhs.__data_);
  }

  // [simd.cassign]

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator+=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v += __v; }
  {
    __lhs.__data_ = __lhs.__data_ + __rhs.__data_;
    return __lhs;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator-=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v -= __v; }
  {
    __lhs.__data_ = __lhs.__data_ - __rhs.__data_;
    return __lhs;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator*=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v *= __v; }
  {
    __lhs.__data_ = __lhs.__data_ * __rhs.__data_;
    return __lhs;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator/=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v /= __v; }
  {
    __lhs.__data_ = __lhs.__data_ / __rhs.__data_;
    return __lhs;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator%=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v %= __v; }
  {
    __lhs.__data_ = __lhs.__data_ % __rhs.__data_;
    return __lhs;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator&=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v &= __v; }
  {
    __lhs.__data_ = __lhs.__data_ & __rhs.__data_;
    return __lhs;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator|=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v |= __v; }
  {
    __lhs.__data_ = __lhs.__data_ | __rhs.__data_;
    return __lhs;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator^=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v ^= __v; }
  {
    __lhs.__data_ = __lhs.__data_ ^ __rhs.__data_;
    return __lhs;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator<<=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v <<= __v; }
  {
    __lhs.__data_ = __lhs.__data_ << __rhs.__data_;
    return __lhs;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr basic_simd& operator>>=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v >>= __v; }
  {
    __lhs.__data_ = __lhs.__data_ >> __rhs.__data_;
    return __lhs;
  }

  // [simd.comparisons]
  _LIBCPP_HIDE_FROM_ABI friend constexpr mask_type operator==(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v == __v; }
  {
    return mask_type(__lhs.__data_ == __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr mask_type operator!=(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v != __v; }
  {
    return mask_type(!(__lhs.__data_ == __rhs.__data_));
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr mask_type operator<(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v < __v; }
  {
    return mask_type(__lhs.__data_ < __rhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr mask_type operator>=(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v >= __v; }
  {
    return mask_type(__rhs.__data_ <= __lhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr mask_type operator>(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v > __v; }
  {
    return mask_type(__rhs.__data_ < __lhs.__data_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr mask_type operator<=(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
    requires requires(value_type __v) { __v <= __v; }
  {
    return mask_type(__lhs.__data_ <= __rhs.__data_);
  }
};
_LIBCPP_DIAGNOSTIC_POP

template <class _Tp, __simd_size_type _Np = __simd_size_v<_Tp, __native_abi<_Tp>>>
using simd = basic_simd<_Tp, __deduce_abi_t<_Tp, _Np>>;

} // namespace datapar
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

#endif // _LIBCPP___SIMD_BASIC_SIMD_H
