//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_CONSTANT_WRAPPER_H
#define _LIBCPP___UTILITY_CONSTANT_WRAPPER_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/invoke.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/declval.h>
#include <__utility/forward.h>
#include <__utility/integer_sequence.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

template <class _Tp>
struct __cw_fixed_value {
  using __type _LIBCPP_NODEBUG = _Tp;
  _LIBCPP_HIDE_FROM_ABI constexpr __cw_fixed_value(__type __v) noexcept : __data(__v) {}
  _Tp __data;
};

template <class _Tp, size_t _Extent>
struct __cw_fixed_value<_Tp[_Extent]> {
  using __type _LIBCPP_NODEBUG = _Tp[_Extent];
  _Tp __data[_Extent];

  _LIBCPP_HIDE_FROM_ABI constexpr __cw_fixed_value(_Tp (&__arr)[_Extent]) noexcept
      : __cw_fixed_value(__arr, make_index_sequence<_Extent>{}) {}

private:
  template <size_t... _Idxs>
  _LIBCPP_HIDE_FROM_ABI constexpr __cw_fixed_value(_Tp (&__arr)[_Extent], index_sequence<_Idxs...>) noexcept
      : __data{__arr[_Idxs]...} {}
};

template <class _Tp, size_t _Extent>
__cw_fixed_value(_Tp (&)[_Extent]) -> __cw_fixed_value<_Tp[_Extent]>;

template <__cw_fixed_value _Xp,
#  ifdef _LIBCPP_COMPILER_GCC
          // gcc bug:  https://gcc.gnu.org/PR117392
          class = typename decltype(__cw_fixed_value(_Xp))::__type
#  else
          class = typename decltype(_Xp)::__type
#  endif
          >
struct constant_wrapper;

template <class _Tp>
concept __constexpr_param = requires { typename constant_wrapper<_Tp::value>; };

template <__cw_fixed_value _Xp>
constexpr auto cw = constant_wrapper<_Xp>{};

struct __cw_operators {
  // unary operators
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator+(_Tp) noexcept -> constant_wrapper<(+_Tp::value)> {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator-(_Tp) noexcept -> constant_wrapper<(-_Tp::value)> {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator~(_Tp) noexcept -> constant_wrapper<(~_Tp::value)> {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator!(_Tp) noexcept -> constant_wrapper<(!_Tp::value)> {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator&(_Tp) noexcept -> constant_wrapper<(&_Tp::value)> {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator*(_Tp) noexcept -> constant_wrapper<(*_Tp::value)> {
    return {};
  }

  // binary operators
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator+(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value + _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator-(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value - _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator*(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value * _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator/(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value / _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator%(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value % _Rp::value)> {
    return {};
  }

  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<<(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value << _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator>>(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value >> _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator&(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value & _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator|(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value | _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator^(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value ^ _Rp::value)> {
    return {};
  }

  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires(!is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>)
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator&&(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value && _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires(!is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>)
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator||(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value || _Rp::value)> {
    return {};
  }

  // comparisons
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<=>(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value <=> _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value < _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<=(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value <= _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator==(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value == _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator!=(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value != _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator>(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value > _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator>=(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value >= _Rp::value)> {
    return {};
  }

  template <__constexpr_param _Lp, __constexpr_param _Rp>
  friend auto operator,(_Lp, _Rp) = delete;

  template <__constexpr_param _Lp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator->*(_Lp, _Rp) noexcept
      -> constant_wrapper<(_Lp::value->*_Rp::value)> {
    return {};
  }

  // pseudo-mutators
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator++(this _Tp) noexcept -> constant_wrapper<(++_Tp::value)> {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator++(this _Tp, int) noexcept
      -> constant_wrapper<(_Tp::value++)> {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator--(this _Tp) noexcept -> constant_wrapper<(--_Tp::value)> {
    return {};
  }
  template <__constexpr_param _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator--(this _Tp, int) noexcept
      -> constant_wrapper<(_Tp::value--)> {
    return {};
  }

  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator+=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value += _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator-=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value -= _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator*=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value *= _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator/=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value /= _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator%=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value %= _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator&=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value &= _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator|=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value |= _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator^=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value ^= _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator<<=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value <<= _Rp::value)> {
    return {};
  }
  template <__constexpr_param _Tp, __constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator>>=(this _Tp, _Rp) noexcept
      -> constant_wrapper<(_Tp::value >>= _Rp::value)> {
    return {};
  }
};

template <const auto& _Callable, class... _Args>
concept __constexpr_callable = (__constexpr_param<remove_cvref_t<_Args>> && ...) && requires {
  typename constant_wrapper<std::invoke(_Callable, remove_cvref_t<_Args>::value...)>;
};

template <const auto& _Obj, class... _Args>
concept __constexpr_indexable = (__constexpr_param<remove_cvref_t<_Args>> && ...) && requires {
  typename constant_wrapper<_Obj[remove_cvref_t<_Args>::value...]>;
};

template <__cw_fixed_value _Xp, class>
struct constant_wrapper : __cw_operators {
  static constexpr const auto& value = _Xp.__data;
  using type                         = constant_wrapper;
  using value_type                   = decltype(_Xp)::__type;

  template <__constexpr_param _Rp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator=(_Rp) const noexcept
      -> constant_wrapper<(value = _Rp::value)> {
    return {};
  }

  _LIBCPP_HIDE_FROM_ABI constexpr operator decltype(value)() const noexcept { return value; }

  template <class... _Args>
    requires __constexpr_callable<value, _Args...>
  [[nodiscard]]
  _LIBCPP_HIDE_FROM_ABI static constexpr constant_wrapper<std::invoke(value, remove_cvref_t<_Args>::value...)>
  operator()(_Args&&...) noexcept {
    return {};
  }

  template <class... _Args>
    requires(!__constexpr_callable<value, _Args...> && is_invocable_v<const value_type&, _Args && ...>)
  _LIBCPP_HIDE_FROM_ABI static constexpr decltype(auto)
  operator()(_Args&&... __args) noexcept(noexcept(std::invoke(value, std::forward<_Args>(__args)...))) {
    return std::invoke(value, std::forward<_Args>(__args)...);
  }

  template <class... _Args>
    requires __constexpr_indexable<value, _Args...>
  [[nodiscard]]
  _LIBCPP_HIDE_FROM_ABI static constexpr constant_wrapper<value[remove_cvref_t<_Args>::value...]>
  operator[](_Args&&...) noexcept {
    return {};
  }

  template <class... _Args>
    requires(!__constexpr_indexable<value, _Args...> && requires { value[std::declval<_Args>()...]; })
  _LIBCPP_HIDE_FROM_ABI static constexpr decltype(auto)
  operator[](_Args&&... __args) noexcept(noexcept(value[std::forward<_Args>(__args)...])) {
    return value[std::forward<_Args>(__args)...];
  }
};

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_CONSTANT_WRAPPER_H
