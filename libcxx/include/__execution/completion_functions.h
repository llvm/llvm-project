//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___EXECUTION_COMPLETION_FUNCTIONS_H
#define _LIBCPP___EXECUTION_COMPLETION_FUNCTIONS_H

#include <__config>
#include <__type_traits/is_same.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_EXECUTION

_LIBCPP_BEGIN_NAMESPACE_STD

namespace execution {
namespace __detail {

// For deduced forwarding-reference arguments, rejects lvalues and const rvalues.
template <class _Tp>
concept __non_const_rvalue = !is_same_v<_Tp, const _Tp>;

template <class _Receiver, class... _Args>
concept __set_value_member = requires(_Receiver&& __rcvr, _Args&&... __args) {
  std::forward<_Receiver>(__rcvr).set_value(std::forward<_Args>(__args)...);
};

template <class _Receiver, class _Error>
concept __set_error_member = requires(_Receiver&& __rcvr, _Error&& __error) {
  std::forward<_Receiver>(__rcvr).set_error(std::forward<_Error>(__error));
};

template <class _Receiver>
concept __set_stopped_member = requires(_Receiver&& __rcvr) { std::forward<_Receiver>(__rcvr).set_stopped(); };

} // namespace __detail

// [exec.set.value]
struct set_value_t {
  template <class _Receiver, class... _Args>
    requires __detail::__non_const_rvalue<_Receiver> && __detail::__set_value_member<_Receiver, _Args...>
  _LIBCPP_HIDE_FROM_ABI constexpr void operator()(_Receiver&& __rcvr, _Args&&... __args) const noexcept {
    static_assert(noexcept(std::forward<_Receiver>(__rcvr).set_value(std::forward<_Args>(__args)...)),
                  "set_value must be noexcept");
    static_assert(is_same_v<decltype(std::forward<_Receiver>(__rcvr).set_value(std::forward<_Args>(__args)...)), void>,
                  "set_value must return void");
    std::forward<_Receiver>(__rcvr).set_value(std::forward<_Args>(__args)...);
  }
};

// [exec.set.error]
struct set_error_t {
  template <class _Receiver, class _Error>
    requires __detail::__non_const_rvalue<_Receiver> && __detail::__set_error_member<_Receiver, _Error>
  _LIBCPP_HIDE_FROM_ABI constexpr void operator()(_Receiver&& __rcvr, _Error&& __error) const noexcept {
    static_assert(noexcept(std::forward<_Receiver>(__rcvr).set_error(std::forward<_Error>(__error))),
                  "set_error must be noexcept");
    static_assert(is_same_v<decltype(std::forward<_Receiver>(__rcvr).set_error(std::forward<_Error>(__error))), void>,
                  "set_error must return void");
    std::forward<_Receiver>(__rcvr).set_error(std::forward<_Error>(__error));
  }
};

// [exec.set.stopped]
struct set_stopped_t {
  template <class _Receiver>
    requires __detail::__non_const_rvalue<_Receiver> && __detail::__set_stopped_member<_Receiver>
  _LIBCPP_HIDE_FROM_ABI constexpr void operator()(_Receiver&& __rcvr) const noexcept {
    static_assert(noexcept(std::forward<_Receiver>(__rcvr).set_stopped()), "set_stopped must be noexcept");
    static_assert(
        is_same_v<decltype(std::forward<_Receiver>(__rcvr).set_stopped()), void>, "set_stopped must return void");
    std::forward<_Receiver>(__rcvr).set_stopped();
  }
};

inline constexpr set_value_t set_value{};
inline constexpr set_error_t set_error{};
inline constexpr set_stopped_t set_stopped{};

} // namespace execution

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_EXECUTION

#endif // _LIBCPP___EXECUTION_COMPLETION_FUNCTIONS_H
