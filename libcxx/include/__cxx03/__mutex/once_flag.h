//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___MUTEX_ONCE_FLAG_H
#define _LIBCPP___CXX03___MUTEX_ONCE_FLAG_H

#include <__cxx03/__config>
#include <__cxx03/__memory/addressof.h>
#include <__cxx03/__memory/shared_ptr.h> // __libcpp_acquire_load
#include <__cxx03/__tuple/tuple_indices.h>
#include <__cxx03/__tuple/tuple_size.h>
#include <__cxx03/__type_traits/invoke.h>
#include <__cxx03/__utility/forward.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

struct _LIBCPP_TEMPLATE_VIS once_flag;

template <class _Callable>
_LIBCPP_HIDE_FROM_ABI void call_once(once_flag&, _Callable&);

template <class _Callable>
_LIBCPP_HIDE_FROM_ABI void call_once(once_flag&, const _Callable&);

struct _LIBCPP_TEMPLATE_VIS once_flag {
  _LIBCPP_HIDE_FROM_ABI once_flag() _NOEXCEPT : __state_(_Unset) {}
  once_flag(const once_flag&)            = delete;
  once_flag& operator=(const once_flag&) = delete;

#if defined(_LIBCPP_ABI_MICROSOFT)
  typedef uintptr_t _State_type;
#else
  typedef unsigned long _State_type;
#endif

  static const _State_type _Unset    = 0;
  static const _State_type _Pending  = 1;
  static const _State_type _Complete = ~_State_type(0);

private:
  _State_type __state_;

  template <class _Callable>
  friend void call_once(once_flag&, _Callable&);

  template <class _Callable>
  friend void call_once(once_flag&, const _Callable&);
};

template <class _Fp>
class __call_once_param {
  _Fp& __f_;

public:
  _LIBCPP_HIDE_FROM_ABI explicit __call_once_param(_Fp& __f) : __f_(__f) {}

  _LIBCPP_HIDE_FROM_ABI void operator()() { __f_(); }
};

template <class _Fp>
void _LIBCPP_HIDE_FROM_ABI __call_once_proxy(void* __vp) {
  __call_once_param<_Fp>* __p = static_cast<__call_once_param<_Fp>*>(__vp);
  (*__p)();
}

_LIBCPP_EXPORTED_FROM_ABI void __call_once(volatile once_flag::_State_type&, void*, void (*)(void*));

template <class _Callable>
inline _LIBCPP_HIDE_FROM_ABI void call_once(once_flag& __flag, _Callable& __func) {
  if (__libcpp_acquire_load(&__flag.__state_) != once_flag::_Complete) {
    __call_once_param<_Callable> __p(__func);
    std::__call_once(__flag.__state_, std::addressof(__p), std::addressof(__call_once_proxy<_Callable>));
  }
}

template <class _Callable>
inline _LIBCPP_HIDE_FROM_ABI void call_once(once_flag& __flag, const _Callable& __func) {
  if (__libcpp_acquire_load(&__flag.__state_) != once_flag::_Complete) {
    __call_once_param<const _Callable> __p(__func);
    std::__call_once(__flag.__state_, std::addressof(__p), std::addressof(__call_once_proxy<const _Callable>));
  }
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___MUTEX_ONCE_FLAG_H
