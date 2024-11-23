//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_RELOCATE_AT_H
#define _LIBCPP___MEMORY_RELOCATE_AT_H

#include <__memory/allocator_traits.h>
#include <__memory/construct_at.h>
#include <__memory/is_trivially_allocator_relocatable.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__utility/move.h>
#include <__utility/scope_guard.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __destroy_object {
  _LIBCPP_CONSTEXPR_SINCE_CXX20 void operator()() const { std::__destroy_at(__obj_); }
  _Tp* __obj_;
};

template <class _Alloc, class _Tp>
struct __allocator_destroy_object {
  _LIBCPP_CONSTEXPR_SINCE_CXX20 void operator()() const { allocator_traits<_Alloc>::destroy(__alloc_, __obj_); }
  _Alloc& __alloc_;
  _Tp* __obj_;
};

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp* __libcpp_builtin_trivially_relocate_at(_Tp* __source, _Tp* __dest) _NOEXCEPT {
  static_assert(__libcpp_is_trivially_relocatable<_Tp>::value, "");
  // Casting to void* to suppress clang complaining that this is technically UB.
  __builtin_memcpy(static_cast<void*>(__dest), __source, sizeof(_Tp));
  return __dest;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp* __libcpp_builtin_trivially_relocate_at(_Tp* __first, _Tp* __last, _Tp* __dest) _NOEXCEPT {
  static_assert(__libcpp_is_trivially_relocatable<_Tp>::value, "");
  // Casting to void* to suppress clang complaining that this is technically UB.
  __builtin_memmove(static_cast<void*>(__dest), __first, (__last - __first) * sizeof(_Tp));
  return __dest;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp* __relocate_at(_Tp* __dest, _Tp* __source) {
  if constexpr (__libcpp_is_trivially_relocatable<_Tp>::value) {
    if (!__libcpp_is_constant_evaluated()) {
      return std::__libcpp_builtin_trivially_relocate_at(__source, __dest);
    }
  }
  auto __guard = std::__make_scope_guard(__destroy_object<_Tp>{__source});
  return std::__construct_at(__dest, std::move(*__source));
}

template <class _Alloc, class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp*
__allocator_relocate_at(_Alloc& __alloc, _Tp* __dest, _Tp* __source) {
  if constexpr (__allocator_has_trivial_move_construct<_Alloc, _Tp>::value &&
                __allocator_has_trivial_destroy<_Alloc, _Tp>::value) {
    (void)__alloc; // ignore the allocator
    return std::__relocate_at(__dest, __source);
  } else {
    auto __guard = std::__make_scope_guard(__allocator_destroy_object<_Alloc, _Tp>{__alloc, __source});
    allocator_traits<_Alloc>::construct(__alloc, __dest, std::move(*__source));
    return __dest;
  }
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_RELOCATE_AT_H
