//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NEW_ALLOCATE_H
#define _LIBCPP___NEW_ALLOCATE_H

#include <__config>
#include <__cstddef/max_align_t.h>
#include <__cstddef/size_t.h>
#include <__new/align_val_t.h>
#include <__new/global_new_delete.h> // for _LIBCPP_HAS_SIZED_DEALLOCATION

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_CONSTEXPR inline _LIBCPP_HIDE_FROM_ABI bool __is_overaligned_for_new(size_t __align) _NOEXCEPT {
#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
  return __align > __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#else
  return __align > _LIBCPP_ALIGNOF(max_align_t);
#endif
}

template <class _Tp, class... _Args>
_LIBCPP_HIDE_FROM_ABI void* __libcpp_operator_new(_Args... __args) {
#if __has_builtin(__builtin_operator_new) && __has_builtin(__builtin_operator_delete)
  return __builtin_operator_new(__args...);
#else
  return ::operator new(__args...);
#endif
}

template <class _Tp, class... _Args>
_LIBCPP_HIDE_FROM_ABI void __libcpp_operator_delete(_Args... __args) _NOEXCEPT {
#if __has_builtin(__builtin_operator_new) && __has_builtin(__builtin_operator_delete)
  __builtin_operator_delete(__args...);
#else
  ::operator delete(__args...);
#endif
}

template <class _Tp>
inline _LIBCPP_HIDE_FROM_ABI void* __libcpp_allocate(size_t __size, size_t __align) {
#if _LIBCPP_HAS_ALIGNED_ALLOCATION
  if (__is_overaligned_for_new(__align)) {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return std::__libcpp_operator_new<_Tp>(__size, __align_val);
  }
#endif

  (void)__align;
  return std::__libcpp_operator_new<_Tp>(__size);
}

#if _LIBCPP_HAS_SIZED_DEALLOCATION
#  define _LIBCPP_ONLY_IF_SIZED_DEALLOCATION(...) __VA_ARGS__
#else
#  define _LIBCPP_ONLY_IF_SIZED_DEALLOCATION(...) /* nothing */
#endif

template <class _Tp>
inline _LIBCPP_HIDE_FROM_ABI void __libcpp_deallocate(void* __ptr, size_t __size, size_t __align) _NOEXCEPT {
  (void)__size;
#if !_LIBCPP_HAS_ALIGNED_ALLOCATION
  (void)__align;
  return std::__libcpp_operator_delete<_Tp>(__ptr _LIBCPP_ONLY_IF_SIZED_DEALLOCATION(, __size));
#else
  if (__is_overaligned_for_new(__align)) {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return std::__libcpp_operator_delete<_Tp>(__ptr _LIBCPP_ONLY_IF_SIZED_DEALLOCATION(, __size), __align_val);
  } else {
    return std::__libcpp_operator_delete<_Tp>(__ptr _LIBCPP_ONLY_IF_SIZED_DEALLOCATION(, __size));
  }
#endif
}

template <class _Tp>
struct __deallocating_deleter {
  _LIBCPP_HIDE_FROM_ABI void operator()(void* __p) const {
    std::__libcpp_deallocate<_Tp>(__p, sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp));
  }
};

#undef _LIBCPP_ONLY_IF_SIZED_DEALLOCATION

template <class _Tp>
inline _LIBCPP_HIDE_FROM_ABI void __libcpp_deallocate_unsized(void* __ptr, size_t __align) _NOEXCEPT {
#if !_LIBCPP_HAS_ALIGNED_ALLOCATION
  (void)__align;
  return std::__libcpp_operator_delete<_Tp>(__ptr);
#else
  if (__is_overaligned_for_new(__align)) {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return std::__libcpp_operator_delete<_Tp>(__ptr, __align_val);
  } else {
    return std::__libcpp_operator_delete<_Tp>(__ptr);
  }
#endif
}
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___NEW_ALLOCATE_H
