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

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI void* __libcpp_operator_new(_Args... __args) {
#if __has_builtin(__builtin_operator_new) && __has_builtin(__builtin_operator_delete)
  return __builtin_operator_new(__args...);
#else
  return ::operator new(__args...);
#endif
}

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI void __libcpp_operator_delete(_Args... __args) _NOEXCEPT {
#if __has_builtin(__builtin_operator_new) && __has_builtin(__builtin_operator_delete)
  __builtin_operator_delete(__args...);
#else
  ::operator delete(__args...);
#endif
}

inline _LIBCPP_HIDE_FROM_ABI void* __libcpp_allocate(size_t __size, size_t __align) {
#if _LIBCPP_HAS_ALIGNED_ALLOCATION
  if (__is_overaligned_for_new(__align)) {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return __libcpp_operator_new(__size, __align_val);
  }
#endif

  (void)__align;
  return __libcpp_operator_new(__size);
}

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI void __do_deallocate_handle_size(void* __ptr, size_t __size, _Args... __args) _NOEXCEPT {
#if !_LIBCPP_HAS_SIZED_DEALLOCATION
  (void)__size;
  return std::__libcpp_operator_delete(__ptr, __args...);
#else
  return std::__libcpp_operator_delete(__ptr, __size, __args...);
#endif
}

inline _LIBCPP_HIDE_FROM_ABI void __libcpp_deallocate(void* __ptr, size_t __size, size_t __align) _NOEXCEPT {
#if !_LIBCPP_HAS_ALIGNED_ALLOCATION
  (void)__align;
  return __do_deallocate_handle_size(__ptr, __size);
#else
  if (__is_overaligned_for_new(__align)) {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return __do_deallocate_handle_size(__ptr, __size, __align_val);
  } else {
    return __do_deallocate_handle_size(__ptr, __size);
  }
#endif
}

inline _LIBCPP_HIDE_FROM_ABI void __libcpp_deallocate_unsized(void* __ptr, size_t __align) _NOEXCEPT {
#if !_LIBCPP_HAS_ALIGNED_ALLOCATION
  (void)__align;
  return __libcpp_operator_delete(__ptr);
#else
  if (__is_overaligned_for_new(__align)) {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return __libcpp_operator_delete(__ptr, __align_val);
  } else {
    return __libcpp_operator_delete(__ptr);
  }
#endif
}
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___NEW_ALLOCATE_H
