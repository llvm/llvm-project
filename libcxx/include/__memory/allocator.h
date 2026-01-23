// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ALLOCATOR_H
#define _LIBCPP___MEMORY_ALLOCATOR_H

#include <__config>
#include <__cstddef/ptrdiff_t.h>
#include <__cstddef/size_t.h>
#include <__memory/addressof.h>
#include <__memory/allocator_traits.h>
#include <__new/allocate.h>
#include <__new/exceptions.h>
#include <__type_traits/is_const.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_void.h>
#include <__type_traits/is_volatile.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
class allocator;

#if _LIBCPP_STD_VER <= 17
// These specializations shouldn't be marked _LIBCPP_DEPRECATED_IN_CXX17.
// Specializing allocator<void> is deprecated, but not using it.
template <>
class allocator<void> {
public:
  _LIBCPP_DEPRECATED_IN_CXX17 typedef void* pointer;
  _LIBCPP_DEPRECATED_IN_CXX17 typedef const void* const_pointer;
  _LIBCPP_DEPRECATED_IN_CXX17 typedef void value_type;

  template <class _Up>
  struct _LIBCPP_DEPRECATED_IN_CXX17 rebind {
    typedef allocator<_Up> other;
  };
};
#endif // _LIBCPP_STD_VER <= 17

template <bool, class _Unique>
struct __non_trivially_default_constructible_if {};

template <class _Unique>
struct __non_trivially_default_constructible_if<true, _Unique> {
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR __non_trivially_default_constructible_if() {}
};

template <class _Tp>
class allocator
// TODO(LLVM 24): Remove the opt-out
#ifdef _LIBCPP_DEPRECATED_ABI_NON_TRIVIAL_ALLOCATOR
    : __non_trivially_default_constructible_if<!is_void<_Tp>::value, allocator<_Tp> >
#endif
{
  static_assert(!is_const<_Tp>::value, "std::allocator does not support const types");
  static_assert(!is_volatile<_Tp>::value, "std::allocator does not support volatile types");

public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef _Tp value_type;
  typedef true_type propagate_on_container_move_assignment;
#if _LIBCPP_STD_VER <= 23 || defined(_LIBCPP_ENABLE_CXX26_REMOVED_ALLOCATOR_MEMBERS)
  _LIBCPP_DEPRECATED_IN_CXX23 typedef true_type is_always_equal;
#endif

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 allocator() _NOEXCEPT = default;

  template <class _Up>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 allocator(const allocator<_Up>&) _NOEXCEPT {}

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp* allocate(size_t __n) {
    static_assert(sizeof(_Tp) >= 0, "cannot allocate memory for an incomplete type");
    if (__n > allocator_traits<allocator>::max_size(*this))
      std::__throw_bad_array_new_length();
    if (__libcpp_is_constant_evaluated()) {
      return static_cast<_Tp*>(::operator new(__n * sizeof(_Tp)));
    } else {
      return std::__libcpp_allocate<_Tp>(__element_count(__n));
    }
  }

#if _LIBCPP_STD_VER >= 23
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr allocation_result<_Tp*> allocate_at_least(size_t __n) {
    static_assert(sizeof(_Tp) >= 0, "cannot allocate memory for an incomplete type");
    return {allocate(__n), __n};
  }
#endif

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void deallocate(_Tp* __p, size_t __n) _NOEXCEPT {
    if (__libcpp_is_constant_evaluated()) {
      ::operator delete(__p);
    } else {
      std::__libcpp_deallocate<_Tp>(__p, __element_count(__n));
    }
  }

  // C++20 Removed members
#if _LIBCPP_STD_VER <= 17
  _LIBCPP_DEPRECATED_IN_CXX17 typedef _Tp* pointer;
  _LIBCPP_DEPRECATED_IN_CXX17 typedef const _Tp* const_pointer;
  _LIBCPP_DEPRECATED_IN_CXX17 typedef _Tp& reference;
  _LIBCPP_DEPRECATED_IN_CXX17 typedef const _Tp& const_reference;

  template <class _Up>
  struct _LIBCPP_DEPRECATED_IN_CXX17 rebind {
    typedef allocator<_Up> other;
  };

  [[__nodiscard__]] _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_HIDE_FROM_ABI pointer address(reference __x) const _NOEXCEPT {
    return std::addressof(__x);
  }
  [[__nodiscard__]] _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_HIDE_FROM_ABI const_pointer
  address(const_reference __x) const _NOEXCEPT {
    return std::addressof(__x);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_DEPRECATED_IN_CXX17 _Tp* allocate(size_t __n, const void*) {
    return allocate(__n);
  }

  [[__nodiscard__]] _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_HIDE_FROM_ABI size_type max_size() const _NOEXCEPT {
    return size_type(~0) / sizeof(_Tp);
  }

  template <class _Up, class... _Args>
  _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_HIDE_FROM_ABI void construct(_Up* __p, _Args&&... __args) {
    ::new ((void*)__p) _Up(std::forward<_Args>(__args)...);
  }

  _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_HIDE_FROM_ABI void destroy(pointer __p) { __p->~_Tp(); }
#endif
};

template <class _Tp, class _Up>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 bool
operator==(const allocator<_Tp>&, const allocator<_Up>&) _NOEXCEPT {
  return true;
}

#if _LIBCPP_STD_VER <= 17

template <class _Tp, class _Up>
inline _LIBCPP_HIDE_FROM_ABI bool operator!=(const allocator<_Tp>&, const allocator<_Up>&) _NOEXCEPT {
  return false;
}

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_ALLOCATOR_H
