// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___MEMORY_ALLOCATOR_H
#define _LIBCPP___CXX03___MEMORY_ALLOCATOR_H

#include <__cxx03/__config>
#include <__cxx03/__memory/addressof.h>
#include <__cxx03/__memory/allocate_at_least.h>
#include <__cxx03/__memory/allocator_traits.h>
#include <__cxx03/__type_traits/is_const.h>
#include <__cxx03/__type_traits/is_constant_evaluated.h>
#include <__cxx03/__type_traits/is_same.h>
#include <__cxx03/__type_traits/is_void.h>
#include <__cxx03/__type_traits/is_volatile.h>
#include <__cxx03/__utility/forward.h>
#include <__cxx03/cstddef>
#include <__cxx03/new>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
class allocator;

template <>
class _LIBCPP_TEMPLATE_VIS allocator<void> {
public:
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  template <class _Up>
  struct rebind {
    typedef allocator<_Up> other;
  };
};

// TODO(LLVM 20): Remove the escape hatch
#ifdef _LIBCPP_ENABLE_REMOVED_ALLOCATOR_CONST
template <>
class _LIBCPP_TEMPLATE_VIS allocator<const void> {
public:
  typedef const void* pointer;
  typedef const void* const_pointer;
  typedef const void value_type;

  template <class _Up>
  struct rebind {
    typedef allocator<_Up> other;
  };
};
#endif // _LIBCPP_ENABLE_REMOVED_ALLOCATOR_CONST

// This class provides a non-trivial default constructor to the class that derives from it
// if the condition is satisfied.
//
// The second template parameter exists to allow giving a unique type to __non_trivial_if,
// which makes it possible to avoid breaking the ABI when making this a base class of an
// existing class. Without that, imagine we have classes D1 and D2, both of which used to
// have no base classes, but which now derive from __non_trivial_if. The layout of a class
// that inherits from both D1 and D2 will change because the two __non_trivial_if base
// classes are not allowed to share the same address.
//
// By making those __non_trivial_if base classes unique, we work around this problem and
// it is safe to start deriving from __non_trivial_if in existing classes.
template <bool _Cond, class _Unique>
struct __non_trivial_if {};

template <class _Unique>
struct __non_trivial_if<true, _Unique> {
  _LIBCPP_HIDE_FROM_ABI __non_trivial_if() _NOEXCEPT {}
};

// allocator
//
// Note: For ABI compatibility between C++20 and previous standards, we make
//       allocator<void> trivial in C++20.

template <class _Tp>
class _LIBCPP_TEMPLATE_VIS allocator : private __non_trivial_if<!is_void<_Tp>::value, allocator<_Tp> > {
  static_assert(!is_const<_Tp>::value, "std::allocator does not support const types");
  static_assert(!is_volatile<_Tp>::value, "std::allocator does not support volatile types");

public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef _Tp value_type;
  typedef true_type propagate_on_container_move_assignment;
  typedef true_type is_always_equal;

  _LIBCPP_HIDE_FROM_ABI allocator() _NOEXCEPT = default;

  template <class _Up>
  _LIBCPP_HIDE_FROM_ABI allocator(const allocator<_Up>&) _NOEXCEPT {}

  _LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _Tp* allocate(size_t __n) {
    if (__n > allocator_traits<allocator>::max_size(*this))
      __throw_bad_array_new_length();
    if (__libcpp_is_constant_evaluated()) {
      return static_cast<_Tp*>(::operator new(__n * sizeof(_Tp)));
    } else {
      return static_cast<_Tp*>(std::__libcpp_allocate(__n * sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp)));
    }
  }

  _LIBCPP_HIDE_FROM_ABI void deallocate(_Tp* __p, size_t __n) _NOEXCEPT {
    if (__libcpp_is_constant_evaluated()) {
      ::operator delete(__p);
    } else {
      std::__libcpp_deallocate((void*)__p, __n * sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp));
    }
  }

  // C++20 Removed members
  typedef _Tp* pointer;
  typedef const _Tp* const_pointer;
  typedef _Tp& reference;
  typedef const _Tp& const_reference;

  template <class _Up>
  struct rebind {
    typedef allocator<_Up> other;
  };

  _LIBCPP_HIDE_FROM_ABI pointer address(reference __x) const _NOEXCEPT { return std::addressof(__x); }
  _LIBCPP_HIDE_FROM_ABI const_pointer address(const_reference __x) const _NOEXCEPT { return std::addressof(__x); }

  _LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _Tp* allocate(size_t __n, const void*) { return allocate(__n); }

  _LIBCPP_HIDE_FROM_ABI size_type max_size() const _NOEXCEPT { return size_type(~0) / sizeof(_Tp); }

  template <class _Up, class... _Args>
  _LIBCPP_HIDE_FROM_ABI void construct(_Up* __p, _Args&&... __args) {
    ::new ((void*)__p) _Up(std::forward<_Args>(__args)...);
  }

  _LIBCPP_HIDE_FROM_ABI void destroy(pointer __p) { __p->~_Tp(); }
};

template <class _Tp, class _Up>
inline _LIBCPP_HIDE_FROM_ABI bool operator==(const allocator<_Tp>&, const allocator<_Up>&) _NOEXCEPT {
  return true;
}

template <class _Tp, class _Up>
inline _LIBCPP_HIDE_FROM_ABI bool operator!=(const allocator<_Tp>&, const allocator<_Up>&) _NOEXCEPT {
  return false;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___MEMORY_ALLOCATOR_H
