//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_UNINITIALIZED_BUFFER_H
#define _LIBCPP___MEMORY_UNINITIALIZED_BUFFER_H

#include <__config>
#include <__memory/construct_at.h>
#include <__memory/unique_ptr.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_default_constructible.h>
#include <__type_traits/remove_extent.h>
#include <__utility/move.h>
#include <cstddef>
#include <new>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// __make_uninitialized_buffer is a utility function to allocate some memory for scratch storage. The __destructor is
// called before deleting the memory, making it possible to destroy any leftover elements. The __destructor is called
// with the pointer to the first element and the total number of elements.

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Destructor>
class __uninitialized_buffer_deleter {
  size_t __count_;
  _Destructor __destructor_;

public:
  template <class _Dummy = int, __enable_if_t<is_default_constructible<_Destructor>::value, _Dummy> = 0>
  _LIBCPP_HIDE_FROM_ABI __uninitialized_buffer_deleter() : __count_(0) {}

  _LIBCPP_HIDE_FROM_ABI __uninitialized_buffer_deleter(size_t __count, _Destructor __destructor)
      : __count_(__count), __destructor_(std::move(__destructor)) {}

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI void operator()(_Tp* __ptr) {
    __destructor_(__ptr, __count_);
#ifdef _LIBCPP_HAS_NO_ALIGNED_ALLOCATION
    ::operator delete(__ptr);
#else
    ::operator delete(__ptr, __count_ * sizeof(_Tp), align_val_t(_LIBCPP_ALIGNOF(_Tp)));
#endif
  }
};

struct __noop {
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void operator()(_Args&&...) const {}
};

template <class _Array, class _Destructor = __noop>
using __uninitialized_buffer_t = unique_ptr<_Array, __uninitialized_buffer_deleter<_Destructor> >;

template <class _Array, class _Destructor = __noop>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_NO_CFI __uninitialized_buffer_t<_Array, _Destructor>
__make_uninitialized_buffer(nothrow_t, size_t __count, _Destructor __destructor = __noop()) {
  static_assert(is_array<_Array>::value, "");
  using _Tp = __remove_extent_t<_Array>;

#ifdef _LIBCPP_HAS_NO_ALIGNED_ALLOCATION
  _Tp* __ptr = static_cast<_Tp*>(::operator new(sizeof(_Tp) * __count, nothrow));
#else
  _Tp* __ptr = static_cast<_Tp*>(::operator new(sizeof(_Tp) * __count, align_val_t(_LIBCPP_ALIGNOF(_Tp)), nothrow));
#endif

  using _Deleter = __uninitialized_buffer_deleter<_Destructor>;
  return unique_ptr<_Array, _Deleter>(__ptr, _Deleter(__count, std::move(__destructor)));
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_UNINITIALIZED_BUFFER_H
