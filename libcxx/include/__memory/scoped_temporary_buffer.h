// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_SCOPED_TEMPORARY_BUFFER_H
#define _LIBCPP___MEMORY_SCOPED_TEMPORARY_BUFFER_H

#include <__config>
#include <__memory/allocator.h>
#include <__type_traits/is_constant_evaluated.h>
#include <cstddef>
#include <new>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __temporary_allocation_result {
  _Tp* __ptr;
  ptrdiff_t __count;
};

template <class _Tp>
class __scoped_temporary_buffer {
public:
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 __scoped_temporary_buffer() _NOEXCEPT
      : __ptr_(NULL),
        __count_(0) {}

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 explicit __scoped_temporary_buffer(ptrdiff_t __count) _NOEXCEPT
      : __ptr_(NULL),
        __count_(0) {
    __try_allocate(__count);
  }

#if _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER)
  // pre: __buf_ptr points to the beginning of a previously allocated scoped temporary buffer or is null
  // notes: __count_ is ignored in non-constant evaluation
  _LIBCPP_HIDE_FROM_ABI explicit __scoped_temporary_buffer(_Tp* __buf_ptr) _NOEXCEPT : __ptr_(__buf_ptr), __count_(0) {}
#endif // _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER)

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 ~__scoped_temporary_buffer() _NOEXCEPT {
    if (__libcpp_is_constant_evaluated()) {
      allocator<_Tp>().deallocate(__ptr_, __count_);
    }

    std::__libcpp_deallocate_unsized((void*)__ptr_, _LIBCPP_ALIGNOF(_Tp));
  }

  __scoped_temporary_buffer(const __scoped_temporary_buffer&)            = delete;
  __scoped_temporary_buffer& operator=(const __scoped_temporary_buffer&) = delete;

  // pre: __ptr_ == nullptr && __count_ == 0
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __try_allocate(ptrdiff_t __count) _NOEXCEPT {
    if (__libcpp_is_constant_evaluated()) {
      __ptr_   = allocator<_Tp>().allocate(__count);
      __count_ = __count;
      return;
    }

    const ptrdiff_t __max_count =
        (~ptrdiff_t(0) ^ ptrdiff_t(ptrdiff_t(1) << (sizeof(ptrdiff_t) * __CHAR_BIT__ - 1))) / sizeof(_Tp);
    if (__count > __max_count)
      __count = __max_count;
    while (__count > 0) {
#if !defined(_LIBCPP_HAS_NO_ALIGNED_ALLOCATION)
      if (__is_overaligned_for_new(_LIBCPP_ALIGNOF(_Tp))) {
        align_val_t __al = align_val_t(_LIBCPP_ALIGNOF(_Tp));
        __ptr_           = static_cast<_Tp*>(::operator new(__count * sizeof(_Tp), __al, nothrow));
      } else {
        __ptr_ = static_cast<_Tp*>(::operator new(__count * sizeof(_Tp), nothrow));
      }
#else
      if (__is_overaligned_for_new(_LIBCPP_ALIGNOF(_Tp))) {
        // Since aligned operator new is unavailable, constructs an empty buffer rather than one with invalid alignment.
        return;
      }

      __ptr_ = static_cast<_Tp*>(::operator new(__count * sizeof(_Tp), nothrow));
#endif

      if (__ptr_) {
        __count_ = __count;
        break;
      }
      __count_ /= 2;
    }
  }

  _LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 __temporary_allocation_result<_Tp>
  __get() const _NOEXCEPT {
    __temporary_allocation_result<_Tp> __result = {__ptr_, __count_};
    return __result;
  }

#if _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER)
  _LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI __temporary_allocation_result<_Tp> __release_to_raw() _NOEXCEPT {
    __temporary_allocation_result<_Tp> __result = {__ptr_, __count_};

    __ptr_   = NULL;
    __count_ = 0;

    return __result;
  }
#endif // _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER)

private:
  _Tp* __ptr_;
  ptrdiff_t __count_;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_SCOPED_TEMPORARY_BUFFER_H
