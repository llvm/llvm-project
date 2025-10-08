// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__exception/terminate.h>

namespace __cxxabiv1 {

extern "C" {
_LIBCPP_OVERRIDABLE_FUNC_VIS void __cxa_increment_exception_refcount(void*) noexcept;
_LIBCPP_OVERRIDABLE_FUNC_VIS void __cxa_decrement_exception_refcount(void*) noexcept;
_LIBCPP_OVERRIDABLE_FUNC_VIS void* __cxa_current_primary_exception() noexcept;
_LIBCPP_OVERRIDABLE_FUNC_VIS void __cxa_rethrow_primary_exception(void*);
}
  
} // namespace __cxxabiv1

namespace std {

_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE inline void exception_ptr::__increment_refcount(void* __ptr) _NOEXCEPT {
  if (__ptr)
    __cxxabiv1::__cxa_increment_exception_refcount(__ptr);
}

_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE inline void exception_ptr::__decrement_refcount(void* __ptr) _NOEXCEPT {
  if (__ptr)
    __cxxabiv1::__cxa_decrement_exception_refcount(__ptr);
}

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr current_exception() _NOEXCEPT {
  // It would be nicer if there was a constructor that took a ptr, then
  // this whole function would be just:
  //    return exception_ptr(__cxa_current_primary_exception());
  exception_ptr __ptr;
  __ptr.__ptr_ =  __cxxabiv1::__cxa_current_primary_exception();
  return __ptr;
}

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE void rethrow_exception(exception_ptr __ptr) {
  __cxxabiv1::__cxa_rethrow_primary_exception(__ptr.__ptr_);
  // if __ptr.__ptr_ is NULL, above returns so we terminate.
  terminate();
}

} // namespace std
