// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__verbose_abort>

namespace std {

#ifdef _LIBCPP_BUILDING_LIBCXXABI
#  warning exception_ptr not yet implemented
#endif

_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE inline void exception_ptr::__increment_refcount(void* __ptr) _NOEXCEPT {
  if (__ptr)
    __libcpp_verbose_abort("exception_ptr not yet implemented\n");
}

_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE inline void exception_ptr::__decrement_refcount(void* __ptr) _NOEXCEPT {
  if (__ptr)
    __libcpp_verbose_abort("exception_ptr not yet implemented\n");
}

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr current_exception() _NOEXCEPT {
  __libcpp_verbose_abort("exception_ptr not yet implemented\n");
}

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE [[noreturn]] void rethrow_exception(exception_ptr p) {
  __libcpp_verbose_abort("exception_ptr not yet implemented\n");
}

} // namespace std
