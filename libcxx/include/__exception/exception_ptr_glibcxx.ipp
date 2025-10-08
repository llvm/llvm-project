// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace std {

// libsupc++ does not implement the dependent EH ABI and the functionality
// it uses to implement std::exception_ptr (which it declares as an alias of
// std::__exception_ptr::exception_ptr) is not directly exported to clients. So
// we have little choice but to hijack std::__exception_ptr::exception_ptr's
// (which fortunately has the same layout as our std::exception_ptr) copy
// constructor, assignment operator and destructor (which are part of its
// stable ABI), and its rethrow_exception(std::__exception_ptr::exception_ptr)
// function.
namespace __exception_ptr {

struct exception_ptr {
  void* __ptr_;

  void _M_addref() _GLIBCXX_USE_NOEXCEPT;
  void _M_release() _GLIBCXX_USE_NOEXCEPT;
};

} // namespace __exception_ptr

_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE inline void exception_ptr::__increment_refcount(void* __ptr) _NOEXCEPT {
  if (__ptr)
    reinterpret_cast<__exception_ptr::exception_ptr*>(this)->_M_addref();
}

_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE inline void exception_ptr::__decrement_refcount(void* __ptr) _NOEXCEPT {
  if (__ptr)
    reinterpret_cast<__exception_ptr::exception_ptr*>(this)->_M_release();
}

[[noreturn]] void rethrow_exception(__exception_ptr::exception_ptr);

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE [[noreturn]] void rethrow_exception(exception_ptr __ptr) {
  rethrow_exception(reinterpret_cast<__exception_ptr::exception_ptr&>(__ptr));
}

} // namespace std
