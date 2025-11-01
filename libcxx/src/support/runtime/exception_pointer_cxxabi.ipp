// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HAVE_DEPENDENT_EH_ABI
#  error this header may only be used with libc++abi or libcxxrt
#endif

namespace std {

void exception_ptr::__do_increment_refcount(void* __ptr) noexcept {
  __cxa_increment_exception_refcount(__ptr);
}

void exception_ptr::__do_decrement_refcount(void* __ptr) noexcept {
  __cxa_decrement_exception_refcount(__ptr);
}

exception_ptr exception_ptr::__from_native_exception_pointer(void* __e) noexcept {
  exception_ptr ptr;
  ptr.__ptr_ = __e;
  __cxa_increment_exception_refcount(ptr.__ptr_);

  return ptr;
}

nested_exception::nested_exception() noexcept : __ptr_(current_exception()) {}

nested_exception::~nested_exception() noexcept {}

void nested_exception::rethrow_nested() const {
  if (__ptr_ == nullptr)
    terminate();
  rethrow_exception(__ptr_);
}

exception_ptr current_exception() noexcept {
  // be nicer if there was a constructor that took a ptr, then
  // this whole function would be just:
  //    return exception_ptr(__cxa_current_primary_exception());
  exception_ptr ptr;
  ptr.__ptr_ = __cxa_current_primary_exception();
  return ptr;
}

void rethrow_exception(exception_ptr p) {
  __cxa_rethrow_primary_exception(p.__ptr_);
  // if p.__ptr_ is NULL, above returns so we terminate
  terminate();
}

} // namespace std
