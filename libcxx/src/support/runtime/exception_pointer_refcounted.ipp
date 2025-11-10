// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Provides the common functionality shared between cxxabi and glibcxx.

namespace std {

exception_ptr exception_ptr::__from_native_exception_pointer(void* __e) noexcept {
  exception_ptr ptr;
  ptr.__ptr_ = __e;
  __increment_refcount(ptr.__ptr_);

  return ptr;
}

exception_ptr::~exception_ptr() noexcept { __decrement_refcount(__ptr_); }

exception_ptr::exception_ptr(const exception_ptr& other) noexcept : __ptr_(other.__ptr_) {
  __increment_refcount(__ptr_);
}

exception_ptr& exception_ptr::operator=(const exception_ptr& other) noexcept {
  if (__ptr_ != other.__ptr_) {
    __increment_refcount(other.__ptr_);
    __decrement_refcount(__ptr_);
    __ptr_ = other.__ptr_;
  }
  return *this;
}

} // namespace std
