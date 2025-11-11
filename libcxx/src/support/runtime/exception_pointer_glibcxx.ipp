// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// libsupc++ does not implement the dependent EH ABI and the functionality
// it uses to implement std::exception_ptr (which it declares as an alias of
// std::__exception_ptr::exception_ptr) is not directly exported to clients. So
// we have little choice but to hijack std::__exception_ptr::exception_ptr's
// _M_addref and _M_release and its rethrow_exception function. Fortunately,
// glibcxx's exception_ptr has the same layout as our exception_ptr and we can
// reinterpret_cast between the two.

namespace std {

namespace __exception_ptr {

struct exception_ptr {
  void* __ptr_;

  void _M_addref() noexcept;
  void _M_release() noexcept;
};

} // namespace __exception_ptr

[[noreturn]] void rethrow_exception(__exception_ptr::exception_ptr);

void exception_ptr::__increment_refcount([[__gnu__::__nonnull__]] _LIBCPP_NOESCAPE void* __ptr) noexcept {
  reinterpret_cast<__exception_ptr::exception_ptr*>(this)->_M_addref();
}

void exception_ptr::__decrement_refcount([[__gnu__::__nonnull__]] _LIBCPP_NOESCAPE void* __ptr) noexcept {
  reinterpret_cast<__exception_ptr::exception_ptr*>(this)->_M_release();
}

nested_exception::nested_exception() noexcept : __ptr_(current_exception()) {}

[[noreturn]] void nested_exception::rethrow_nested() const {
  if (__ptr_ == nullptr)
    terminate();
  rethrow_exception(__ptr_);
}

[[noreturn]] void rethrow_exception(exception_ptr p) {
  rethrow_exception(reinterpret_cast<__exception_ptr::exception_ptr&>(p));
}

} // namespace std
