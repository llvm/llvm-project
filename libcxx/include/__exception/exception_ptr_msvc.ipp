// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_LIBCPP_CRT_FUNC void __cdecl __ExceptionPtrCreate(void*);
_LIBCPP_CRT_FUNC void __cdecl __ExceptionPtrDestroy(void*);
_LIBCPP_CRT_FUNC void __cdecl __ExceptionPtrCopy(void*, const void*);
_LIBCPP_CRT_FUNC void __cdecl __ExceptionPtrAssign(void*, const void*);
_LIBCPP_CRT_FUNC bool __cdecl __ExceptionPtrCompare(const void*, const void*);
_LIBCPP_CRT_FUNC bool __cdecl __ExceptionPtrToBool(const void*);
_LIBCPP_CRT_FUNC void __cdecl __ExceptionPtrSwap(void*, void*);
_LIBCPP_CRT_FUNC void __cdecl __ExceptionPtrCurrentException(void*);
[[noreturn]] _LIBCPP_CRT_FUNC void __cdecl __ExceptionPtrRethrow(const void*);
_LIBCPP_CRT_FUNC void __cdecl __ExceptionPtrCopyException(void*, const void*, const void*);

namespace std {

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr::exception_ptr() _NOEXCEPT { __ExceptionPtrCreate(this); }
_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr::exception_ptr(nullptr_t) _NOEXCEPT { __ExceptionPtrCreate(this); }

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr::exception_ptr(const exception_ptr& __other) _NOEXCEPT { __ExceptionPtrCopy(this, &__other); }
_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr& exception_ptr::operator=(const exception_ptr& __other) _NOEXCEPT {
  __ExceptionPtrAssign(this, &__other);
  return *this;
}

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr& exception_ptr::operator=(nullptr_t) _NOEXCEPT {
  exception_ptr __dummy;
  __ExceptionPtrAssign(this, &__dummy);
  return *this;
}

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr::~exception_ptr() _NOEXCEPT { __ExceptionPtrDestroy(this); }

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr::operator bool() const _NOEXCEPT { return __ExceptionPtrToBool(this); }

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE bool operator==(const exception_ptr& __x, const exception_ptr& __y) _NOEXCEPT {
  return __ExceptionPtrCompare(&__x, &__y);
}

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE void swap(exception_ptr& lhs, exception_ptr& rhs) _NOEXCEPT { __ExceptionPtrSwap(&rhs, &lhs); }

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr __copy_exception_ptr(void* __except, const void* __ptr) {
  exception_ptr __ret = nullptr;
  if (__ptr)
    __ExceptionPtrCopyException(&__ret, __except, __ptr);
  return __ret;
}

_LIBCPP_EXPORTED_FROM_LIB_INLINEABLE exception_ptr current_exception() _NOEXCEPT {
  exception_ptr __ret;
  __ExceptionPtrCurrentException(&__ret);
  return __ret;
}

[[__noreturn__]] _LIBCPP_EXPORTED_FROM_LIB_INLINEABLE void rethrow_exception(exception_ptr __ptr) { __ExceptionPtrRethrow(&__ptr); }

} // namespace std
