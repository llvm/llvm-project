//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _LIBCPP_ENABLE_CXX20_REMOVED_UNCAUGHT_EXCEPTION
#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS
#define _LIBCPP_EMIT_CODE_FOR_EXCEPTION_PTR

#include <exception>
#include <new>
#include <typeinfo>

#if defined(_LIBCPP_CXX_ABI_NONE)
#  include "include/atomic_support.h"
#  include "support/runtime/exception_fallback.ipp"
#elif defined(_LIBCPP_CXX_ABI_LIBCXXABI)
#  include "support/runtime/exception_libcxxabi.ipp"
#elif defined(_LIBCPP_CXX_ABI_LIBCXXRT)
#  include "support/runtime/exception_libcxxrt.ipp"
#elif defined(_LIBCPP_CXX_ABI_LIBSTDCXX) || defined(_LIBCPP_CXX_ABI_LIBSUPCXX)
#  include "support/runtime/exception_glibcxx.ipp"
#elif defined(_LIBCPP_CXX_ABI_VCRUNTIME)
#  include "support/runtime/exception_msvc.ipp"
#else
#  error "Unsupported C++ ABI library"
#endif

namespace std {

nested_exception::nested_exception() noexcept : __ptr_(current_exception()) {}

#if !defined(_LIBCPP_CXX_ABI_LIBSTDCXX) && !defined(_LIBCPP_CXX_ABI_LIBSUPCXX)
nested_exception::~nested_exception() noexcept {}
#endif

void nested_exception::rethrow_nested() const {
  if (__ptr_ == nullptr)
    terminate();
  rethrow_exception(__ptr_);
}

} // namespace std
