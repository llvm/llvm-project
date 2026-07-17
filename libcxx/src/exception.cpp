//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _LIBCPP_ENABLE_CXX20_REMOVED_UNCAUGHT_EXCEPTION
#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <__config>

#if defined(_LIBCPP_ABI_MICROSOFT)
#  include "support/runtime/exception_msvc.ipp"
#  include "support/runtime/exception_pointer_msvc.ipp"
#elif defined(LIBCXX_BUILDING_LIBCXXABI)
#  include "support/runtime/exception_libcxxabi.ipp"
#  include "support/runtime/exception_pointer_cxxabi.ipp"
#elif defined(LIBCXXRT)
#  include "support/runtime/exception_libcxxrt.ipp"
#  include "support/runtime/exception_pointer_cxxabi.ipp"
#elif defined(__GLIBCXX__)
#  include "support/runtime/exception_glibcxx.ipp"
#  include "support/runtime/exception_pointer_glibcxx.ipp"
#else
#  include "include/atomic_support.h"
#  include "support/runtime/exception_fallback.ipp"
#  include "support/runtime/exception_pointer_unimplemented.ipp"
#endif
