// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <exception>

#include <cxxabi.h>

#ifndef _LIBCPPABI_VERSION
#  error this header can only be used with libc++abi
#endif

namespace std {

bool uncaught_exception() noexcept { return uncaught_exceptions() > 0; }

int uncaught_exceptions() noexcept {
#if _LIBCPPABI_VERSION > 1001
  return abi::__cxa_uncaught_exceptions();
#else
  return abi::__cxa_uncaught_exception() ? 1 : 0;
#endif
}

} // namespace std
