//===-- lib/runtime/io-api-minimal.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Declare function that is used in place of `std::__libcpp_verbose_abort` to
// avoid dependency on the symbol provided by libc++.

void flang_rt_verbose_abort(char const *format, ...);
