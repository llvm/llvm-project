//===-- lib/runtime/stl-overrides.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

// Provide function that is used in place of `std::__libcpp_verbose_abort` to
// avoid dependency on the symbol provided by libc++.
void flang_rt_verbose_abort(char const *format, ...) {
  va_list list;
  va_start(list, format);
  std::vfprintf(stderr, format, list);
  va_end(list);

  std::abort();
}
