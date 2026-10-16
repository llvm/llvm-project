//===-- lib/runtime/stl-overrides.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file is inserted implicitly to all translation units using -include on
// the command line.  The reason is that it configures the C++ standard
// template library (libc++ or libstdc++) using preprocessor macro definitions
// that must appear before any C++ library include.

// We define our own _GLIBCXX_THROW_OR_ABORT here because, as of GCC 15.1, the
// libstdc++ header file <bits/c++config> uses (void)_EXC in its definition of
// _GLIBCXX_THROW_OR_ABORT to silence a warning.
//
// This is a problem for us because some compilers, specifically clang, do not
// always optimize away that (void)_EXC even though it is unreachable since it
// occurs after a call to _builtin_abort().  Because _EXC is typically an
// object derived from std::exception, (void)_EXC, when not optimized away,
// calls std::exception methods defined in the libstdc++ shared library.  We
// shouldn't link against that library since our build version may conflict
// with the version used by a hybrid Fortran/C++ application.
//
// Redefining _GLIBCXX_THROW_OR_ABORT in this manner is not supported by the
// maintainers of libstdc++, so future changes to libstdc++ may require future
// changes to this build script and/or future changes to the Fortran runtime
// source code.
#define _GLIBCXX_THROW_OR_ABORT(_EXC) (__builtin_abort())

// Declare function that is used in place of `std::__libcpp_verbose_abort` to
// avoid dependency on the symbol provided by libc++.
#ifndef _LIBCPP_VERBOSE_ABORT
#define _LIBCPP_VERBOSE_ABORT(...) flang_rt_verbose_abort(__VA_ARGS__)
void flang_rt_verbose_abort(char const *format, ...)
    __attribute__((format(printf, 1, 2)));
#endif
