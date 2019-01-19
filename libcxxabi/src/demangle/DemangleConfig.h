//===--- Compiler.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file is contains a subset of macros copied from
// llvm/lib/Demangle/Compiler.h.
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_DEMANGLE_COMPILER_H
#define LIBCXX_DEMANGLE_COMPILER_H

#include "__config"

#ifdef _MSC_VER
// snprintf is implemented in VS 2015
#if _MSC_VER < 1900
#define snprintf _snprintf_s
#endif
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#ifndef NDEBUG
#if __has_attribute(noinline) && __has_attribute(used)
#define DEMANGLE_DUMP_METHOD __attribute__((noinline, used))
#else
#define DEMANGLE_DUMP_METHOD
#endif
#endif

#define DEMANGLE_FALLTHROUGH _LIBCPP_FALLTHROUGH()
#define DEMANGLE_UNREACHABLE _LIBCPP_UNREACHABLE()

#define DEMANGLE_NAMESPACE_BEGIN namespace { namespace itanium_demangle {
#define DEMANGLE_NAMESPACE_END } }

#endif
