//===-- Compile time compiler detection -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_COMPILER_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_COMPILER_H

#if defined(__clang__)
#define LIBC_COMPILER_IS_CLANG
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define LIBC_COMPILER_IS_GCC
#endif

#if defined(_MSC_VER)
#define LIBC_COMPILER_IS_MSC
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_PROPERTIES_COMPILER_H
