//===-- Compile time compiler detection -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SUPPORT_MACROS_COMPILER_FEATURES_H
#define LLVM_LIBC_SUPPORT_MACROS_COMPILER_FEATURES_H

#if defined(__clang__)
#define LIBC_COMPILER_IS_CLANG
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define LIBC_COMPILER_IS_GCC
#endif

#if defined(_MSC_VER)
#define LIBC_COMPILER_IS_MSC
#endif

// Compiler builtin-detection.
// clang.llvm.org/docs/LanguageExtensions.html#has-builtin
#if defined(LIBC_COMPILER_IS_CLANG) ||                                       \
    (defined(LIBC_COMPILER_IS_GCC) && (__GNUC__ >= 10))
#define LLVM_LIBC_HAS_BUILTIN(BUILTIN) __has_builtin(BUILTIN)
#else
#define LLVM_LIBC_HAS_BUILTIN(BUILTIN) 0
#endif

// Compiler feature-detection.
// clang.llvm.org/docs/LanguageExtensions.html#has-feature-and-has-extension
#if defined(LIBC_COMPILER_IS_CLANG)
#define LLVM_LIBC_HAS_FEATURE(FEATURE) __has_feature(FEATURE)
#else
#define LLVM_LIBC_HAS_FEATURE(FEATURE) 0
#endif

#if defined(LIBC_COMPILER_IS_CLANG)
#define LLVM_LIBC_LOOP_NOUNROLL _Pragma("nounroll")
#elif defined(LIBC_COMPILER_IS_GCC)
#define LLVM_LIBC_LOOP_NOUNROLL _Pragma("GCC unroll 0")
#else
#define LLVM_LIBC_LOOP_NOUNROLL
#endif

#endif // LLVM_LIBC_SUPPORT_MACROS_COMPILER_FEATURES_H
