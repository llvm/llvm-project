//===-- Compile time compiler detection -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SUPPORT_COMPILER_FEATURES_H
#define LLVM_LIBC_SUPPORT_COMPILER_FEATURES_H

#if defined(__clang__)
#define LLVM_LIBC_COMPILER_CLANG
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define LLVM_LIBC_COMPILER_GCC
#endif

#if defined(_MSC_VER)
#define LLVM_LIBC_COMPILER_MSC
#endif

// Compiler builtin-detection.
// clang.llvm.org/docs/LanguageExtensions.html#has-builtin
#if defined(LLVM_LIBC_COMPILER_CLANG) ||                                       \
    (defined(LLVM_LIBC_COMPILER_GCC) && (__GNUC__ >= 10))
#define LLVM_LIBC_HAS_BUILTIN(BUILTIN) __has_builtin(BUILTIN)
#else
#define LLVM_LIBC_HAS_BUILTIN(BUILTIN) 0
#endif

// Compiler feature-detection.
// clang.llvm.org/docs/LanguageExtensions.html#has-feature-and-has-extension
#if defined(LLVM_LIBC_COMPILER_CLANG)
#define LLVM_LIBC_HAS_FEATURE(FEATURE) __has_feature(FEATURE)
#else
#define LLVM_LIBC_HAS_FEATURE(FEATURE) 0
#endif

#endif // LLVM_LIBC_SUPPORT_COMPILER_FEATURES_H
