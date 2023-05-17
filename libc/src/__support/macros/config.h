//===-- Portable attributes -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header file defines a set of macros for checking the presence of
// important compiler and platform features. Such macros can be used to
// produce portable code by parameterizing compilation based on the presence or
// lack of a given feature.

#ifndef LLVM_LIBC_SUPPORT_MACROS_CONFIG_H
#define LLVM_LIBC_SUPPORT_MACROS_CONFIG_H

// LIBC_HAS_BUILTIN()
//
// Checks whether the compiler supports a Clang Feature Checking Macro, and if
// so, checks whether it supports the provided builtin function "x" where x
// is one of the functions noted in
// https://clang.llvm.org/docs/LanguageExtensions.html
//
// Note: Use this macro to avoid an extra level of #ifdef __has_builtin check.
// http://releases.llvm.org/3.3/tools/clang/docs/LanguageExtensions.html

// Compiler builtin-detection.
// clang.llvm.org/docs/LanguageExtensions.html#has-builtin
#ifdef __has_builtin
#define LIBC_HAS_BUILTIN(x) __has_builtin(x)
#else
#define LIBC_HAS_BUILTIN(x) 0
#endif

// Compiler feature-detection.
// clang.llvm.org/docs/LanguageExtensions.html#has-feature-and-has-extension
#ifdef __has_feature
#define LIBC_HAS_FEATURE(f) __has_feature(f)
#else
#define LIBC_HAS_FEATURE(f) 0
#endif

#endif // LLVM_LIBC_SUPPORT_MACROS_CONFIG_H
