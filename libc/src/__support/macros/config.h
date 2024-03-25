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

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_CONFIG_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_CONFIG_H

// Compiler feature-detection.
// clang.llvm.org/docs/LanguageExtensions.html#has-feature-and-has-extension
#ifdef __has_feature
#define LIBC_HAS_FEATURE(f) __has_feature(f)
#else
#define LIBC_HAS_FEATURE(f) 0
#endif

// Compiler attribute-detection.
// https://clang.llvm.org/docs/LanguageExtensions.html#has-attribute
#ifdef __has_attribute
#define LIBC_HAS_ATTRIBUTE(f) __has_attribute(f)
#else
#define LIBC_HAS_ATTRIBUTE(f) 0
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_CONFIG_H
