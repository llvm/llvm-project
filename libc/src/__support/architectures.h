//===-- Compile time architecture detection ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SUPPORT_ARCHITECTURES_H
#define LLVM_LIBC_SUPPORT_ARCHITECTURES_H

#if defined(__pnacl__) || defined(__CLR_VER)
#define LLVM_LIBC_ARCH_VM
#endif

#if (defined(_M_IX86) || defined(__i386__)) && !defined(LLVM_LIBC_ARCH_VM)
#define LLVM_LIBC_ARCH_X86_32
#endif

#if (defined(_M_X64) || defined(__x86_64__)) && !defined(LLVM_LIBC_ARCH_VM)
#define LLVM_LIBC_ARCH_X86_64
#endif

#if defined(LLVM_LIBC_ARCH_X86_32) || defined(LLVM_LIBC_ARCH_X86_64)
#define LLVM_LIBC_ARCH_X86
#endif

#if (defined(__arm__) || defined(_M_ARM))
#define LLVM_LIBC_ARCH_ARM
#endif

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#define LLVM_LIBC_ARCH_AARCH64
#endif

#if (defined(LLVM_LIBC_ARCH_AARCH64) || defined(LLVM_LIBC_ARCH_ARM))
#define LLVM_LIBC_ARCH_ANY_ARM
#endif

#if defined(LLVM_LIBC_ARCH_AARCH64)
#define LIBC_TARGET_HAS_FMA
#elif defined(LLVM_LIBC_ARCH_X86_64)
#if (defined(__AVX2__) || defined(__FMA__))
#define LIBC_TARGET_HAS_FMA
#endif
#endif

#if (defined(LLVM_LIBC_ARCH_X86_64) && defined(LIBC_TARGET_HAS_FMA))
#define INLINE_FMA __attribute__((target("fma")))
#else
#define INLINE_FMA
#endif // LLVM_LIBC_ARCH_X86_64

#endif // LLVM_LIBC_SUPPORT_ARCHITECTURES_H
