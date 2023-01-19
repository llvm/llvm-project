//===-- Compile time architecture detection ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SUPPORT_ARCHITECTURES_H
#define LLVM_LIBC_SUPPORT_ARCHITECTURES_H

#if defined(__AMDGPU__)
#define LLVM_LIBC_ARCH_AMDGPU
#endif

#if defined(__NVPTX__)
#define LLVM_LIBC_ARCH_NVPTX
#endif

#if defined(LLVM_LIBC_ARCH_NVPTX) || defined(LLVM_LIBC_ARCH_AMDGPU)
#define LLVM_LIBC_ARCH_GPU
#endif

#if defined(__pnacl__) || defined(__CLR_VER) || defined(LLVM_LIBC_ARCH_GPU)
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

#endif // LLVM_LIBC_SUPPORT_ARCHITECTURES_H
