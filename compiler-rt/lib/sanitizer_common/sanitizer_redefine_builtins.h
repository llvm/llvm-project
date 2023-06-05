//===-- sanitizer_redefine_builtins.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Redefine builtin functions to use internal versions. This is needed where
// compiler optimizations end up producing unwanted libcalls!
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_COMMON_NO_REDEFINE_BUILTINS
#ifndef SANITIZER_REDEFINE_BUILTINS_H
#define SANITIZER_REDEFINE_BUILTINS_H

// The asm hack only works with GCC and Clang.
#if !defined(_MSC_VER) || defined(__clang__)

asm("memcpy = __sanitizer_internal_memcpy");
asm("memmove = __sanitizer_internal_memmove");
asm("memset = __sanitizer_internal_memset");

#endif  // !_MSC_VER || __clang__

#endif  // SANITIZER_REDEFINE_BUILTINS_H
#endif  // SANITIZER_COMMON_NO_REDEFINE_BUILTINS
