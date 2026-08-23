//===-- sanitizer_ptrauth.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_PTRAUTH_H
#define SANITIZER_PTRAUTH_H

#if __has_feature(ptrauth_intrinsics)
#  include <ptrauth.h>
#elif defined(__aarch64__) && !defined(__APPLE__)
// On the stack the link register is protected with a Pointer Authentication
// Code when the code that spilled it was compiled with -mbranch-protection.
// That is a property of the code being unwound, not of this runtime, so it
// cannot be tested for here (__ARM_FEATURE_PAC_DEFAULT would describe the
// wrong binary). Strip unconditionally instead: xpaclri is in the NOP space,
// so it does nothing where pointer authentication is not enabled or not
// available, and it is the identity on an unsigned canonical address.
#  define ptrauth_strip(__value, __key) \
    ({                                  \
      __typeof(__value) ret;            \
      asm volatile(                     \
          "mov x30, %1\n\t"             \
          "hint #7\n\t"                 \
          "mov %0, x30\n\t"             \
          "mov x30, xzr\n\t"            \
          : "=r"(ret)                   \
          : "r"(__value)                \
          : "x30");                     \
      ret;                              \
    })
#  define ptrauth_auth_data(__value, __old_key, __old_data) __value
#  define ptrauth_string_discriminator(__string) ((int)0)
#else
// Copied from <ptrauth.h>
#  define ptrauth_strip(__value, __key) __value
#  define ptrauth_auth_data(__value, __old_key, __old_data) __value
#  define ptrauth_string_discriminator(__string) ((int)0)
#endif

#define STRIP_PAC_PC(pc) ((uptr)ptrauth_strip(pc, 0))

#endif // SANITIZER_PTRAUTH_H
