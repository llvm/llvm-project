//===-- Definition of macros for stdchdint.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_STDCHDINT_MACROS_H
#define __LLVM_LIBC_MACROS_STDCHDINT_MACROS_H

#ifdef __GNUC__

#ifndef __STDC_VERSION_STDCKDINT_H__
#define __STDC_VERSION_STDCKDINT_H__ 202311L

#define ckd_add(R, A, B) __builtin_add_overflow((A), (B), (R))
#define ckd_sub(R, A, B) __builtin_sub_overflow((A), (B), (R))
#define ckd_mul(R, A, B) __builtin_mul_overflow((A), (B), (R))
#endif // __STDC_VERSION_STDCKDINT_H__
#endif // __GNUC__
#endif // __LLVM_LIBC_MACROS_STDCHDINT_MACROS_H
