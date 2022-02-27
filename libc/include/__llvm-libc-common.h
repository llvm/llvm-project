//===-- Common definitions for LLVM-libc public header files --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC___COMMON_H
#define LLVM_LIBC___COMMON_H

#ifdef __cplusplus

#undef __BEGIN_C_DECLS
#define __BEGIN_C_DECLS extern "C" {

#undef __END_C_DECLS
#define __END_C_DECLS }

#undef _Noreturn
#define _Noreturn [[noreturn]]

#undef _Alignas
#define _Alignas alignas

#undef _Alignof
#define _Alignof alignof

#else // not __cplusplus

#undef __BEGIN_C_DECLS
#define __BEGIN_C_DECLS

#undef __END_C_DECLS
#define __END_C_DECLS

#undef __restrict
#define __restrict restrict // C99 and above support the restrict keyword.

#endif // __cplusplus

#endif // LLVM_LIBC___COMMON_H
