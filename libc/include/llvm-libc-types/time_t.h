//===-- Definition of the type time_t -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_TIME_T_H__
#define __LLVM_LIBC_TYPES_TIME_T_H__

#if (defined(__arm__) || defined(_M_ARM))
typedef __INTPTR_TYPE__ time_t;
#else
typedef __INT64_TYPE__ time_t;
#endif

#endif // __LLVM_LIBC_TYPES_TIME_T_H__
