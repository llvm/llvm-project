//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of off_t type.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_OFF_T_H
#define LLVM_LIBC_TYPES_OFF_T_H

#if defined(__arm__) || defined(_M_ARM)
typedef __INT32_TYPE__ off_t;
#else
typedef __INT64_TYPE__ off_t;
#endif

#endif // LLVM_LIBC_TYPES_OFF_T_H
