//===-- Definition of size_t types ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_RSIZE_T_H
#define LLVM_LIBC_TYPES_RSIZE_T_H

#ifdef LIBC_HAS_ANNEX_K

typedef __SIZE_TYPE__ rsize_t;

#endif // LIBC_HAS_ANNEX_K

#endif // LLVM_LIBC_TYPES_RSIZE_T_H
