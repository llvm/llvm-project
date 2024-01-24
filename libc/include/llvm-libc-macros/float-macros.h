//===-- Definition of macros from float.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_FLOAT_MACROS_H
#define __LLVM_LIBC_MACROS_FLOAT_MACROS_H

#undef FLT_MANT_DIG
#define FLT_MANT_DIG __FLT_MANT_DIG__

#undef DBL_MANT_DIG
#define DBL_MANT_DIG __DBL_MANT_DIG__

#undef LDBL_MANT_DIG
#define LDBL_MANT_DIG __LDBL_MANT_DIG__

#endif // __LLVM_LIBC_MACROS_FLOAT_MACROS_H
