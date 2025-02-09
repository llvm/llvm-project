//===-- Definition of macros from endian.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_ENDIAN_MACROS_H
#define LLVM_LIBC_MACROS_ENDIAN_MACROS_H

#define LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
#define BIG_ENDIAN __ORDER_BIG_ENDIAN__
#define BYTE_ORDER __BYTE_ORDER__

#endif // LLVM_LIBC_MACROS_ENDIAN_MACROS_H
