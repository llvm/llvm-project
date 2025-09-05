//===-- Definition of cfloat128 type --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_CFLOAT128_H
#define LLVM_LIBC_TYPES_CFLOAT128_H

#include "../llvm-libc-macros/cfloat128-macros.h"

#ifdef LIBC_TYPES_HAS_CFLOAT128
#ifndef LIBC_TYPES_CFLOAT128_IS_COMPLEX_LONG_DOUBLE
typedef _Complex __float128 cfloat128;
#else
typedef _Complex long double cfloat128;
#endif // LIBC_TYPES_CFLOAT128_IS_COMPLEX_LONG_DOUBLE
#endif // LIBC_TYPES_HAS_CFLOAT128

#endif // LLVM_LIBC_TYPES_CFLOAT128_H
