//===-- Definition of float16 type ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_FLOAT16_H
#define LLVM_LIBC_TYPES_FLOAT16_H

#include "src/__support/macros/properties/compiler.h"

#if defined(__FLT16_MANT_DIG__) &&                                             \
    (!defined(LIBC_COMPILER_IS_GCC) || LIBC_COMPILER_GCC_VER >= 1301)
#define LIBC_TYPES_HAS_FLOAT16
using float16 = _Float16;
#endif

#endif // LLVM_LIBC_TYPES_FLOAT16_H
