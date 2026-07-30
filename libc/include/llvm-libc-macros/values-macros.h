//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of macros from values.h.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_VALUES_MACROS_H
#define LLVM_LIBC_MACROS_VALUES_MACROS_H

#include "float-macros.h"
#include "limits-macros.h"

#define BITSPERBYTE CHAR_BIT
#define MAXINT INT_MAX
#define MININT INT_MIN
#define MAXSHORT SHRT_MAX
#define MINSHORT SHRT_MIN
#define MAXLONG LONG_MAX
#define MINLONG LONG_MIN

#define MAXDOUBLE DBL_MAX
#define MINDOUBLE DBL_MIN
#define MAXFLOAT FLT_MAX
#define MINFLOAT FLT_MIN

#endif // LLVM_LIBC_MACROS_VALUES_MACROS_H
