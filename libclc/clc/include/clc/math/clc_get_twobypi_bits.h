//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility function for trigonometric reductions to extract bits out of 2/pi
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MATH_CLC_GET_TWOBYPI_BITS_H__
#define __CLC_MATH_CLC_GET_TWOBYPI_BITS_H__

#define __CLC_DOUBLE_ONLY
#define __CLC_BODY "clc/math/clc_get_twobypi_bits_decl.inc"
#define __CLC_FUNCTION __clc_get_twobypi_bits

#include "clc/math/gentype.inc"

#undef __CLC_FUNCTION
#undef __CLC_DOUBLE_ONLY

#endif // __CLC_MATH_CLC_GET_TWOBYPI_BITS_H__
