//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MATH_DIV_CR_H__
#define __CLC_MATH_DIV_CR_H__

// Declare overloads of __clc_div_cr. This is a wrapper around the
// floating-point / operator. This is a utilty to deal with the language default
// division not being correctly rounded, and requires the
// -cl-fp32-correctly-rounded-divide-sqrt flag. This will just be the operator
// compiled with that option. Ideally clang would expose a direct way to get the
// correctly rounded and opencl precision versions.

#define __CLC_FUNCTION __clc_div_cr
#define __CLC_BODY "clc/shared/binary_decl.inc"

#include "clc/math/gentype.inc"

#undef __CLC_FUNCTION

#endif // __CLC_MATH_DIV_CR_H__
