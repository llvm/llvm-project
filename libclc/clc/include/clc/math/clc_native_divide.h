//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLC_MATH_CLC_NATIVE_DIVIDE_H
#define CLC_MATH_CLC_NATIVE_DIVIDE_H

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION __clc_native_divide
#define __CLC_BODY <clc/shared/binary_decl.inc>

#include <clc/math/gentype.inc>

#undef __CLC_FUNCTION

#endif // CLC_MATH_CLC_NATIVE_DIVIDE_H
