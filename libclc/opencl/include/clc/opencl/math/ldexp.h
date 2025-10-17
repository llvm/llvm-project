//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_MATH_LDEXP_H__
#define __CLC_OPENCL_MATH_LDEXP_H__

#define __CLC_FUNCTION ldexp
#define __CLC_BODY <clc/shared/binary_decl_with_int_second_arg.inc>
#include <clc/math/gentype.inc>
#undef __CLC_FUNCTION

#define __CLC_BODY <clc/opencl/math/ldexp.inc>
#include <clc/math/gentype.inc>

#endif // __CLC_OPENCL_MATH_LDEXP_H__
