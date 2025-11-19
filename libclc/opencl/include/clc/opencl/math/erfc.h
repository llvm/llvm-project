//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_MATH_ERFC_H__
#define __CLC_OPENCL_MATH_ERFC_H__

#undef erfc

#define __CLC_BODY <clc/math/unary_decl.inc>
#define __CLC_FUNCTION erfc

#include <clc/math/gentype.inc>

#undef __CLC_FUNCTION

#endif // __CLC_OPENCL_MATH_ERFC_H__
