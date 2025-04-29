//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __OPENCL_C_VERSION__ >= CL_VERSION_2_0

#define __CLC_FUNCTION ctz
#define __CLC_BODY <clc/shared/unary_decl.inc>

#include <clc/integer/gentype.inc>

#undef __CLC_BODY
#undef __CLC_FUNCTION

#endif // __OPENCL_C_VERSION__ >= CL_VERSION_2_0
