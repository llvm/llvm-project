//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_INTEGER_BIT_REVERSE_H__
#define __CLC_OPENCL_INTEGER_BIT_REVERSE_H__

#ifdef cl_khr_extended_bit_ops

#include <clc/opencl/opencl-base.h>

#define __CLC_FUNCTION bit_reverse
#define __CLC_BODY <clc/shared/unary_decl.inc>

#include <clc/integer/gentype.inc>

#undef __CLC_FUNCTION

#endif // cl_khr_extended_bit_ops

#endif // __CLC_OPENCL_INTEGER_BIT_REVERSE_H__
