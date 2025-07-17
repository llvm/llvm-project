//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_INTEGER_BITFIELD_INSERT_H__
#define __CLC_OPENCL_INTEGER_BITFIELD_INSERT_H__

#ifdef cl_khr_extended_bit_ops

#include <clc/opencl/opencl-base.h>

#define __CLC_BODY <clc/integer/clc_bitfield_insert.inc>
#include <clc/integer/gentype.inc>

#undef FUNCTION

#endif // cl_khr_extended_bit_ops

#endif // __CLC_OPENCL_INTEGER_BITFIELD_INSERT_H__
