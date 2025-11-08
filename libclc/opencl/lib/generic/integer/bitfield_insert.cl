//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef cl_khr_extended_bit_ops

#include <clc/integer/clc_bitfield_insert.h>
#include <clc/opencl/integer/bitfield_insert.h>

#define __CLC_FUNCTION bitfield_insert
#define __CLC_BODY <clc/integer/clc_bitfield_insert.inc>
#include <clc/integer/gentype.inc>

#endif // cl_khr_extended_bit_ops
