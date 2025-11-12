//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef cl_khr_extended_bit_ops

#include <clc/integer/clc_bit_reverse.h>
#include <clc/opencl/integer/bit_reverse.h>

#define __CLC_FUNCTION bit_reverse
#define __CLC_BODY <clc/shared/unary_def.inc>

#include <clc/integer/gentype.inc>

#endif // cl_khr_extended_bit_ops
