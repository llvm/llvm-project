//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef cl_khr_extended_bit_ops

#include <clc/integer/clc_bitfield_extract_unsigned.h>
#include <clc/opencl/integer/bitfield_extract_unsigned.h>

#define FUNCTION bitfield_extract_unsigned
#define __RETTYPE __CLC_U_GENTYPE

#define __CLC_BODY <bitfield_extract_def.inc>
#include <clc/integer/gentype.inc>

#endif // cl_khr_extended_bit_ops
