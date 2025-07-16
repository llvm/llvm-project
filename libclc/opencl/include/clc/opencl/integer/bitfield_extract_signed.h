//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_INTEGER_BITFIELD_EXTRACT_SIGNED_H__
#define __CLC_OPENCL_INTEGER_BITFIELD_EXTRACT_SIGNED_H__

#include <clc/opencl/opencl-base.h>

#define FUNCTION bitfield_extract_signed
#define __RETTYPE __CLC_S_GENTYPE

#define __CLC_BODY <clc/opencl/integer/bitfield_extract.inc>
#include <clc/integer/gentype.inc>

#undef __RETTYPE
#undef FUNCTION

#endif // __CLC_OPENCL_INTEGER_BITFIELD_EXTRACT_SIGNED_H__
