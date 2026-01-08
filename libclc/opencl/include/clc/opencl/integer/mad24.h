//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_INTEGER_MAD24_H__
#define __CLC_OPENCL_INTEGER_MAD24_H__

#include <clc/opencl/opencl-base.h>

#define __CLC_FUNCTION mad24
#define __CLC_BODY <clc/shared/ternary_decl.inc>

#include <clc/integer/gentype24.inc>

#undef __CLC_FUNCTION

#endif // __CLC_OPENCL_INTEGER_MAD24_H__
