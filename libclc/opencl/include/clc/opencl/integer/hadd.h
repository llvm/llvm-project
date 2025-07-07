//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_INTEGER_HADD_H__
#define __CLC_OPENCL_INTEGER_HADD_H__

#include <clc/opencl/opencl-base.h>

#define FUNCTION hadd
#define __CLC_BODY <clc/shared/binary_decl.inc>

#include <clc/integer/gentype.inc>

#undef FUNCTION

#endif // __CLC_OPENCL_INTEGER_HADD_H__
