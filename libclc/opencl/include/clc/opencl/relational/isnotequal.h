//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_RELATIONAL_ISNOTEQUAL_H__
#define __CLC_OPENCL_RELATIONAL_ISNOTEQUAL_H__

#include <clc/opencl/opencl-base.h>

#define __CLC_FUNCTION isnotequal
#define __CLC_BODY <clc/relational/binary_decl.inc>

#include <clc/math/gentype.inc>

#undef __CLC_FUNCTION

#endif // __CLC_OPENCL_RELATIONAL_ISNOTEQUAL_H__
