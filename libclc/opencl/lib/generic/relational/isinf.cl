//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/relational/isinf.h>
#include <clc/relational/clc_isinf.h>

#define __CLC_FUNCTION isinf
#define __CLC_BODY "unary_def.inc"

#include <clc/math/gentype.inc>
