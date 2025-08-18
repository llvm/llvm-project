//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/relational/isnan.h>
#include <clc/relational/clc_isnan.h>

#define __CLC_FUNCTION isnan
#define __CLC_BODY "unary_def.inc"

#include <clc/math/gentype.inc>
