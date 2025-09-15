//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/geometric/clc_fast_normalize.h>
#include <clc/opencl/geometric/fast_normalize.h>

#define __CLC_FUNCTION fast_normalize
#define __CLC_FLOAT_ONLY
#define __CLC_GEOMETRIC_RET_GENTYPE
#define __CLC_BODY <clc/geometric/unary_def.inc>

#include <clc/math/gentype.inc>
