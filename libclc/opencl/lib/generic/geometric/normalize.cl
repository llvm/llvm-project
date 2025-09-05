//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/geometric/clc_normalize.h>
#include <clc/opencl/geometric/normalize.h>

#define __CLC_FUNCTION normalize
#define __CLC_GEOMETRIC_RET_GENTYPE
#define __CLC_BODY <clc/geometric/unary_def.inc>

#include <clc/math/gentype.inc>
