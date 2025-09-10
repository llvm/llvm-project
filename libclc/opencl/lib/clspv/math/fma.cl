//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/math/clc_sw_fma.h>
#include <clc/opencl/math/fma.h>

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION fma
#define __CLC_IMPL_FUNCTION(x) __clc_sw_fma
#define __CLC_BODY <clc/shared/ternary_def.inc>

#include <clc/math/gentype.inc>
