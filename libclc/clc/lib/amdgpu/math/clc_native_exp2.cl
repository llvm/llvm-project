//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>

#define __CLC_FLOAT_ONLY
#define __CLC_MIN_VECSIZE 1
#define __CLC_FUNCTION __clc_native_exp2
#define __CLC_IMPL_FUNCTION __builtin_amdgcn_exp2f
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
