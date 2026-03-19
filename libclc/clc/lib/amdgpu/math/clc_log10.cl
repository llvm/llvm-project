//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/float/definitions.h"
#include "clc/math/clc_ep.h"
#include "clc/math/clc_frexp.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_log10.h"
#include "clc/math/clc_mad.h"
#include "clc/relational/clc_isinf.h"

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION __clc_log10
#define __CLC_IMPL_FUNCTION(x) __builtin_elementwise_log10
#define __CLC_BODY "clc/shared/unary_def.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION
#undef __CLC_IMPL_FUNCTION
#undef __CLC_FLOAT_ONLY

#define __CLC_HALF_ONLY
#define __CLC_FUNCTION __clc_log10
#define __CLC_IMPL_FUNCTION(x) __builtin_elementwise_log10
#define __CLC_BODY "clc/shared/unary_def.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION
#undef __CLC_IMPL_FUNCTION
#undef __CLC_HALF_ONLY

#define COMPILING_LOG10
#define __CLC_DOUBLE_ONLY
#define __CLC_BODY "clc_amdgpu_log.inc"
#include "clc/math/gentype.inc"

#define __CLC_DOUBLE_ONLY
#define __CLC_FUNCTION __clc_log10
#define __CLC_BODY "clc/shared/unary_def_scalarize_loop.inc"
#include "clc/math/gentype.inc"
