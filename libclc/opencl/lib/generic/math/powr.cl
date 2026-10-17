//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/math/clc_powr.h"

#define __CLC_FUNCTION powr
#define __CLC_BODY "clc/shared/binary_def.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION
#undef __CLC_IMPL_FUNCTION

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION __powr_fast
#define __CLC_IMPL_FUNCTION(x) __clc_powr_fast
#define __CLC_BODY "clc/shared/binary_def.inc"
#include "clc/math/gentype.inc"
