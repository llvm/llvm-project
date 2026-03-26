//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/float/definitions.h"
#include "clc/internal/clc.h"
#include "clc/math/clc_ep.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_frexp.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_mad.h"
#include "clc/math/math.h"
#include "clc/math/tables.h"
#include "clc/relational/clc_isinf.h"
#include "clc/relational/clc_isnan.h"

#define COMPILING_LOG2
#define __CLC_BODY "clc_log_base.inc"
#include "clc/math/gentype.inc"
#undef COMPILING_LOG2

#define __CLC_FUNCTION __clc_log2
#define __CLC_BODY "clc/shared/unary_def_scalarize_loop.inc"
#include "clc/math/gentype.inc"
