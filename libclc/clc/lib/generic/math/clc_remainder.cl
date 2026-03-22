//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/math/clc_remainder.h"

#include "clc/clc_convert.h"
#include "clc/integer/clc_clz.h"

#include "clc/float/definitions.h"
#include "clc/math/clc_copysign.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_floor.h"
#include "clc/math/clc_flush_if_daz.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_frexp.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_recip_fast.h"
#include "clc/math/clc_rint.h"
#include "clc/math/clc_trunc.h"
#include "clc/math/math.h"
#include "clc/relational/clc_isfinite.h"
#include "clc/relational/clc_isnan.h"
#include "clc/relational/clc_signbit.h"

#define __CLC_BODY "clc_remainder.inc"
#include "clc/math/gentype.inc"

#define __CLC_FUNCTION __clc_remainder
#define __CLC_BODY "clc/shared/binary_def_scalarize_loop.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION
