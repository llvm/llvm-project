//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/math/clc_tgamma.h"

#include "clc/clc_convert.h"
#include "clc/float/definitions.h"
#include "clc/math/clc_copysign.h"
#include "clc/math/clc_exp.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_powr.h"
#include "clc/math/clc_recip_fast.h"
#include "clc/math/clc_sinpi.h"
#include "clc/math/clc_trunc.h"
#include "clc/relational/clc_isnan.h"

#define __CLC_BODY "clc_tgamma.inc"
#include "clc/math/gentype.inc"

#define __CLC_FUNCTION __clc_tgamma
#define __CLC_BODY "clc/shared/unary_def_scalarize_loop.inc"
#include "clc/math/gentype.inc"
