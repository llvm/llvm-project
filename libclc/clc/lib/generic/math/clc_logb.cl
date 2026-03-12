//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/float/definitions.h"
#include "clc/integer/clc_clz.h"
#include "clc/internal/clc.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_frexp_exp.h"
#include "clc/math/math.h"
#include "clc/relational/clc_isfinite.h"
#include "clc/relational/clc_select.h"

#define __CLC_BODY <clc_logb.inc>
#include <clc/math/gentype.inc>
