//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/integer/clc_abs.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_mad.h>
#include <clc/math/clc_sqrt.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>
#include <clc/relational/clc_isnan.h>
#include <clc/shared/clc_clamp.h>

#define __CLC_BODY <clc_hypot.inc>
#include <clc/math/gentype.inc>
