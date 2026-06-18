//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/internal/clc.h"
#include "clc/math/clc_ep.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_sqrt_fast.h"
#include "clc/math/math.h"

#define __CLC_BODY "clc_acospi.inc"
#include "clc/math/gentype.inc"
