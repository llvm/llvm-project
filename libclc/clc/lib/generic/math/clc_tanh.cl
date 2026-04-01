//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/math/clc_copysign.h"
#include "clc/math/clc_ep.h"
#include "clc/math/clc_exp.h"
#include "clc/math/clc_exp2_fast.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_recip_fast.h"
#include "clc/math/clc_tanh.h"

#define __CLC_BODY "clc_tanh.inc"
#include "clc/math/gentype.inc"
