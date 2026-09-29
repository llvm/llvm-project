//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/math/clc_atan_helpers.h"
#include "clc/math/clc_atanpi.h"
#include "clc/math/clc_copysign.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_recip_fast.h"

#define __CLC_BODY "clc_atanpi.inc"
#include "clc/math/gentype.inc"
