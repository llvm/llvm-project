//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/float/definitions.h"
#include "clc/internal/clc.h"
#include "clc/math/clc_atan_helpers.h"
#include "clc/math/clc_copysign.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_fmax.h"
#include "clc/math/clc_fmin.h"
#include "clc/math/clc_mad.h"
#include "clc/relational/clc_isinf.h"
#include "clc/relational/clc_isunordered.h"
#include "clc/relational/clc_select.h"
#include "clc/relational/clc_signbit.h"
#include "clc/shared/clc_max.h"
#include "clc/shared/clc_min.h"

#define __CLC_BODY "clc_atan2pi.inc"
#include "clc/math/gentype.inc"
