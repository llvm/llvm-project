//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/math/clc_div_fast.h"
#include "clc/math/clc_ep.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_recip_fast.h"
#include "clc/math/clc_sqrt_fast.h"
#include "clc/relational/clc_isinf.h"
#include "clc/relational/clc_signbit.h"

#define __CLC_BODY <clc_ep.inc>
#include <clc/math/gentype.inc>
