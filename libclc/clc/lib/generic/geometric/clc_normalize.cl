//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/common/clc_sign.h>
#include <clc/float/definitions.h>
#include <clc/geometric/clc_dot.h>
#include <clc/geometric/clc_normalize.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_copysign.h>
#include <clc/math/clc_rsqrt.h>
#include <clc/relational/clc_all.h>
#include <clc/relational/clc_isinf.h>
#include <clc/relational/clc_select.h>

#define __CLC_BODY <clc_normalize.inc>
#include <clc/math/gentype.inc>
