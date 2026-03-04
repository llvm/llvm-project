//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/geometric/clc_dot.h>
#include <clc/geometric/clc_normalize.h>
#include <clc/math/clc_half_rsqrt.h>

#define __CLC_FLOAT_ONLY
#define __CLC_BODY <clc_fast_normalize.inc>
#include <clc/math/gentype.inc>
