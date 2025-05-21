//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/clcmacro.h>
#include <clc/integer/clc_clz.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_floor.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_ldexp.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/clc_trunc.h>
#include <clc/math/math.h>
#include <clc/shared/clc_max.h>

#define __CLC_ADDRESS_SPACE private
#include <clc_remquo.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_ADDRESS_SPACE global
#include <clc_remquo.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_ADDRESS_SPACE local
#include <clc_remquo.inc>
#undef __CLC_ADDRESS_SPACE
