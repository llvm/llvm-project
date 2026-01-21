//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/float/definitions.h>
#include <clc/integer/clc_clz.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_copysign.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_floor.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_frexp.h>
#include <clc/math/clc_ldexp.h>
#include <clc/math/clc_nan.h>
#include <clc/math/clc_native_recip.h>
#include <clc/math/clc_rint.h>
#include <clc/math/clc_sincos_helpers.h>
#include <clc/math/clc_trunc.h>
#include <clc/math/math.h>
#include <clc/relational/clc_isfinite.h>
#include <clc/relational/clc_isnan.h>
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

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define __CLC_ADDRESS_SPACE generic
#include <clc_remquo.inc>
#undef __CLC_ADDRESS_SPACE
#endif
