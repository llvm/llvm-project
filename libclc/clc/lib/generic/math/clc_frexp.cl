//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/clcfunc.h>
#include <clc/internal/clc.h>
#include <clc/math/math.h>
#include <clc/relational/clc_isinf.h>
#include <clc/relational/clc_isnan.h>
#include <clc/relational/clc_select.h>
#include <clc/utils.h>

#define __CLC_BODY <clc_frexp.inc>
#define __CLC_ADDRESS_SPACE private
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <clc_frexp.inc>
#define __CLC_ADDRESS_SPACE global
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <clc_frexp.inc>
#define __CLC_ADDRESS_SPACE local
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define __CLC_BODY <clc_frexp.inc>
#define __CLC_ADDRESS_SPACE generic
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE
#endif
