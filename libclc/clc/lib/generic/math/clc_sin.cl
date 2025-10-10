//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/float/definitions.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_sincos_helpers.h>
#include <clc/math/clc_sincos_piby4.h>
#include <clc/math/clc_trunc.h>
#include <clc/math/math.h>
#include <clc/math/tables.h>
#include <clc/relational/clc_isinf.h>
#include <clc/relational/clc_isnan.h>
#include <clc/relational/clc_select.h>
#include <clc/shared/clc_max.h>

#define __CLC_BODY <clc_sin.inc>
#include <clc/math/gentype.inc>
