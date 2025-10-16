//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_nan.h>

#define __CLC_BODY <clc_nan.inc>
#include <clc/math/gentype.inc>

#define __CLC_FUNCTION __clc_nan

#define __CLC_ARG1_TYPE __CLC_S_GENTYPE
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ARG1_TYPE

#define __CLC_ARG1_TYPE __CLC_U_GENTYPE
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ARG1_TYPE
