//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLC_MATH_CLC_SINCOS_HELPERS_H
#define CLC_MATH_CLC_SINCOS_HELPERS_H

#define __CLC_FLOAT_ONLY
#define __CLC_BODY <clc/math/clc_sincos_helpers.inc>

#include <clc/math/gentype.inc>

#define __CLC_DOUBLE_ONLY
#define __CLC_BODY <clc/math/clc_sincos_helpers_fp64.inc>

#include <clc/math/gentype.inc>

#endif // CLC_MATH_CLC_SINCOS_HELPERS_H
