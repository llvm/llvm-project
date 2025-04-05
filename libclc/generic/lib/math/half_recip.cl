//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

#define recip(x) (1.0f/x)

#define __CLC_FUNC recip
#define __CLC_BODY <half_unary.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>

#undef recip
