//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_native_sqrt.h>

#define __CLC_FUNC sqrt
#define __FLOAT_ONLY
#define __CLC_BODY <clc_half_native_unary.inc>
#include <clc/math/gentype.inc>
