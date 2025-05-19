//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/geometric/clc_fast_length.h>

#define __FLOAT_ONLY
#define FUNCTION fast_length
#define __CLC_BODY <clc/geometric/unary_def.inc>
#include <clc/math/gentype.inc>
