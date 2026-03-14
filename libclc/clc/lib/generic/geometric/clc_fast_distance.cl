//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/geometric/clc_fast_length.h>
#include <clc/internal/clc.h>

#define __CLC_FLOAT_ONLY
#define __CLC_BODY <clc_fast_distance.inc>
#include <clc/math/gentype.inc>
