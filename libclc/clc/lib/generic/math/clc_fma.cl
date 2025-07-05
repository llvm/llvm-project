//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/internal/math/clc_sw_fma.h>
#include <clc/math/math.h>

#define __CLC_BODY <clc_fma.inc>
#include <clc/math/gentype.inc>
