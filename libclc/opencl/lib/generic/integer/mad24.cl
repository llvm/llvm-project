//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/integer/clc_mad24.h>
#include <clc/opencl/integer/mad24.h>

#define __CLC_FUNCTION mad24
#define __CLC_BODY <clc/shared/ternary_def.inc>

#include <clc/integer/gentype24.inc>
