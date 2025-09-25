//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/integer/clc_clz.h>
#include <clc/opencl/integer/clz.h>

#define __CLC_FUNCTION clz
#define __CLC_BODY <clc/shared/unary_def.inc>

#include <clc/integer/gentype.inc>
