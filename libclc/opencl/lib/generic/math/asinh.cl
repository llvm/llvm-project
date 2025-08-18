//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_asinh.h>
#include <clc/opencl/math/asinh.h>

#define __CLC_FUNCTION asinh
#define __CLC_BODY <clc/shared/unary_def.inc>

#include <clc/math/gentype.inc>
