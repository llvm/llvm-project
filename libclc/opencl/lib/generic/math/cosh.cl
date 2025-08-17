//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_cosh.h>
#include <clc/opencl/math/cosh.h>

#define __CLC_FUNCTION cosh
#define __CLC_BODY <clc/shared/unary_def.inc>
#include <clc/math/gentype.inc>
