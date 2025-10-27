//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_lgamma_r.h>
#include <clc/opencl/math/lgamma_r.h>

#define __CLC_FUNCTION lgamma_r
#define __CLC_BODY <clc/math/unary_def_with_int_ptr.inc>
#include <clc/math/gentype.inc>
