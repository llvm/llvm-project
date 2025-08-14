//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_native_sin.h>
#include <clc/opencl/math/native_sin.h>

#define __FLOAT_ONLY
#define FUNCTION native_sin
#define __CLC_BODY <clc/shared/unary_def.inc>

#include <clc/math/gentype.inc>
