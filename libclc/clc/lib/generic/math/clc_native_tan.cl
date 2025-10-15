//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_native_cos.h>
#include <clc/math/clc_native_sin.h>

#define __CLC_FLOAT_ONLY
#define __CLC_BODY <clc_native_tan.inc>

#include <clc/math/gentype.inc>
