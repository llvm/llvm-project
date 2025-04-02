//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_native_exp2.h>
#include <clc/math/clc_native_log2.h>

#define __FLOAT_ONLY
#define __CLC_BODY <clc_native_powr.inc>

#include <clc/math/gentype.inc>
