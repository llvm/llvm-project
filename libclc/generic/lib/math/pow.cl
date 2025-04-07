//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

#include <math/clc_pow.h>

#define __CLC_FUNC pow
#define __CLC_BODY <clc_sw_binary.inc>
#include <clc/math/gentype.inc>
