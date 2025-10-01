//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/shared/vload.h>
#include <clc/shared/clc_vload.h>

#define __CLC_BODY "vload.inc"
#include <clc/integer/gentype.inc>

#define __CLC_BODY "vload.inc"
#include <clc/math/gentype.inc>
