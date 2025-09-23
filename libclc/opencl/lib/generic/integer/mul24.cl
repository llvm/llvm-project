//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/integer/clc_mul24.h>
#include <clc/opencl/integer/mul24.h>

#define __CLC_FUNCTION mul24
#define __CLC_BODY <clc/shared/binary_def.inc>

#include <clc/integer/gentype24.inc>
