//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_fdim.h>
#include <clc/opencl/math/fdim.h>

#define __CLC_FUNCTION fdim
#define __CLC_BODY <clc/shared/binary_def.inc>
#include <clc/math/gentype.inc>
