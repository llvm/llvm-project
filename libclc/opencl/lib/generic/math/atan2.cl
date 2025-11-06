//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_atan2.h>
#include <clc/opencl/math/atan2.h>

#define __CLC_FUNCTION atan2
#define __CLC_BODY <clc/shared/binary_def.inc>

#include <clc/math/gentype.inc>
