//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/geometric/clc_distance.h>
#include <clc/opencl/geometric/distance.h>

#define __CLC_FUNCTION distance
#define __CLC_BODY <clc/geometric/binary_def.inc>
#include <clc/math/gentype.inc>
