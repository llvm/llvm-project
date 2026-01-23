//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef cl_khr_fp64

#include <clc/clc_convert.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_ep_log.h>
#include <clc/math/clc_fma.h>
#include <clc/math/math.h>
#include <clc/math/tables.h>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __CLC_BODY <clc_ep_log.inc>
#include <clc/math/gentype.inc>

#endif
