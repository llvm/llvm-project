//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_fetch_sub.h>
#include <clc/opencl/atomic/atomic_fetch_sub.h>
#include <clc/opencl/utils.h>

#define __CLC_BODY <atomic_init.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <atomic_init.inc>
#include <clc/math/gentype.inc>
