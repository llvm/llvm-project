//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_exchange.h>
#include <clc/opencl/utils.h>

#define __CLC_FUNCTION atomic_exchange
#define __CLC_IMPL_FUNCTION __clc_atomic_exchange

#define __CLC_BODY <atomic_def.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <atomic_def.inc>
#include <clc/math/gentype.inc>
