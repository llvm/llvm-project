//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_dec.h>
#include <clc/opencl/atomic/atomic_dec.h>

#define __CLC_FUNCTION atomic_dec
#define __CLC_IMPL_FUNCTION __clc_atomic_dec

#define __CLC_BODY <atomic_inc_dec.inc>
#include <clc/integer/gentype.inc>
