//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOMIC_INIT_H__
#define __CLC_OPENCL_ATOMIC_ATOMIC_INIT_H__

#include <clc/opencl/opencl-base.h>

#define __CLC_ATOMIC_GENTYPE __CLC_XCONCAT(atomic_, __CLC_GENTYPE)

#define __CLC_BODY <clc/opencl/atomic/atomic_init.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc/opencl/atomic/atomic_init.inc>
#include <clc/math/gentype.inc>

#undef __CLC_ATOMIC_GENTYPE

#endif // __CLC_OPENCL_ATOMIC_ATOMIC_INIT_H__
