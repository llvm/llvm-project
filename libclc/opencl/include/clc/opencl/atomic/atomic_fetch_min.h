//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOMIC_FETCH_MIN_H__
#define __CLC_OPENCL_ATOMIC_ATOMIC_FETCH_MIN_H__

#define FUNCTION atomic_fetch_min

#define __CLC_BODY <clc/opencl/atomic/atomic_decl.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc/opencl/atomic/atomic_decl.inc>
#include <clc/math/gentype.inc>

#undef FUNCTION

#endif // __CLC_OPENCL_ATOMIC_ATOMIC_FETCH_MIN_H__
