//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOMIC_COMPARE_EXCHANGE_WEAK_H__
#define __CLC_OPENCL_ATOMIC_ATOMIC_COMPARE_EXCHANGE_WEAK_H__

#define __CLC_FUNCTION atomic_compare_exchange_weak
#define __CLC_COMPARE_EXCHANGE

#define __CLC_BODY <clc/opencl/atomic/atomic_decl.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc/opencl/atomic/atomic_decl.inc>
#include <clc/math/gentype.inc>

#undef __CLC_COMPARE_EXCHANGE
#undef __CLC_FUNCTION

#endif // __CLC_OPENCL_ATOMIC_ATOMIC_COMPARE_EXCHANGE_WEAK_H__
