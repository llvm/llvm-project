//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__opencl_c_atomic_order_seq_cst) &&                                \
    defined(__opencl_c_atomic_scope_device)

#include <clc/atomic/clc_atomic_compare_exchange.h>
#include <clc/opencl/atomic/atomic_compare_exchange_strong.h>

#define __CLC_FUNCTION atomic_compare_exchange_strong
#define __CLC_COMPARE_EXCHANGE

#define __CLC_BODY <atomic_def.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <atomic_def.inc>
#include <clc/math/gentype.inc>

#endif // defined(__opencl_c_atomic_order_seq_cst) &&
       // defined(__opencl_c_atomic_scope_device)
