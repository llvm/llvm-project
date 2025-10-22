//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_exchange.h>

#define __CLC_FUNCTION __clc_atomic_exchange
#define __CLC_IMPL_FUNCTION __scoped_atomic_exchange_n

#define __CLC_BODY <clc_atomic_def.inc>
#include <clc/integer/gentype.inc>

#undef __CLC_PTR_CASTTYPE
#undef __CLC_AS_RETTYPE
#define __CLC_PTR_CASTTYPE __CLC_BIT_INTN
#define __CLC_AS_RETTYPE(x) __CLC_AS_GENTYPE(x)

#define __CLC_BODY <clc_atomic_def.inc>
#include <clc/math/gentype.inc>
