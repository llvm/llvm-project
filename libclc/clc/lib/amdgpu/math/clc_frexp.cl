//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_frexp.h>

#define __CLC_BODY <clc_frexp_builtin.inc>
#define __CLC_ADDRESS_SPACE private
#define __CLC_PRIVATE
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE
#undef __CLC_PRIVATE

#define __CLC_BODY <clc_frexp_builtin.inc>
#define __CLC_ADDRESS_SPACE global
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <clc_frexp_builtin.inc>
#define __CLC_ADDRESS_SPACE local
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define __CLC_BODY <clc_frexp_builtin.inc>
#define __CLC_ADDRESS_SPACE generic
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE
#endif

#define __CLC_FUNCTION __clc_frexp
#define __CLC_ARG2_TYPE int
#define __CLC_ADDRSPACE private
#define __CLC_BODY <clc/shared/unary_def_with_ptr_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ADDRSPACE
#undef __CLC_ARG2_TYPE
#undef __CLC_FUNCTION
