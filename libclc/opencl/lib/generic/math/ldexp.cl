//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_ldexp.h>
#include <clc/opencl/math/ldexp.h>

#define __CLC_FUNCTION ldexp
#define __CLC_IMPL_FUNCTION(x) __clc_ldexp
#define __CLC_BODY <clc/shared/binary_def_with_int_second_arg.inc>

#include <clc/math/gentype.inc>

// This defines all the ldexp(GENTYPE, int) variants
#define __CLC_BODY <ldexp.inc>
#include <clc/math/gentype.inc>
