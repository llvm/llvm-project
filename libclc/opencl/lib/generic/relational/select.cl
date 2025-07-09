//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/relational/select.h>
#include <clc/relational/clc_select.h>
#include <clc/utils.h>

#define __CLC_SELECT_FN select
#define __CLC_SELECT_DEF(x, y, z) return __clc_select(x, y, z)

#define __CLC_BODY <clc/relational/clc_select_impl.inc>
#include <clc/math/gentype.inc>
#define __CLC_BODY <clc/relational/clc_select_impl.inc>
#include <clc/integer/gentype.inc>
