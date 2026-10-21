//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/subgroup/clc_sub_group_non_uniform_reduce.h"
#include "clc/subgroup/clc_sub_group_reduce.h"

// The implementation is the same as the nonuniform case, so just call the
// nonuniform versions of every function.

#define __CLC_BODY "clc_sub_group_reduce.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "clc_sub_group_reduce.inc"
#include "clc/math/gentype.inc"
