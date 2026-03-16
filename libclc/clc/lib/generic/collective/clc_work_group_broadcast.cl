//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/atomic/clc_atomic_load.h"
#include "clc/atomic/clc_atomic_store.h"
#include "clc/collective/clc_work_group_broadcast.h"
#include "clc/subgroup/clc_sub_group_broadcast.h"
#include "clc/subgroup/clc_subgroup.h"
#include "clc/synchronization/clc_work_group_barrier.h"
#include "clc/workitem/clc_get_local_id.h"

#pragma OPENCL EXTENSION __cl_clang_function_scope_local_variables : enable

#define __CLC_BODY <clc_work_group_broadcast.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc_work_group_broadcast.inc>
#include <clc/math/gentype.inc>
