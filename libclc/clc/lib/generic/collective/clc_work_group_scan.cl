//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clc/collective/clc_work_group_scan.h"

#include "clc/clc_target_defines.h"

#include "clc/atomic/clc_atomic_load.h"
#include "clc/atomic/clc_atomic_store.h"
#include "clc/math/clc_fmax.h"
#include "clc/math/clc_fmin.h"
#include "clc/shared/clc_max.h"
#include "clc/shared/clc_min.h"

#include "clc/subgroup/clc_sub_group_scan.h"
#include "clc/subgroup/clc_subgroup.h"
#include "clc/synchronization/clc_work_group_barrier.h"
#include "clc/workitem/clc_get_num_sub_groups.h"
#include "clc/workitem/clc_get_sub_group_id.h"
#include "clc/workitem/clc_get_sub_group_local_id.h"

#pragma OPENCL EXTENSION __cl_clang_function_scope_local_variables : enable

enum __CLC_WORK_GROUP_SCAN_OP {
  __CLC_WORK_GROUP_SCAN_ADD,
  __CLC_WORK_GROUP_SCAN_MIN,
  __CLC_WORK_GROUP_SCAN_MAX
};

#define __CLC_BODY "clc_work_group_scan.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "clc_work_group_scan.inc"
#include "clc/math/gentype.inc"
