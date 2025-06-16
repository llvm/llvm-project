//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/workitem/get_work_dim.h>
#include <clc/workitem/clc_get_work_dim.h>

_CLC_OVERLOAD _CLC_DEF uint clc_get_work_dim() { return clc_get_work_dim(); }
