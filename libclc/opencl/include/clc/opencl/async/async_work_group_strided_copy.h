//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ASYNC_ASYNC_WORK_GROUP_STRIDED_COPY_H__
#define __CLC_OPENCL_ASYNC_ASYNC_WORK_GROUP_STRIDED_COPY_H__

#define __CLC_DST_ADDR_SPACE local
#define __CLC_SRC_ADDR_SPACE global
#define __CLC_BODY <clc/opencl/async/async_work_group_strided_copy.inc>
#include <clc/integer/gentype.inc>
#define __CLC_BODY <clc/opencl/async/async_work_group_strided_copy.inc>
#include <clc/math/gentype.inc>
#undef __CLC_DST_ADDR_SPACE
#undef __CLC_SRC_ADDR_SPACE

#define __CLC_DST_ADDR_SPACE global
#define __CLC_SRC_ADDR_SPACE local
#define __CLC_BODY <clc/opencl/async/async_work_group_strided_copy.inc>
#include <clc/integer/gentype.inc>
#define __CLC_BODY <clc/opencl/async/async_work_group_strided_copy.inc>
#include <clc/math/gentype.inc>
#undef __CLC_DST_ADDR_SPACE
#undef __CLC_SRC_ADDR_SPACE

#endif // __CLC_OPENCL_ASYNC_ASYNC_WORK_GROUP_STRIDED_COPY_H__
