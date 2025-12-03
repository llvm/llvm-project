//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ASYNC_WAIT_GROUP_EVENTS_H__
#define __CLC_OPENCL_ASYNC_WAIT_GROUP_EVENTS_H__

#include <clc/opencl/opencl-base.h>

_CLC_DECL _CLC_OVERLOAD void wait_group_events(int num_events,
                                               event_t *event_list);

#endif // __CLC_OPENCL_ASYNC_WAIT_GROUP_EVENTS_H__
