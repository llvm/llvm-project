//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_CLC_TARGET_DEFINES_H__
#define __CLC_CLC_TARGET_DEFINES_H__

#if defined(__AMDGPU__) || defined(__NVPTX__)
#define __CLC_MAX_WORK_GROUP_SIZE 1024
#define __CLC_MIN_NATIVE_SUB_GROUP_SIZE 32
#else
#define __CLC_MAX_WORK_GROUP_SIZE 4096
#define __CLC_MIN_NATIVE_SUB_GROUP_SIZE 1
#endif

#define __CLC_MAX_NUM_WORK_GROUPS                                              \
  (__CLC_MAX_WORK_GROUP_SIZE / __CLC_MIN_NATIVE_SUB_GROUP_SIZE)

#endif // __CLC_CLC_TARGET_DEFINES_H__
