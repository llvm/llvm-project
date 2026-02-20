//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_UTILS_H__
#define __CLC_OPENCL_UTILS_H__

#include <clc/opencl/opencl-base.h>

static _CLC_INLINE int __opencl_get_clang_memory_scope(memory_scope scope) {
  switch (scope) {
  case __OPENCL_MEMORY_SCOPE_WORK_ITEM:
    return __MEMORY_SCOPE_SINGLE;
#if defined(cl_intel_subgroups) || defined(cl_khr_subgroups) ||                \
    defined(__opencl_c_subgroups)
  case __OPENCL_MEMORY_SCOPE_SUB_GROUP:
    return __MEMORY_SCOPE_WVFRNT;
#endif
  case __OPENCL_MEMORY_SCOPE_WORK_GROUP:
    return __MEMORY_SCOPE_WRKGRP;
  case __OPENCL_MEMORY_SCOPE_DEVICE:
    return __MEMORY_SCOPE_DEVICE;
  default:
    return __MEMORY_SCOPE_SYSTEM;
  }
}

#endif // __CLC_OPENCL_UTILS_H__
