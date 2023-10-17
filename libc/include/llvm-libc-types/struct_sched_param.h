//===-- Definition of type struct sched_param -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_STRUCT_SCHED_PARAM_H__
#define __LLVM_LIBC_TYPES_STRUCT_SCHED_PARAM_H__

#include <llvm-libc-types/pid_t.h>
#include <llvm-libc-types/struct_timespec.h>
#include <llvm-libc-types/time_t.h>

struct sched_param {
  // Process or thread execution scheduling priority.
  int sched_priority;
};

#endif // __LLVM_LIBC_TYPES_STRUCT_SCHED_PARAM_H__
