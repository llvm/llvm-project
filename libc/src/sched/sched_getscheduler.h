//===-- Implementation header for sched_getscheduler -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SCHED_SCHED_GETSCHEDULER_H
#define LLVM_LIBC_SRC_SCHED_SCHED_GETSCHEDULER_H

#include <sched.h>

namespace LIBC_NAMESPACE {

int sched_getscheduler(pid_t tid);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SCHED_SCHED_GETSCHEDULER_H
