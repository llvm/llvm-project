//===-- Implementation header for sched_setscheduler -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SCHED_SCHED_SETSCHEDULER_H
#define LLVM_LIBC_SRC_SCHED_SCHED_SETSCHEDULER_H

#include <sched.h>

namespace __llvm_libc {

int sched_setscheduler(pid_t tid, int policy, const struct sched_param *param);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SCHED_SCHED_SETSCHEDULER_H
