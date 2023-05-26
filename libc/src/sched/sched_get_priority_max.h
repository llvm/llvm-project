//===-- Implementation header for sched_get_priority_max ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SCHED_SCHED_GET_PRIORITY_MAX_H
#define LLVM_LIBC_SRC_SCHED_SCHED_GET_PRIORITY_MAX_H

namespace __llvm_libc {

int sched_get_priority_max(int policy);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SCHED_SCHED_GET_PRIORITY_MAX_H
