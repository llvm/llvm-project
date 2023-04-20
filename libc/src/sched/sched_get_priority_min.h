//===-- Implementation header for sched_get_priority_min ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SCHED_SCHED_GET_PRIORITY_MIN_H
#define LLVM_LIBC_SRC_SCHED_SCHED_GET_PRIORITY_MIN_H

namespace __llvm_libc {

int sched_get_priority_min(int policy);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SCHED_SCHED_GET_PRIORITY_MIN_H
