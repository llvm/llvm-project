//===-- Implementation header for sched_getaffinity -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_SCHED_GETAFFINITY_H
#define LLVM_LIBC_SRC_UNISTD_SCHED_GETAFFINITY_H

#include <sched.h>

namespace __llvm_libc {

int sched_getaffinity(pid_t tid, size_t cpuset_size, cpu_set_t *mask);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_UNISTD_SCHED_GETAFFINITY_H
