//===-- Definition of macros from sched.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_LINUX_SCHED_MACROS_H
#define __LLVM_LIBC_MACROS_LINUX_SCHED_MACROS_H

// Definitions of SCHED_* macros must match was linux as at:
// https://elixir.bootlin.com/linux/latest/source/include/uapi/linux/sched.h

// Posix required
#define SCHED_OTHER 0
#define SCHED_FIFO 1
#define SCHED_RR 2

// Linux extentions
#define SCHED_BATCH 3
#define SCHED_ISO 4 // Not yet implemented, reserved.
#define SCHED_IDLE 5
#define SCHED_DEADLINE 6

#define CPU_COUNT_S(setsize, set) __sched_getcpucount(setsize, set)
#define CPU_COUNT(set) CPU_COUNT_S(sizeof(cpu_set_t), set)

#endif // __LLVM_LIBC_MACROS_LINUX_SCHED_MACROS_H
