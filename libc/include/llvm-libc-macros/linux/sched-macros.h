//===-- Definition of macros from sched.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_LINUX_SCHED_MACROS_H
#define __LLVM_LIBC_MACROS_LINUX_SCHED_MACROS_H

#define CPU_COUNT_S(setsize, set) __sched_getcpucount(setsize, set)
#define CPU_COUNT(set) CPU_COUNT_S(sizeof(cpu_set_t), set)

#endif // __LLVM_LIBC_MACROS_LINUX_SCHED_MACROS_H
