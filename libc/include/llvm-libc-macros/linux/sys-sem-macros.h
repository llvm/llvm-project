//===-- Definition of macros from sys/sem.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_LINUX_SYS_SEM_MACROS_H
#define LLVM_LIBC_MACROS_LINUX_SYS_SEM_MACROS_H

// semop flags
#define SEM_UNDO 0x1000

// semctl command definitions
#define GETPID 11
#define GETVAL 12
#define GETALL 13
#define GETNCNT 14
#define GETZCNT 15
#define SETVAL 16
#define SETALL 17

// linux specific extensions
#define SEM_STAT 18
#define SEM_INFO 19
#define SEM_STAT_ANY 20

// The union a caller passes to semctl is the caller's to declare: the
// standard says so, and no libc defines it. Saying that here is what lets a
// program tell "this libc leaves it to me" from "this libc has it already",
// which is the test a great deal of software makes before declaring its own.
#define _SEM_SEMUN_UNDEFINED 1

#endif // LLVM_LIBC_MACROS_LINUX_SYS_SEM_MACROS_H
