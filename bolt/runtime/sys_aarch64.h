//===- bolt/runtime/sys_aarch64.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_SYS_AARCH64
#define LLVM_TOOLS_LLVM_BOLT_SYS_AARCH64

#include "runtime_types.h"

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                                                               \
  "stp x0, x1, [sp, #-16]!\n"                                                  \
  "stp x2, x3, [sp, #-16]!\n"                                                  \
  "stp x4, x5, [sp, #-16]!\n"                                                  \
  "stp x6, x7, [sp, #-16]!\n"                                                  \
  "stp x8, x9, [sp, #-16]!\n"                                                  \
  "stp x10, x11, [sp, #-16]!\n"                                                \
  "stp x12, x13, [sp, #-16]!\n"                                                \
  "stp x14, x15, [sp, #-16]!\n"                                                \
  "stp x16, x17, [sp, #-16]!\n"                                                \
  "stp x18, x19, [sp, #-16]!\n"                                                \
  "stp x20, x21, [sp, #-16]!\n"                                                \
  "stp x22, x23, [sp, #-16]!\n"                                                \
  "stp x24, x25, [sp, #-16]!\n"                                                \
  "stp x26, x27, [sp, #-16]!\n"                                                \
  "stp x28, x29, [sp, #-16]!\n"                                                \
  "mrs x29, nzcv\n"                                                            \
  "stp x29, x30, [sp, #-16]!\n"
// Mirrors SAVE_ALL
#define RESTORE_ALL                                                            \
  "ldp x29, x30, [sp], #16\n"                                                  \
  "msr nzcv, x29\n"                                                            \
  "ldp x28, x29, [sp], #16\n"                                                  \
  "ldp x26, x27, [sp], #16\n"                                                  \
  "ldp x24, x25, [sp], #16\n"                                                  \
  "ldp x22, x23, [sp], #16\n"                                                  \
  "ldp x20, x21, [sp], #16\n"                                                  \
  "ldp x18, x19, [sp], #16\n"                                                  \
  "ldp x16, x17, [sp], #16\n"                                                  \
  "ldp x14, x15, [sp], #16\n"                                                  \
  "ldp x12, x13, [sp], #16\n"                                                  \
  "ldp x10, x11, [sp], #16\n"                                                  \
  "ldp x8, x9, [sp], #16\n"                                                    \
  "ldp x6, x7, [sp], #16\n"                                                    \
  "ldp x4, x5, [sp], #16\n"                                                    \
  "ldp x2, x3, [sp], #16\n"                                                    \
  "ldp x0, x1, [sp], #16\n"

// https://github.com/torvalds/linux/blob/v7.1/arch/arm64/tools/syscall_64.tbl
// https://github.com/torvalds/linux/blob/v7.1/scripts/syscall.tbl
// scripts/syscalltbl.sh --abis common,64 arch/arm64/tools/syscall_64.tbl
// arm64_syscalls.h
#define __NR_ftruncate 46
#define __NR_openat 56
#define __NR_close 57
#define __NR_getdents64 61
#define __NR_lseek 62
#define __NR_read 63
#define __NR_write 64
#define __NR_readlinkat 78
#define __NR_fsync 82
#define __NR_exit 93
#define __NR_exit_group 94
#define __NR_nanosleep 101
#define __NR_kill 129
#define __NR_rt_sigprocmask 135
#define __NR_setpgid 154
#define __NR_getpgid 155
#define __NR_uname 160
#define __NR_prctl 167
#define __NR_getpid 172
#define __NR_getppid 173
#define __NR_munmap 215
#define __NR_clone 220
#define __NR_mmap 222
#define __NR_mprotect 226
#define __NR_madvise 233

// Anonymous namespace covering everything but our library entry point
namespace {

// Get the difference between runtime address of .text section and
// static address in section header table. Can be extracted from arbitrary
// pc value recorded at runtime to get the corresponding static address, which
// in turn can be used to search for indirect call description. Needed because
// indirect call descriptions are read-only non-relocatable data.
uint64_t getTextBaseAddress() {
  uint64_t DynAddr;
  uint64_t StaticAddr;
  __asm__ volatile("b .instr%=\n\t"
                   ".StaticAddr%=:\n\t"
                   ".dword __hot_end\n\t"
                   ".instr%=:\n\t"
                   "ldr %0, .StaticAddr%=\n\t"
                   "adrp %1, __hot_end\n\t"
                   "add %1, %1, :lo12:__hot_end\n\t"
                   : "=r"(StaticAddr), "=r"(DynAddr));
  return DynAddr - StaticAddr;
}

} // anonymous namespace

#endif /* LLVM_TOOLS_LLVM_BOLT_SYS_AARCH64 */
