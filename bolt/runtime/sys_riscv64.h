//===- bolt/runtime/sys_riscv64.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_SYS_RISCV
#define LLVM_TOOLS_LLVM_BOLT_SYS_RISCV

#include "runtime_types.h"

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                                                               \
  "addi sp, sp, -256\n"                                                        \
  "sd x1, 0(sp)\n"                                                             \
  "sd x2, 8(sp)\n"                                                             \
  "sd x3, 16(sp)\n"                                                            \
  "sd x4, 24(sp)\n"                                                            \
  "sd x5, 32(sp)\n"                                                            \
  "sd x6, 40(sp)\n"                                                            \
  "sd x7, 48(sp)\n"                                                            \
  "sd x8, 56(sp)\n"                                                            \
  "sd x9, 64(sp)\n"                                                            \
  "sd x10, 72(sp)\n"                                                           \
  "sd x11, 80(sp)\n"                                                           \
  "sd x12, 88(sp)\n"                                                           \
  "sd x13, 96(sp)\n"                                                           \
  "sd x14, 104(sp)\n"                                                          \
  "sd x15, 112(sp)\n"                                                          \
  "sd x16, 120(sp)\n"                                                          \
  "sd x17, 128(sp)\n"                                                          \
  "sd x18, 136(sp)\n"                                                          \
  "sd x19, 144(sp)\n"                                                          \
  "sd x20, 152(sp)\n"                                                          \
  "sd x21, 160(sp)\n"                                                          \
  "sd x22, 168(sp)\n"                                                          \
  "sd x23, 176(sp)\n"                                                          \
  "sd x24, 184(sp)\n"                                                          \
  "sd x25, 192(sp)\n"                                                          \
  "sd x26, 200(sp)\n"                                                          \
  "sd x27, 208(sp)\n"                                                          \
  "sd x28, 216(sp)\n"                                                          \
  "sd x29, 224(sp)\n"                                                          \
  "sd x30, 232(sp)\n"                                                          \
  "sd x31, 240(sp)\n"
// Mirrors SAVE_ALL
#define RESTORE_ALL                                                            \
  "ld x1, 0(sp)\n"                                                             \
  "ld x2, 8(sp)\n"                                                             \
  "ld x3, 16(sp)\n"                                                            \
  "ld x4, 24(sp)\n"                                                            \
  "ld x5, 32(sp)\n"                                                            \
  "ld x6, 40(sp)\n"                                                            \
  "ld x7, 48(sp)\n"                                                            \
  "ld x8, 56(sp)\n"                                                            \
  "ld x9, 64(sp)\n"                                                            \
  "ld x10, 72(sp)\n"                                                           \
  "ld x11, 80(sp)\n"                                                           \
  "ld x12, 88(sp)\n"                                                           \
  "ld x13, 96(sp)\n"                                                           \
  "ld x14, 104(sp)\n"                                                          \
  "ld x15, 112(sp)\n"                                                          \
  "ld x16, 120(sp)\n"                                                          \
  "ld x17, 128(sp)\n"                                                          \
  "ld x18, 136(sp)\n"                                                          \
  "ld x19, 144(sp)\n"                                                          \
  "ld x20, 152(sp)\n"                                                          \
  "ld x21, 160(sp)\n"                                                          \
  "ld x22, 168(sp)\n"                                                          \
  "ld x23, 176(sp)\n"                                                          \
  "ld x24, 184(sp)\n"                                                          \
  "ld x25, 192(sp)\n"                                                          \
  "ld x26, 200(sp)\n"                                                          \
  "ld x27, 208(sp)\n"                                                          \
  "ld x28, 216(sp)\n"                                                          \
  "ld x29, 224(sp)\n"                                                          \
  "ld x30, 232(sp)\n"                                                          \
  "ld x31, 240(sp)\n"                                                          \
  "addi sp, sp,  256\n"

// https://github.com/torvalds/linux/blob/v7.1/scripts/syscall.tbl
// scripts/syscalltbl.sh --abis common,64 scripts/syscall.tbl riscv64_syscalls.h
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
  __asm__ volatile("lla %0, __hot_end\n\t"
                   "lui %1, %%hi(__hot_end)\n\t"
                   "addi %1, %1, %%lo(__hot_end)\n\t"
                   : "=r"(DynAddr), "=r"(StaticAddr));
  return DynAddr - StaticAddr;
}

} // anonymous namespace

#endif /* LLVM_TOOLS_LLVM_BOLT_SYS_RISCV */
