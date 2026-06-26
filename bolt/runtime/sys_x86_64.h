//===- bolt/runtime/sys_x86_64.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_SYS_X86_64
#define LLVM_TOOLS_LLVM_BOLT_SYS_X86_64

#include "runtime_types.h"

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                                                               \
  "push %%rax\n"                                                               \
  "push %%rbx\n"                                                               \
  "push %%rcx\n"                                                               \
  "push %%rdx\n"                                                               \
  "push %%rdi\n"                                                               \
  "push %%rsi\n"                                                               \
  "push %%rbp\n"                                                               \
  "push %%r8\n"                                                                \
  "push %%r9\n"                                                                \
  "push %%r10\n"                                                               \
  "push %%r11\n"                                                               \
  "push %%r12\n"                                                               \
  "push %%r13\n"                                                               \
  "push %%r14\n"                                                               \
  "push %%r15\n"                                                               \
  "sub $8, %%rsp\n"
// Mirrors SAVE_ALL
#define RESTORE_ALL                                                            \
  "add $8, %%rsp\n"                                                            \
  "pop %%r15\n"                                                                \
  "pop %%r14\n"                                                                \
  "pop %%r13\n"                                                                \
  "pop %%r12\n"                                                                \
  "pop %%r11\n"                                                                \
  "pop %%r10\n"                                                                \
  "pop %%r9\n"                                                                 \
  "pop %%r8\n"                                                                 \
  "pop %%rbp\n"                                                                \
  "pop %%rsi\n"                                                                \
  "pop %%rdi\n"                                                                \
  "pop %%rdx\n"                                                                \
  "pop %%rcx\n"                                                                \
  "pop %%rbx\n"                                                                \
  "pop %%rax\n"

#if defined(__APPLE__)
#define __NR_exit 0x2000001
#define __NR_read 0x2000003
#define __NR_write 0x2000004
#define __NR_sigprocmask 0x2000030
#define __NR_munmap 0x2000049
#define __NR_mmap 0x20000c5
#define __NR_getpid 20
#else
// scripts/syscalltbl.sh --abis common,64 arch/x86/entry/syscalls/syscall_64.tbl
// x86_64_syscalls.h
#define __NR_read 0
#define __NR_write 1
#define __NR_close 3
#define __NR_lseek 8
#define __NR_mmap 9
#define __NR_mprotect 10
#define __NR_munmap 11
#define __NR_rt_sigprocmask 14
#define __NR_madvise 28
#define __NR_nanosleep 35
#define __NR_getpid 39
#define __NR_clone 56
#define __NR_fork 57
#define __NR_exit 60
#define __NR_kill 62
#define __NR_uname 63
#define __NR_fsync 74
#define __NR_ftruncate 77
#define __NR_setpgid 109
#define __NR_getppid 110
#define __NR_getpgid 121
#define __NR_prctl 157
#define __NR_getdents64 217
#define __NR_exit_group 231
#define __NR_openat 257
#define __NR_readlinkat 267
#endif

namespace {

// Get the difference between runtime address of .text section and
// static address in section header table. Can be extracted from arbitrary
// pc value recorded at runtime to get the corresponding static address, which
// in turn can be used to search for indirect call description. Needed because
// indirect call descriptions are read-only non-relocatable data.
uint64_t getTextBaseAddress() {
  uint64_t DynAddr;
  uint64_t StaticAddr;
  __asm__ volatile("leaq __hot_end(%%rip), %0\n\t"
                   "movabsq $__hot_end, %1\n\t"
                   : "=r"(DynAddr), "=r"(StaticAddr));
  return DynAddr - StaticAddr;
}

} // anonymous namespace

#endif /* LLVM_TOOLS_LLVM_BOLT_SYS_X86_64 */
