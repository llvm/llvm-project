//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sys/procfs.h types.
///
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/sys-procfs-macros.h"
#include "include/llvm-libc-types/elf_fpregset_t.h"
#include "include/llvm-libc-types/elf_gregset_t.h"
#include "include/llvm-libc-types/struct_elf_prpsinfo.h"
#include "include/llvm-libc-types/struct_elf_prstatus.h"
#include "include/llvm-libc-types/struct_user_regs_struct.h"
#include "test/UnitTest/Test.h"

// These layouts are written to core dump notes by the kernel, so they are
// checked against the sizes of the kernel's structures.
#if defined(__x86_64__)
static_assert(ELF_NGREG == 27);
static_assert(sizeof(elf_fpregset_t) == 512);
static_assert(__builtin_offsetof(struct elf_prstatus, pr_reg) == 112);
static_assert(sizeof(struct elf_prstatus) == 336);
static_assert(sizeof(struct elf_prpsinfo) == 136);
#elif defined(__aarch64__)
static_assert(ELF_NGREG == 34);
static_assert(sizeof(elf_fpregset_t) == 528);
static_assert(__builtin_offsetof(struct elf_prstatus, pr_reg) == 112);
static_assert(sizeof(struct elf_prstatus) == 392);
static_assert(sizeof(struct elf_prpsinfo) == 136);
#endif

TEST(LlvmLibcSysProcfsTest, GregsetHoldsUserRegs) {
  ASSERT_EQ(sizeof(elf_gregset_t), sizeof(struct user_regs_struct));
}
