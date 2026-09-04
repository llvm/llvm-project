//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/
///
/// \file
/// Tests for the dl_iterate_phdr implementation.
///
//===----------------------------------------------------------------------===/

#include "hdr/types/size_t.h"
#include "src/link/dl_iterate_phdr.h"
#include "test/UnitTest/Test.h"

#include <elf.h>

extern "C" void *__ehdr_start;

int save_return_1(struct dl_phdr_info *info, [[maybe_unused]] size_t info_size,
                  void *arg) {
  *static_cast<int *>(arg) = info->dlpi_phnum;
  return 1;
}

TEST(LlvmLibcLinkDlIteratePhdrTest, OnlyExecutable) {
  int program_header_count = 0;
  EXPECT_EQ(
      LIBC_NAMESPACE::dl_iterate_phdr(save_return_1, &program_header_count), 1);
  EXPECT_GT(program_header_count, 0);
}

int save_return_0(struct dl_phdr_info *info, [[maybe_unused]] size_t info_size,
                  void *arg) {
  *static_cast<int *>(arg) = info->dlpi_phnum;
  return 0;
}

TEST(LlvmLibcLinkDlIteratePhdrTest, BothExecutableAndVDSO) {
  int program_header_count = 0;
  EXPECT_EQ(
      LIBC_NAMESPACE::dl_iterate_phdr(save_return_0, &program_header_count), 0);
  EXPECT_GT(program_header_count, 0);
}

// dlpi_addr must be the load bias, so that dlpi_addr + p_vaddr gives the
// runtime address of a segment. Previously it was hardcoded to zero, which is
// only correct for a non-PIE executable.
struct BiasCheck {
  bool saw_executable = false;
  bool executable_bias_correct = false;
  bool all_names_non_null = true;
};

int check_bias(struct dl_phdr_info *info, [[maybe_unused]] size_t info_size,
               void *arg) {
  BiasCheck *state = static_cast<BiasCheck *>(arg);
  if (info->dlpi_name == nullptr) {
    state->all_names_non_null = false;
    return 0;
  }
  // The first module reported is the main executable, named "".
  if (!state->saw_executable) {
    state->saw_executable = true;
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
      if (info->dlpi_phdr[i].p_type == PT_LOAD) {
        ElfW(Addr) runtime = info->dlpi_addr + info->dlpi_phdr[i].p_vaddr;
        state->executable_bias_correct =
            runtime == reinterpret_cast<ElfW(Addr)>(&__ehdr_start);
        break;
      }
    }
  }
  return 0;
}

TEST(LlvmLibcLinkDlIteratePhdrTest, LoadBiasIsCorrect) {
  BiasCheck state;
  EXPECT_EQ(LIBC_NAMESPACE::dl_iterate_phdr(check_bias, &state), 0);
  EXPECT_TRUE(state.saw_executable);
  EXPECT_TRUE(state.executable_bias_correct);
  EXPECT_TRUE(state.all_names_non_null);
}
