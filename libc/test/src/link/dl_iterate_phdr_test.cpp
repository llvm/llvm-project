//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "hdr/types/size_t.h"
#include "src/link/dl_iterate_phdr.h"
#include "test/UnitTest/Test.h"

int SaveReturn1(struct dl_phdr_info *info, [[maybe_unused]] size_t info_size,
                void *arg) {
  *static_cast<void **>(arg) = info;
  return 1;
}

TEST(LlvmLibcLinkDlIteratePhdrTest, OnlyExecutable) {
  struct dl_phdr_info executable_info;
  EXPECT_EQ(LIBC_NAMESPACE::dl_iterate_phdr(SaveReturn1, &executable_info), 1);
  int program_header_count = executable_info.dlpi_phnum;
  EXPECT_GT(program_header_count, 0);
}

int SaveReturn0(struct dl_phdr_info *info, [[maybe_unused]] size_t info_size,
                void *arg) {
  *static_cast<void **>(arg) = info;
  return 0;
}

TEST(LlvmLibcLinkDlIteratePhdrTest, BothExecutableAndVDSO) {
  struct dl_phdr_info executable_info;
  EXPECT_EQ(LIBC_NAMESPACE::dl_iterate_phdr(SaveReturn0, &executable_info), 0);
  int program_header_count = executable_info.dlpi_phnum;
  EXPECT_GT(program_header_count, 0);
}
