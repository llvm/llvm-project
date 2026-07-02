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
