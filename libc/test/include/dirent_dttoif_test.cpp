//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests that DTTOIF and IFTODT translate between the d_type of a directory
/// entry and the file type bits of st_mode.
///
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "include/dirent.h"
#include "include/sys/stat.h"

// The translation is checked at compile time, so nothing here has to run.
#define CHECK_TYPE(DT, IF)                                                     \
  static_assert(DTTOIF(DT) == (IF), "DTTOIF(" #DT ") is not " #IF);            \
  static_assert(IFTODT(IF) == (DT), "IFTODT(" #IF ") is not " #DT);            \
  static_assert(IFTODT((IF) | 0777) == (DT),                                   \
                "IFTODT ignores the permission bits of " #IF)

CHECK_TYPE(DT_FIFO, S_IFIFO);
CHECK_TYPE(DT_CHR, S_IFCHR);
CHECK_TYPE(DT_DIR, S_IFDIR);
CHECK_TYPE(DT_BLK, S_IFBLK);
CHECK_TYPE(DT_REG, S_IFREG);
CHECK_TYPE(DT_LNK, S_IFLNK);
CHECK_TYPE(DT_SOCK, S_IFSOCK);

TEST(LlvmLibcDirentDttoifTest, TypeBitsTranslateAtCompileTime) {
  // The checks above are static_asserts; this only gives the test a body.
}
