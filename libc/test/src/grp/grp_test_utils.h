//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared test utilities and fixtures for grp tests.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_GRP_GRP_TEST_UTILS_H
#define LLVM_LIBC_TEST_SRC_GRP_GRP_TEST_UTILS_H

#include "hdr/types/size_t.h"
#include "src/__support/File/file.h"
#include "src/grp/grp_utils.h"
#include "src/stdio/remove.h"
#include "src/string/string_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

// RAII helper class for creating and automatically removing temporary test
// files, while safely scoping the group database path.
class ScopedGroupFile {
  char path[256];

public:
  ScopedGroupFile(const ScopedGroupFile &) = delete;
  ScopedGroupFile &operator=(const ScopedGroupFile &) = delete;

  ScopedGroupFile(const char *file_path, const char *content) {
    LIBC_NAMESPACE::internal::strlcpy(path, file_path, sizeof(path));

    auto file_or = LIBC_NAMESPACE::openfile(path, "w");
    if (file_or.has_value()) {
      auto *f = file_or.value();
      size_t len = LIBC_NAMESPACE::internal::string_length(content);
      f->write(content, len);
      f->close();
    }
    LIBC_NAMESPACE::grp::TESTONLY_set_group_path(path);
  }

  ~ScopedGroupFile() {
    LIBC_NAMESPACE::grp::TESTONLY_reset_group_path();
    LIBC_NAMESPACE::remove(path);
  }
};

// Base test fixture that resets the group database path and validates errno.
class LlvmLibcGrpTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  void TearDown() override {
    LIBC_NAMESPACE::grp::TESTONLY_reset_group_path();
    ErrnoCheckingTest::TearDown();
  }
};

#endif // LLVM_LIBC_TEST_SRC_GRP_GRP_TEST_UTILS_H
