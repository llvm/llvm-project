//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared test utilities and fixtures for pwd tests.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_PWD_PWD_TEST_UTILS_H
#define LLVM_LIBC_TEST_SRC_PWD_PWD_TEST_UTILS_H

#include "hdr/types/size_t.h"
#include "src/__support/File/file.h"
#include "src/pwd/pwd_utils.h"
#include "src/stdio/remove.h"
#include "src/string/string_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

// RAII helper class for creating and automatically removing temporary test
// files, while safely scoping the password database path.
class ScopedPasswdFile {
  char path[256];

public:
  ScopedPasswdFile(const char *file_path, const char *content) {
    LIBC_NAMESPACE::internal::strlcpy(path, file_path, sizeof(path));

    auto file_or = LIBC_NAMESPACE::openfile(path, "w");
    if (file_or.has_value()) {
      auto *f = file_or.value();
      size_t len = LIBC_NAMESPACE::internal::string_length(content);
      f->write(content, len);
      f->close();
    }
    LIBC_NAMESPACE::pwd::TESTONLY_set_passwd_path(path);
  }

  ~ScopedPasswdFile() {
    LIBC_NAMESPACE::pwd::TESTONLY_reset_passwd_path();
    LIBC_NAMESPACE::remove(path);
  }

  const char *get_path() const { return path; }
};

// Base test fixture that resets the password database path and validates errno.
class LlvmLibcPwdTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  void TearDown() override {
    LIBC_NAMESPACE::pwd::TESTONLY_reset_passwd_path();
    ErrnoCheckingTest::TearDown();
  }
};

#endif // LLVM_LIBC_TEST_SRC_PWD_PWD_TEST_UTILS_H
