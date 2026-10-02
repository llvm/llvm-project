//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration test for secure_getenv.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_auxv_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/stdlib/secure_getenv.h"

#include "test/IntegrationTest/test.h"

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv(nullptr), nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv(""), nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv("="), nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv("MISSING ENV VARIABLE"), nullptr);
  EXPECT_NE(LIBC_NAMESPACE::secure_getenv("PATH"), nullptr);
  EXPECT_STREQ(LIBC_NAMESPACE::secure_getenv("FRANCE"), "Paris");
  EXPECT_STREQ(LIBC_NAMESPACE::secure_getenv("GERMANY"), "Berlin");
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv("FRANC"), nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv("FRANCE1"), nullptr);

  constexpr LIBC_NAMESPACE::auxv::Entry SECURE_AUXV[] = {
      {AT_SECURE, 1},
      {AT_NULL, AT_NULL},
  };
  constexpr LIBC_NAMESPACE::auxv::Entry NORMAL_AUXV[] = {
      {AT_SECURE, 0},
      {AT_NULL, AT_NULL},
  };
  constexpr LIBC_NAMESPACE::auxv::Entry EMPTY_AUXV[] = {
      {AT_NULL, AT_NULL},
  };

  LIBC_NAMESPACE::auxv::Vector::initialize_unsafe(SECURE_AUXV);
  LIBC_NAMESPACE::cpp::scope_exit restore_auxv(
      [&] { LIBC_NAMESPACE::auxv::Vector::initialize_unsafe(NORMAL_AUXV); });

  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv("FRANCE"), nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv("GERMANY"), nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv("PATH"), nullptr);

  LIBC_NAMESPACE::auxv::Vector::initialize_unsafe(EMPTY_AUXV);
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv("FRANCE"), nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::secure_getenv("PATH"), nullptr);

  LIBC_NAMESPACE::auxv::Vector::initialize_unsafe(NORMAL_AUXV);
  EXPECT_STREQ(LIBC_NAMESPACE::secure_getenv("FRANCE"), "Paris");

  return 0;
}
