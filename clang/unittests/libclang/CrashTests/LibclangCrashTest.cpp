//===- unittests/libclang/LibclangCrashTest.cpp --- libclang tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TestUtils.h"
#include "clang-c/FatalErrorHandler.h"
#include "gtest/gtest.h"
#include <string>

TEST_F(LibclangParseTest, InstallAbortingLLVMFatalErrorHandler) {
  // gtest death-tests execute in a sub-process (fork), which invalidates
  // any signpost handles and would cause spurious crashes if used. Use the
  // "threadsafe" style of death-test to work around this.
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  clang_toggleCrashRecovery(0);
  clang_install_aborting_llvm_fatal_error_handler();

  std::string Main = "main.h";
  WriteFile(Main, "#pragma clang __debug llvm_fatal_error");

  EXPECT_DEATH(clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0,
                                          nullptr, 0, TUFlags),
               "");
}

TEST_F(LibclangParseTest, UninstallAbortingLLVMFatalErrorHandler) {
  // gtest death-tests execute in a sub-process (fork), which invalidates
  // any signpost handles and would cause spurious crashes if used. Use the
  // "threadsafe" style of death-test to work around this.
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  clang_toggleCrashRecovery(0);
  clang_install_aborting_llvm_fatal_error_handler();
  clang_uninstall_llvm_fatal_error_handler();

  std::string Main = "main.h";
  WriteFile(Main, "#pragma clang __debug llvm_fatal_error");

  EXPECT_DEATH(clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0,
                                          nullptr, 0, TUFlags),
               "ERROR");
}
