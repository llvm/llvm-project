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
  clang_toggleCrashRecovery(0);
  clang_install_aborting_llvm_fatal_error_handler();

  std::string Main = "main.h";
  WriteFile(Main, "#pragma clang __debug llvm_fatal_error");

  EXPECT_DEATH(clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0,
                                          nullptr, 0, TUFlags),
               "");
}

TEST_F(LibclangParseTest, UninstallAbortingLLVMFatalErrorHandler) {
  clang_toggleCrashRecovery(0);
  clang_install_aborting_llvm_fatal_error_handler();
  clang_uninstall_llvm_fatal_error_handler();

  std::string Main = "main.h";
  WriteFile(Main, "#pragma clang __debug llvm_fatal_error");

  EXPECT_EXIT(clang_parseTranslationUnit(
      Index, Main.c_str(), nullptr, 0, nullptr, 0, TUFlags),
      ::testing::ExitedWithCode(1), "ERROR");
}
