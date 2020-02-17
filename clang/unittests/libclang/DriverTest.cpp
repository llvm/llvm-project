//===---- DriverTest.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Driver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "driver-test"

TEST(DriverTests, Basic) {
  const char *ArgV[] = {"clang", "-w", "t.cpp", "-o", "t.ll"};

  CXExternalActionList *EAL = clang_Driver_getExternalActionsForCommand_v0(
      GTEST_ARRAY_SIZE_(ArgV), ArgV, nullptr, nullptr, nullptr);
  ASSERT_NE(EAL, nullptr);
  ASSERT_EQ(EAL->Count, 2);
  auto *CompileAction = EAL->Actions[0];
  ASSERT_GE(CompileAction->ArgC, 2);
  EXPECT_STREQ(CompileAction->ArgV[0], "clang");
  EXPECT_STREQ(CompileAction->ArgV[1], "-cc1");

  clang_Driver_ExternalActionList_dispose(EAL);
}

TEST(DriverTests, WorkingDirectory) {
  const char *ArgV[] = {"clang", "-c", "t.cpp", "-o", "t.o"};

  CXExternalActionList *EAL = clang_Driver_getExternalActionsForCommand_v0(
      GTEST_ARRAY_SIZE_(ArgV), ArgV, nullptr, "/", nullptr);
  ASSERT_NE(EAL, nullptr);
  ASSERT_EQ(EAL->Count, 1);
  auto *CompileAction = EAL->Actions[0];

  const char **FDCD = std::find(CompileAction->ArgV, CompileAction->ArgV +
                                                     CompileAction->ArgC,
                                llvm::StringRef("-fdebug-compilation-dir"));
  ASSERT_NE(FDCD, CompileAction->ArgV + CompileAction->ArgC);
  ASSERT_NE(FDCD + 1, CompileAction->ArgV + CompileAction->ArgC);
  EXPECT_STREQ(*(FDCD + 1), "/");

  clang_Driver_ExternalActionList_dispose(EAL);
}

TEST(DriverTests, Diagnostics) {
  const char *ArgV[] = {"clang", "-c", "nosuchfile.cpp", "-o", "t.o"};

  CXExternalActionList *EAL = clang_Driver_getExternalActionsForCommand_v0(
    GTEST_ARRAY_SIZE_(ArgV), ArgV, nullptr, "/no/such/working/dir", nullptr);
  EXPECT_EQ(nullptr, EAL);
  clang_Driver_ExternalActionList_dispose(EAL);

  CXDiagnosticSet Diags;
  EAL = clang_Driver_getExternalActionsForCommand_v0(
    GTEST_ARRAY_SIZE_(ArgV), ArgV, nullptr, "/no/such/working/dir", &Diags);
  EXPECT_EQ(nullptr, EAL);
  ASSERT_NE(nullptr, Diags);

  unsigned NumDiags = clang_getNumDiagnosticsInSet(Diags);
  ASSERT_EQ(1u, NumDiags);
  CXDiagnostic Diag = clang_getDiagnosticInSet(Diags, 0);
  CXString Str = clang_formatDiagnostic(Diag, 0);
  EXPECT_STREQ(clang_getCString(Str),
               "error: unable to set working directory: /no/such/working/dir");
  clang_disposeString(Str);

  clang_disposeDiagnosticSet(Diags);
  clang_Driver_ExternalActionList_dispose(EAL);
}
