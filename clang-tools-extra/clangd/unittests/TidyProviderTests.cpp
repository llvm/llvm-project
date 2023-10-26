//===-- TidyProviderTests.cpp - Clang tidy configuration provider tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Feature.h"
#include "TestFS.h"
#include "TidyProvider.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

namespace {

TEST(TidyProvider, NestedDirectories) {
  MockFS FS;
  FS.Files[testPath(".clang-tidy")] = R"yaml(
  Checks: 'llvm-*'
  CheckOptions:
    TestKey: 1
)yaml";
  FS.Files[testPath("sub1/.clang-tidy")] = R"yaml(
  Checks: 'misc-*'
  CheckOptions:
    TestKey: 2
)yaml";
  FS.Files[testPath("sub1/sub2/.clang-tidy")] = R"yaml(
  Checks: 'bugprone-*'
  CheckOptions:
    TestKey: 3
  InheritParentConfig: true
)yaml";

  TidyProvider Provider = provideClangTidyFiles(FS);

  auto BaseOptions = getTidyOptionsForFile(Provider, testPath("File.cpp"));
  ASSERT_TRUE(BaseOptions.Checks.has_value());
  EXPECT_EQ(*BaseOptions.Checks, "llvm-*");
  EXPECT_EQ(BaseOptions.CheckOptions.lookup("TestKey").Value, "1");

  auto Sub1Options = getTidyOptionsForFile(Provider, testPath("sub1/File.cpp"));
  ASSERT_TRUE(Sub1Options.Checks.has_value());
  EXPECT_EQ(*Sub1Options.Checks, "misc-*");
  EXPECT_EQ(Sub1Options.CheckOptions.lookup("TestKey").Value, "2");

  auto Sub2Options =
      getTidyOptionsForFile(Provider, testPath("sub1/sub2/File.cpp"));
  ASSERT_TRUE(Sub2Options.Checks.has_value());
  EXPECT_EQ(*Sub2Options.Checks, "misc-*,bugprone-*");
  EXPECT_EQ(Sub2Options.CheckOptions.lookup("TestKey").Value, "3");
}

TEST(TidyProvider, IsFastTidyCheck) {
  EXPECT_THAT(isFastTidyCheck("misc-const-correctness"), llvm::ValueIs(false));
  EXPECT_THAT(isFastTidyCheck("bugprone-suspicious-include"),
              llvm::ValueIs(true));
  // Linked in (ParsedASTTests.cpp) but not measured.
  EXPECT_EQ(isFastTidyCheck("replay-preamble-check"), std::nullopt);
}

#if CLANGD_TIDY_CHECKS
TEST(TidyProvider, IsValidCheck) {
  EXPECT_TRUE(isRegisteredTidyCheck("bugprone-argument-comment"));
  EXPECT_FALSE(isRegisteredTidyCheck("bugprone-argument-clinic"));
}
#endif

} // namespace
} // namespace clangd
} // namespace clang
