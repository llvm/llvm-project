//===-- DefineOutline.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "TestTU.h"
#include "TweakTesting.h"
#include "URI.h"
#include "index/MemIndex.h"
#include "gtest/gtest.h"

namespace clang::clangd {
namespace {

TWEAK_TEST(ScopifyEnum);

TEST_F(ScopifyEnumTest, ReferencesUseDriveAliases) {
  if (!hasWindowsDrive(testRoot()))
    GTEST_SKIP() << "Requires Windows paths";
  FileName = "Test.hpp";
  TestTU TU = TestTU::withHeaderCode("enum E { EV1, EV2 };");
  TU.HeaderFilename = FileName.str();
  auto Symbols = TU.headerSymbols();
  // Stay outside testRoot: the unittest URI scheme canonicalizes its drive.
  std::string OtherPath = "C:/drive-alias/other.cpp";
  std::string URI = clangd::URI::createFile(OtherPath).toString();
  OtherPath[0] = 'c';
  std::string AliasURI = clangd::URI::createFile(OtherPath).toString();
  RefSlab::Builder Refs;
  for (const auto &S : Symbols) {
    if (S.Name != "EV1" && S.Name != "EV2")
      continue;
    bool First = S.Name == "EV1";
    Ref R;
    R.Kind = RefKind::Reference | RefKind::Spelled;
    R.Location.FileURI = (First ? URI : AliasURI).c_str();
    R.Location.Start.setLine(First ? 0 : 1);
    R.Location.Start.setColumn(8);
    R.Location.End.setLine(First ? 0 : 1);
    R.Location.End.setColumn(11);
    Refs.insert(S.ID, R);
  }
  Index = MemIndex::build({}, std::move(Refs).build(), {});
  ExtraFiles["C:/drive-alias/other.cpp"] = "int a = EV1;\nint b = EV2;\n";
  llvm::StringMap<std::string> EditedFiles;
  EXPECT_EQ(apply("enum ^E { EV1, EV2 };", &EditedFiles),
            "enum class E { V1, V2 };");
  EXPECT_THAT(EditedFiles,
              ::testing::UnorderedElementsAre(FileWithContents(
                  llvm::sys::path::native("C:/drive-alias/other.cpp"),
                  "int a = E::V1;\nint b = E::V2;\n")));
}

TEST_F(ScopifyEnumTest, TriggersOnUnscopedEnumDecl) {
  FileName = "Test.hpp";
  // Not available for scoped enum.
  EXPECT_UNAVAILABLE(R"cpp(enum class ^E { V };)cpp");

  // Not available for non-definition.
  EXPECT_UNAVAILABLE(R"cpp(
enum E { V };
enum ^E;
)cpp");
}

TEST_F(ScopifyEnumTest, ApplyTestWithPrefix) {
  std::string Original = R"cpp(
enum ^E { EV1, EV2, EV3 };
enum E;
E func(E in)
{
  E out = EV1;
  if (in == EV2)
    out = E::EV3;
  return out;
}
)cpp";
  std::string Expected = R"cpp(
enum class E { V1, V2, V3 };
enum class E;
E func(E in)
{
  E out = E::V1;
  if (in == E::V2)
    out = E::V3;
  return out;
}
)cpp";
  FileName = "Test.cpp";
  SCOPED_TRACE(Original);
  EXPECT_EQ(apply(Original), Expected);
}

TEST_F(ScopifyEnumTest, ApplyTestWithPrefixAndUnderscore) {
  std::string Original = R"cpp(
enum ^E { E_V1, E_V2, E_V3 };
enum E;
E func(E in)
{
  E out = E_V1;
  if (in == E_V2)
    out = E::E_V3;
  return out;
}
)cpp";
  std::string Expected = R"cpp(
enum class E { V1, V2, V3 };
enum class E;
E func(E in)
{
  E out = E::V1;
  if (in == E::V2)
    out = E::V3;
  return out;
}
)cpp";
  FileName = "Test.cpp";
  SCOPED_TRACE(Original);
  EXPECT_EQ(apply(Original), Expected);
}

TEST_F(ScopifyEnumTest, ApplyTestWithoutPrefix) {
  std::string Original = R"cpp(
enum ^E { V1, V2, V3 };
enum E;
E func(E in)
{
  E out = V1;
  if (in == V2)
    out = E::V3;
  return out;
}
)cpp";
  std::string Expected = R"cpp(
enum class E { V1, V2, V3 };
enum class E;
E func(E in)
{
  E out = E::V1;
  if (in == E::V2)
    out = E::V3;
  return out;
}
)cpp";
  FileName = "Test.cpp";
  SCOPED_TRACE(Original);
  EXPECT_EQ(apply(Original), Expected);
}

} // namespace
} // namespace clang::clangd
