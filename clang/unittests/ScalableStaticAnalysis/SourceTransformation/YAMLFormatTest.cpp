//===- YAMLFormatTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/YAMLSourceEditFormat.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace ssaf;

namespace {

// Materializes a unique temporary file path under the system temp dir and
// removes it on destruction.
struct TempPath {
  SmallString<128> Path;

  TempPath(StringRef Suffix) {
    sys::fs::createUniquePath("ssaf-yaml-%%%%%%." + Suffix, Path,
                              /*MakeAbsolute=*/true);
  }
  ~TempPath() { sys::fs::remove(Path); }
};

TEST(WriteYAMLSourceEditsTest, RoundTripsTwoReplacements) {
  clang::tooling::TranslationUnitReplacements Doc;
  Doc.MainSourceFile = "main.cpp";
  Doc.Replacements.emplace_back("a.cpp", 0, 0, "/*1*/");
  Doc.Replacements.emplace_back("b.cpp", 10, 3, "/*2*/");

  TempPath TP("yaml");
  ASSERT_THAT_ERROR(writeYAMLSourceEdits(Doc, TP.Path), Succeeded());

  auto BufferOrErr = MemoryBuffer::getFile(TP.Path);
  ASSERT_TRUE(static_cast<bool>(BufferOrErr))
      << "Failed to read back '" << TP.Path << "'";

  clang::tooling::TranslationUnitReplacements Parsed;
  yaml::Input YIn((*BufferOrErr)->getBuffer());
  YIn >> Parsed;
  ASSERT_FALSE(YIn.error()) << YIn.error().message();

  EXPECT_EQ(Parsed.MainSourceFile, "main.cpp");
  ASSERT_EQ(Parsed.Replacements.size(), 2u);
  EXPECT_EQ(Parsed.Replacements[0].getFilePath(), "a.cpp");
  EXPECT_EQ(Parsed.Replacements[0].getOffset(), 0u);
  EXPECT_EQ(Parsed.Replacements[0].getLength(), 0u);
  EXPECT_EQ(Parsed.Replacements[0].getReplacementText(), "/*1*/");
  EXPECT_EQ(Parsed.Replacements[1].getFilePath(), "b.cpp");
  EXPECT_EQ(Parsed.Replacements[1].getOffset(), 10u);
  EXPECT_EQ(Parsed.Replacements[1].getLength(), 3u);
  EXPECT_EQ(Parsed.Replacements[1].getReplacementText(), "/*2*/");
}

TEST(WriteYAMLSourceEditsTest, EmptyReplacementsWritesValidDocument) {
  clang::tooling::TranslationUnitReplacements Doc;
  Doc.MainSourceFile = "main.cpp";

  TempPath TP("yaml");
  ASSERT_THAT_ERROR(writeYAMLSourceEdits(Doc, TP.Path), Succeeded());

  auto BufferOrErr = MemoryBuffer::getFile(TP.Path);
  ASSERT_TRUE(static_cast<bool>(BufferOrErr));

  clang::tooling::TranslationUnitReplacements Parsed;
  yaml::Input YIn((*BufferOrErr)->getBuffer());
  YIn >> Parsed;
  ASSERT_FALSE(YIn.error()) << YIn.error().message();
  EXPECT_EQ(Parsed.MainSourceFile, "main.cpp");
  EXPECT_TRUE(Parsed.Replacements.empty());
}

TEST(WriteYAMLSourceEditsTest, OpenErrorReturnsError) {
  clang::tooling::TranslationUnitReplacements Doc;
  Doc.MainSourceFile = "main.cpp";

  // Path under a directory that does not exist.
  SmallString<128> BadPath;
  sys::fs::createUniquePath("ssaf-missing-%%%%%%/edits.yaml", BadPath,
                            /*MakeAbsolute=*/true);

  ASSERT_THAT_ERROR(writeYAMLSourceEdits(Doc, BadPath), Failed());
}

} // namespace
