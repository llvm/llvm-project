//===--- IncludeSpellerTest.cpp--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/IncludeSpeller.h"
#include "clang-include-cleaner/Analysis.h"
#include "clang-include-cleaner/Types.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Testing/TestAST.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"
#include <assert.h>
#include <string>
namespace clang::include_cleaner {
namespace {

const char *testRoot() {
#ifdef _WIN32
  return "C:\\include-cleaner-test";
#else
  return "/include-cleaner-test";
#endif
}

std::string testPath(llvm::StringRef File) {
  assert(llvm::sys::path::is_relative(File) && "FileName should be relative");

  llvm::SmallString<32> NativeFile = File;
  llvm::sys::path::native(NativeFile, llvm::sys::path::Style::native);
  llvm::SmallString<32> Path;
  llvm::sys::path::append(Path, llvm::sys::path::Style::native, testRoot(),
                          NativeFile);
  return std::string(Path.str());
}

class DummyIncludeSpeller : public IncludeSpeller {
public:
  std::string operator()(const IncludeSpeller::Input &Input) const override {
    if (Input.H.kind() == Header::Standard)
      return "<bits/stdc++.h>";
    if (Input.H.kind() != Header::Physical)
      return "";
    llvm::StringRef AbsolutePath =
        Input.H.physical().getFileEntry().tryGetRealPathName();
    std::string RootWithSeparator{testRoot()};
    RootWithSeparator += llvm::sys::path::get_separator();
    if (!AbsolutePath.consume_front(llvm::StringRef{RootWithSeparator}))
      return "";
    return "\"" + AbsolutePath.str() + "\"";
  }
};

TEST(IncludeSpeller, IsRelativeToTestRoot) {
  TestInputs Inputs;

  Inputs.ExtraArgs.push_back("-isystemdir");

  Inputs.ExtraFiles[testPath("foo.h")] = "";
  Inputs.ExtraFiles["dir/header.h"] = "";
  TestAST AST{Inputs};

  auto &FM = AST.fileManager();
  auto &HS = AST.preprocessor().getHeaderSearchInfo();
  const auto *MainFile = AST.sourceManager().getFileEntryForID(
      AST.sourceManager().getMainFileID());

  EXPECT_EQ("\"foo.h\"",
            spellHeader({Header{*FM.getOptionalFileRef(testPath("foo.h"))}, HS,
                         MainFile}));
  EXPECT_EQ("<header.h>",
            spellHeader({Header{*FM.getOptionalFileRef("dir/header.h")}, HS,
                         MainFile}));
}

TEST(IncludeSpeller, CanOverrideSystemHeaders) {
  TestAST AST("");
  auto &HS = AST.preprocessor().getHeaderSearchInfo();
  const auto *MainFile = AST.sourceManager().getFileEntryForID(
      AST.sourceManager().getMainFileID());
  EXPECT_EQ("<bits/stdc++.h>",
            spellHeader({Header{*tooling::stdlib::Header::named("<vector>")},
                         HS, MainFile}));
}

IncludeSpellingStrategy::Add<DummyIncludeSpeller>
    Speller("dummy", "Dummy Include Speller");

} // namespace
} // namespace clang::include_cleaner
