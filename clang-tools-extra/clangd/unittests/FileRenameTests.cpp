//===-- FileRenameTests.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Compiler.h"
#include "Diagnostics.h"
#include "FileRename.h"
#include "ParsedAST.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestTU.h"
#include "support/Logger.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <utility>

namespace clang {
namespace clangd {
namespace {
using testing::HasSubstr;

llvm::Expected<std::vector<TextEdit>>
editsFor(TestTU &TU, llvm::ArrayRef<std::pair<Path, Path>> Renames) {
  MockFS FS;
  auto Inputs = TU.inputs(FS);
  StoreDiags Diags;
  auto CI = buildCompilerInvocation(Inputs, Diags);
  if (!CI)
    return error("failed to build test compiler invocation");
  auto AST = ParsedAST::build(testPath(TU.Filename), Inputs, std::move(CI),
                              Diags.take(), /*Preamble=*/nullptr);
  if (!AST)
    return error("failed to build test AST");
  auto VFS = FS.view(std::nullopt);
  std::vector<FileRenameMapping> Mappings;
  for (const auto &[Old, New] : Renames) {
    auto OldStatus = VFS->status(Old);
    if (!OldStatus)
      return error("missing test input {0}", Old);
    Mappings.push_back({Old, New, OldStatus->getUniqueID()});
  }
  return renameIncludeDirectives(
      testPath(TU.Filename), TU.Code, AST->getIncludeStructure(),
      AST->getPreprocessor().getHeaderSearchInfo(),
      Inputs.CompileCommand.Directory, Mappings, format::getLLVMStyle(), *VFS);
}

llvm::Expected<std::vector<TextEdit>> editsFor(TestTU &TU, PathRef Old,
                                               PathRef New) {
  std::pair<Path, Path> Rename{Old.str(), New.str()};
  return editsFor(TU, {Rename});
}

TEST(FileRename, RenamesQuotedIncludeByResolvedIdentity) {
  TestTU TU;
  TU.Code = "#include \"old.h\"\n";
  TU.AdditionalFiles["old.h"] = "";
  auto Result = editsFor(TU, testPath("old.h"), testPath("new.h"));
  ASSERT_THAT_EXPECTED(Result, llvm::Succeeded());
  ASSERT_EQ(Result->size(), 1u);
  EXPECT_EQ(Result->front().newText, "\"new.h\"");
  EXPECT_EQ(Result->front().range, (Range{{0, 9}, {0, 16}}));
}

TEST(FileRename, DoesNotRenameAHeaderWithOnlyTheSameSpelling) {
  TestTU TU;
  TU.Filename = "source/main.cpp";
  TU.Code = "#include \"same.h\"\n";
  TU.AdditionalFiles["source/same.h"] = "";
  TU.AdditionalFiles["other/same.h"] = "";
  auto Result =
      editsFor(TU, testPath("other/same.h"), testPath("other/renamed.h"));
  ASSERT_THAT_EXPECTED(Result, llvm::Succeeded());
  EXPECT_THAT(*Result, testing::IsEmpty());
}

TEST(FileRename, RejectsMacroGeneratedIncludeOperand) {
  TestTU TU;
  TU.Code = "#define HEADER \"old.h\"\n#include HEADER\n";
  TU.AdditionalFiles["old.h"] = "";
  auto Result = editsFor(TU, testPath("old.h"), testPath("new.h"));
  ASSERT_THAT_EXPECTED(Result,
                       llvm::FailedWithMessage(HasSubstr("macro-generated")));
}

TEST(FileRename, RenamesAngledIncludeUsingHeaderSearch) {
  TestTU TU;
  TU.Code = "#include <old.h>\n";
  TU.AdditionalFiles["include/old.h"] = "";
  TU.ExtraArgs = {"-I", testPath("include")};
  auto Result =
      editsFor(TU, testPath("include/old.h"), testPath("include/new.h"));
  ASSERT_THAT_EXPECTED(Result, llvm::Succeeded());
  ASSERT_EQ(Result->size(), 1u);
  EXPECT_EQ(Result->front().newText, "<new.h>");
}

TEST(FileRename, RejectsAnIncludeThatWouldResolveToAnotherFile) {
  TestTU TU;
  TU.Code = "#include <old.h>\n";
  TU.AdditionalFiles["first/new.h"] = "";
  TU.AdditionalFiles["second/old.h"] = "";
  TU.ExtraArgs = {"-I", testPath("first"), "-I", testPath("second")};
  auto Result =
      editsFor(TU, testPath("second/old.h"), testPath("second/new.h"));
  EXPECT_THAT_EXPECTED(
      Result, llvm::FailedWithMessage(HasSubstr("would resolve to existing")));
}

TEST(FileRename, RenamesObjCImport) {
  TestTU TU;
  TU.Filename = "main.m";
  TU.Code = "#import \"old.h\"\n";
  TU.AdditionalFiles["old.h"] = "";
  auto Result = editsFor(TU, testPath("old.h"), testPath("new.h"));
  ASSERT_THAT_EXPECTED(Result, llvm::Succeeded());
  ASSERT_EQ(Result->size(), 1u);
  EXPECT_EQ(Result->front().newText, "\"new.h\"");
}

TEST(FileRename, RewritesIncludesRelativeToAMovedIncluder) {
  TestTU TU;
  TU.Filename = "old/main.cpp";
  TU.Code = "#include \"header.h\"\n";
  TU.AdditionalFiles["old/header.h"] = "";
  std::pair<Path, Path> Rename{testPath("old/main.cpp"),
                               testPath("new/main.cpp")};
  auto Result = editsFor(TU, {Rename});
  ASSERT_THAT_EXPECTED(Result, llvm::Succeeded());
  ASSERT_EQ(Result->size(), 1u);
  EXPECT_EQ(Result->front().newText, "\"../old/header.h\"");
}

TEST(FileRename, RejectsUnresolvedIncludeInMovedFile) {
  TestTU TU;
  TU.Filename = "old/main.cpp";
  TU.Code = "#include \"missing.h\" // error-ok\n";
  auto Result =
      editsFor(TU, testPath("old/main.cpp"), testPath("new/main.cpp"));
  EXPECT_THAT_EXPECTED(
      Result, llvm::FailedWithMessage(HasSubstr("unresolved include")));
}

TEST(FileRename, IgnoresUnresolvedIncludeOutsideRename) {
  TestTU TU;
  TU.Code = R"cpp(
#include "missing.h" // error-ok
#include "old.h"
)cpp";
  TU.AdditionalFiles["old.h"] = "";
  auto Result = editsFor(TU, testPath("old.h"), testPath("new.h"));
  ASSERT_THAT_EXPECTED(Result, llvm::Succeeded());
  ASSERT_EQ(Result->size(), 1u);
  EXPECT_EQ(Result->front().newText, "\"new.h\"");
}

TEST(FileRename, DirectoryRenameDelegatesToFileBatch) {
  TestTU TU;
  TU.Filename = "src/main.cpp";
  TU.Code = "#include \"../old/old.h\"\n";
  TU.AdditionalFiles["old/old.h"] = "";
  MockFS FS;
  auto Inputs = TU.inputs(FS);
  auto VFS = FS.view(std::nullopt);
  auto Mappings = expandFileRenames({{testPath("old"), testPath("renamed")}},
                                    testRoot(), *VFS);
  ASSERT_THAT_EXPECTED(Mappings, llvm::Succeeded());

  StoreDiags Diags;
  auto CI = buildCompilerInvocation(Inputs, Diags);
  ASSERT_TRUE(CI);
  auto AST = ParsedAST::build(testPath(TU.Filename), Inputs, std::move(CI),
                              Diags.take(), /*Preamble=*/nullptr);
  ASSERT_TRUE(AST);
  auto Result = renameIncludeDirectives(
      testPath(TU.Filename), TU.Code, AST->getIncludeStructure(),
      AST->getPreprocessor().getHeaderSearchInfo(),
      Inputs.CompileCommand.Directory, *Mappings, format::getLLVMStyle(), *VFS);
  ASSERT_THAT_EXPECTED(Result, llvm::Succeeded());
  ASSERT_EQ(Result->size(), 1u);
  EXPECT_EQ(Result->front().newText, "\"../renamed/old.h\"");
}

TEST(FileRename, ExpandsDirectoriesAndRejectsConflictingMappings) {
  MockFS FS;
  FS.Files[testPath("old/a.h")] = "";
  FS.Files[testPath("old/nested/b.h")] = "";
  auto VFS = FS.view(std::nullopt);
  auto Expanded =
      expandFileRenames({{testPath("old"), testPath("new")}}, testRoot(), *VFS);
  ASSERT_THAT_EXPECTED(Expanded, llvm::Succeeded());
  EXPECT_EQ(Expanded->size(), 2u);
  EXPECT_THAT(*Expanded, testing::UnorderedElementsAre(
                             testing::Field(&FileRenameMapping::NewPath,
                                            testPath("new/a.h")),
                             testing::Field(&FileRenameMapping::NewPath,
                                            testPath("new/nested/b.h"))));

  auto Conflict =
      expandFileRenames({{testPath("old/a.h"), testPath("new/a.h")},
                         {testPath("old/a.h"), testPath("other/a.h")}},
                        testRoot(), *VFS);
  EXPECT_THAT_EXPECTED(
      Conflict, llvm::FailedWithMessage(HasSubstr("same file is renamed")));
}

TEST(FileRename, NormalizesMultipleAndOverlappingMappings) {
  MockFS FS;
  FS.Files[testPath("old/a.h")] = "";
  FS.Files[testPath("second.h")] = "";
  auto VFS = FS.view(std::nullopt);
  auto Expanded =
      expandFileRenames({{testPath("old/./"), testPath("new/../new")},
                         {testPath("old/a.h"), testPath("new/a.h")},
                         {testPath("second.h"), testPath("renamed-second.h")}},
                        testRoot(), *VFS);
  ASSERT_THAT_EXPECTED(Expanded, llvm::Succeeded());
  EXPECT_THAT(*Expanded, testing::UnorderedElementsAre(
                             testing::Field(&FileRenameMapping::NewPath,
                                            testPath("new/a.h")),
                             testing::Field(&FileRenameMapping::NewPath,
                                            testPath("renamed-second.h"))));
}

TEST(FileRename, RejectsExistingDestinationAndOutsideWorkspace) {
  MockFS FS;
  FS.Files[testPath("old.h")] = "";
  FS.Files[testPath("existing.h")] = "";
  auto VFS = FS.view(std::nullopt);
  EXPECT_THAT_EXPECTED(
      expandFileRenames({{testPath("old.h"), testPath("existing.h")}},
                        testRoot(), *VFS),
      llvm::FailedWithMessage(HasSubstr("destination already exists")));

  llvm::SmallString<128> Outside(testRoot());
  llvm::sys::path::remove_filename(Outside);
  llvm::sys::path::append(Outside, "outside.h");
  EXPECT_THAT_EXPECTED(
      expandFileRenames({{testPath("old.h"), Outside.str().str()}}, testRoot(),
                        *VFS),
      llvm::FailedWithMessage(HasSubstr("outside the workspace")));
}

TEST(FileRename, RejectsInvalidDirectoryDestinations) {
  MockFS FS;
  FS.Files[testPath("old/file.h")] = "";
  FS.Files[testPath("existing/other.h")] = "";
  auto VFS = FS.view(std::nullopt);
  EXPECT_THAT_EXPECTED(
      expandFileRenames({{testPath("old"), testPath("existing")}}, testRoot(),
                        *VFS),
      llvm::FailedWithMessage(HasSubstr("destination already exists")));
  EXPECT_THAT_EXPECTED(
      expandFileRenames({{testPath("old"), testPath("old/nested")}}, testRoot(),
                        *VFS),
      llvm::FailedWithMessage(HasSubstr("inside its source directory")));
}

} // namespace
} // namespace clangd
} // namespace clang
