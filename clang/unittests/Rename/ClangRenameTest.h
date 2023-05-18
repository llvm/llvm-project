//===-- ClangRenameTests.cpp - clang-rename unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_RENAME_CLANGRENAMETEST_H
#define LLVM_CLANG_UNITTESTS_RENAME_CLANGRENAMETEST_H

#include "unittests/Tooling/RewriterTestContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"
#include "clang/Tooling/Refactoring/Rename/USRFindingAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace clang_rename {
namespace test {

struct Case {
  std::string Before;
  std::string After;
  std::string OldName;
  std::string NewName;
};

class ClangRenameTest : public testing::Test,
                        public testing::WithParamInterface<Case> {
protected:
  void AppendToHeader(StringRef Code) { HeaderContent += Code.str(); }

  std::string runClangRenameOnCode(llvm::StringRef Code,
                                   llvm::StringRef OldName,
                                   llvm::StringRef NewName) {
    std::string NewCode;
    llvm::raw_string_ostream(NewCode) << llvm::format(
        "#include \"%s\"\n%s", HeaderName.c_str(), Code.str().c_str());
    tooling::FileContentMappings FileContents = {{HeaderName, HeaderContent},
                                                 {CCName, NewCode}};
    clang::RewriterTestContext Context;
    Context.createInMemoryFile(HeaderName, HeaderContent);
    clang::FileID InputFileID = Context.createInMemoryFile(CCName, NewCode);

    tooling::USRFindingAction FindingAction({}, {std::string(OldName)}, false);
    std::unique_ptr<tooling::FrontendActionFactory> USRFindingActionFactory =
        tooling::newFrontendActionFactory(&FindingAction);

    if (!tooling::runToolOnCodeWithArgs(
            USRFindingActionFactory->create(), NewCode, {"-std=c++11"}, CCName,
            "clang-rename", std::make_shared<PCHContainerOperations>(),
            FileContents))
      return "";

    const std::vector<std::vector<std::string>> &USRList =
        FindingAction.getUSRList();
    std::vector<std::string> NewNames = {std::string(NewName)};
    std::map<std::string, tooling::Replacements> FileToReplacements;
    tooling::QualifiedRenamingAction RenameAction(NewNames, USRList,
                                                  FileToReplacements);
    auto RenameActionFactory = tooling::newFrontendActionFactory(&RenameAction);
    if (!tooling::runToolOnCodeWithArgs(
            RenameActionFactory->create(), NewCode, {"-std=c++11"}, CCName,
            "clang-rename", std::make_shared<PCHContainerOperations>(),
            FileContents))
      return "";

    formatAndApplyAllReplacements(FileToReplacements, Context.Rewrite, "llvm");
    return Context.getRewrittenText(InputFileID);
  }

  void CompareSnippets(StringRef Expected, StringRef Actual) {
    std::string ExpectedCode;
    llvm::raw_string_ostream(ExpectedCode) << llvm::format(
        "#include \"%s\"\n%s", HeaderName.c_str(), Expected.str().c_str());
    EXPECT_EQ(format(ExpectedCode), format(Actual));
  }

  std::string format(llvm::StringRef Code) {
    tooling::Replacements Replaces = format::reformat(
        format::getLLVMStyle(), Code, {tooling::Range(0, Code.size())});
    auto ChangedCode = tooling::applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(ChangedCode));
    if (!ChangedCode) {
      llvm::errs() << llvm::toString(ChangedCode.takeError());
      return "";
    }
    return *ChangedCode;
  }

  std::string HeaderContent;
  std::string HeaderName = "header.h";
  std::string CCName = "input.cc";
};

} // namespace test
} // namespace clang_rename
} // namesdpace clang

#endif
