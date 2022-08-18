//===- IncludeTreePPActions.cpp - PP actions using include-tree -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/IncludeTreePPActions.h"
#include "clang/CAS/IncludeTree.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/PPCachedActions.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;

namespace {

struct IncludeStackInfo {
  cas::IncludeTree Tree;
  SourceLocation FileStartLoc;
  unsigned CurIncludeIndex = 0;
  unsigned CurHasIncludeCheckIndex = 0;
};

/// Uses the info from an \p IncludeTreeRoot to resolve include directives
/// and evaluate \p __has_include checks.
class IncludeTreePPActions final : public PPCachedActions {
  cas::IncludeTree MainTree;
  SmallVector<IncludeStackInfo> IncludeStack;
  bool HasCASErrorOccurred = false;

  void reportError(Preprocessor &PP, llvm::Error &&E) {
    PP.getDiagnostics().Report(diag::err_unable_to_load_include_tree_node)
        << llvm::toString(std::move(E));
    HasCASErrorOccurred = true;
  }

public:
  IncludeTreePPActions(cas::IncludeTree MainTree)
      : MainTree(std::move(MainTree)) {}

  FileID handlePredefines(Preprocessor &PP) override {
    auto createEmptyFID = [&]() -> FileID {
      llvm::MemoryBufferRef Buffer({}, "<built-in>");
      return PP.getSourceManager().createFileID(Buffer);
    };

    if (HasCASErrorOccurred) {
      return createEmptyFID();
    }

    auto reportError = [&](llvm::Error &&E) -> FileID {
      this->reportError(PP, std::move(E));
      return createEmptyFID();
    };

    SourceManager &SM = PP.getSourceManager();
    SourceLocation MainFileLoc = SM.getLocForStartOfFile(SM.getMainFileID());
    IncludeStack.push_back({std::move(MainTree), MainFileLoc});

    IncludeStackInfo &IncludeInfo = IncludeStack.back();
    Expected<cas::IncludeTree> EnteredTree =
        IncludeInfo.Tree.getInclude(IncludeInfo.CurIncludeIndex++);
    if (!EnteredTree)
      return reportError(EnteredTree.takeError());
    auto FileInfo = EnteredTree->getBaseFileInfo();
    if (!FileInfo)
      return reportError(FileInfo.takeError());
    llvm::MemoryBufferRef Buffer(FileInfo->Contents, FileInfo->Filename);
    FileID FID = SM.createFileID(Buffer);
    IncludeStack.push_back(
        {std::move(*EnteredTree), SM.getLocForStartOfFile(FID)});
    return FID;
  }

  bool evaluateHasInclude(Preprocessor &PP, SourceLocation Loc,
                          bool IsIncludeNext) override {
    if (HasCASErrorOccurred)
      return false;

    IncludeStackInfo &IncludeInfo = IncludeStack.back();
    unsigned Index = IncludeInfo.CurHasIncludeCheckIndex++;
    return IncludeInfo.Tree.getCheckResult(Index);
  }

  Optional<FileID>
  handleIncludeDirective(Preprocessor &PP, SourceLocation IncludeLoc,
                         SourceLocation AfterDirectiveLoc) override {
    if (HasCASErrorOccurred)
      return None;

    IncludeStackInfo &IncludeInfo = IncludeStack.back();
    if (IncludeInfo.CurIncludeIndex >= IncludeInfo.Tree.getNumIncludes())
      return None;

    unsigned ExpectedOffset =
        IncludeInfo.Tree.getIncludeOffset(IncludeInfo.CurIncludeIndex);
    SourceLocation ExpectedLoc =
        IncludeInfo.FileStartLoc.getLocWithOffset(ExpectedOffset);
    if (ExpectedLoc != AfterDirectiveLoc)
      return None;

    auto reportError = [&](llvm::Error &&E) -> Optional<FileID> {
      this->reportError(PP, std::move(E));
      return None;
    };

    Expected<cas::IncludeTree> EnteredTree =
        IncludeInfo.Tree.getInclude(IncludeInfo.CurIncludeIndex++);
    if (!EnteredTree)
      return reportError(EnteredTree.takeError());
    auto File = EnteredTree->getBaseFile();
    if (!File)
      return reportError(File.takeError());
    auto FilenameBlob = File->getFilename();
    if (!FilenameBlob)
      return reportError(FilenameBlob.takeError());

    SourceManager &SM = PP.getSourceManager();
    Expected<FileEntryRef> FE =
        SM.getFileManager().getFileRef(FilenameBlob->getData(),
                                       /*OpenFile=*/true);
    if (!FE)
      return reportError(FE.takeError());
    FileID FID =
        SM.createFileID(*FE, IncludeLoc, EnteredTree->getFileCharacteristic());
    PP.markIncluded(*FE);
    IncludeStack.push_back(
        {std::move(*EnteredTree), SM.getLocForStartOfFile(FID)});
    return FID;
  }

  void exitedFile(Preprocessor &PP, FileID FID) override {
    if (HasCASErrorOccurred)
      return;

    assert(!IncludeStack.empty());
    assert(IncludeStack.back().FileStartLoc ==
           PP.getSourceManager().getLocForStartOfFile(FID));
    assert(IncludeStack.back().CurIncludeIndex ==
           IncludeStack.back().Tree.getNumIncludes());
    IncludeStack.pop_back();
  }
};
} // namespace

Expected<std::unique_ptr<PPCachedActions>>
clang::createPPActionsFromIncludeTree(cas::IncludeTreeRoot &Root) {
  auto MainTree = Root.getMainFileTree();
  if (!MainTree)
    return MainTree.takeError();
  return std::make_unique<IncludeTreePPActions>(std::move(*MainTree));
}
