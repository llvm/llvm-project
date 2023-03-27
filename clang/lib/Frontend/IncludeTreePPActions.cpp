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
        IncludeInfo.Tree.getIncludeTree(IncludeInfo.CurIncludeIndex++);
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

  std::variant<std::monostate, IncludeFile, IncludeModule>
  handleIncludeDirective(Preprocessor &PP, SourceLocation IncludeLoc,
                         SourceLocation AfterDirectiveLoc) override {
    if (HasCASErrorOccurred)
      return {};

    IncludeStackInfo &IncludeInfo = IncludeStack.back();
    if (IncludeInfo.CurIncludeIndex >= IncludeInfo.Tree.getNumIncludes())
      return {};

    unsigned ExpectedOffset =
        IncludeInfo.Tree.getIncludeOffset(IncludeInfo.CurIncludeIndex);
    SourceLocation ExpectedLoc =
        IncludeInfo.FileStartLoc.getLocWithOffset(ExpectedOffset);
    if (ExpectedLoc != AfterDirectiveLoc)
      return {};

    auto reportError = [&](llvm::Error &&E) -> std::monostate {
      this->reportError(PP, std::move(E));
      return {};
    };

    auto reportErrorTwine = [&](const llvm::Twine &T) -> std::monostate {
      return reportError(
          llvm::createStringError(llvm::inconvertibleErrorCode(), T));
    };

    Expected<cas::IncludeTree::Node> Node =
        IncludeInfo.Tree.getIncludeNode(IncludeInfo.CurIncludeIndex++);
    if (!Node)
      return reportError(Node.takeError());

    if (Node->getKind() == cas::IncludeTree::NodeKind::ModuleImport) {
      cas::IncludeTree::ModuleImport Import = Node->getModuleImport();
      SmallVector<std::pair<IdentifierInfo *, SourceLocation>, 2> Path;
      SmallVector<StringRef, 2> ModuleComponents;
      Import.getModuleName().split(ModuleComponents, '.');
      for (StringRef Component : ModuleComponents)
        Path.emplace_back(PP.getIdentifierInfo(Component), IncludeLoc);
      return IncludeModule{std::move(Path), Import.visibilityOnly()};
    }

    assert(Node->getKind() == cas::IncludeTree::NodeKind::Tree);

    cas::IncludeTree EnteredTree = Node->getIncludeTree();
    auto File = EnteredTree.getBaseFile();
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
        SM.createFileID(*FE, IncludeLoc, EnteredTree.getFileCharacteristic());
    PP.markIncluded(*FE);
    IncludeStack.push_back(
        {std::move(EnteredTree), SM.getLocForStartOfFile(FID)});

    Module *M = nullptr;
    auto SubmoduleName = EnteredTree.getSubmoduleName();
    if (!SubmoduleName)
      return reportError(SubmoduleName.takeError());
    if (*SubmoduleName) {
      SmallVector<StringRef> ModuleComponents;
      (*SubmoduleName)->split(ModuleComponents, '.');
      M = PP.getHeaderSearchInfo().lookupModule(
          ModuleComponents[0], IncludeLoc,
          /*AllowSearch=*/false, /*AllowExtraModuleMapSearch=*/false);
      if (!M)
        return reportErrorTwine(llvm::Twine("failed to find module '") +
                                ModuleComponents[0] + "'");
      for (StringRef Sub : ArrayRef(ModuleComponents).drop_front()) {
        M = M->findOrInferSubmodule(Sub);
        if (!M)
          return reportErrorTwine(
              llvm::Twine("failed to find or infer submodule '") + Sub + "'");
      }

      // Add to known headers for the module.
      ModuleMap &MMap = PP.getHeaderSearchInfo().getModuleMap();
      Module::Header H;
      H.Entry = *FE;
      MMap.addHeader(M, std::move(H), ModuleMap::NormalHeader);
    }

    return IncludeFile{FID, M};
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
