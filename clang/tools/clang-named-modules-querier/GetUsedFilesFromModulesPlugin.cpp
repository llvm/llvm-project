//===- ClangGetUsedFilesFromModules.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Serialization/ASTDeserializationListener.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Path.h"

namespace clang {

class DeclsQuerier : public ASTDeserializationListener {
public:
  void DeclRead(serialization::DeclID ID, const Decl *D) override {
    if (!isa<NamedDecl>(D))
      return;

    if (!D->getLexicalDeclContext())
      return;

    if (!isa<FunctionDecl>(D) &&
        !D->getLexicalDeclContext()->getRedeclContext()->isFileContext())
      return;

    ASTContext &Ctx = D->getASTContext();
    SourceManager &SMgr = Ctx.getSourceManager();
    FullSourceLoc FSL(D->getLocation(), SMgr);
    if (!FSL.isValid())
      return;

    FileIDSet.insert(FSL.getFileID());
  }

  llvm::DenseSet<FileID> FileIDSet;
};

class DeclsQuerierConsumer : public ASTConsumer {
  CompilerInstance &CI;
  StringRef InFile;
  std::string OutputFile;
  DeclsQuerier Querier;
  
public:
  DeclsQuerierConsumer(CompilerInstance &CI, StringRef InFile, StringRef OutputFile)
    : CI(CI), InFile(InFile), OutputFile(OutputFile) {}

  ASTDeserializationListener *GetASTDeserializationListener() override {
    return &Querier;
  }

  std::unique_ptr<raw_pwrite_stream> getOutputFile() {
    if (OutputFile.empty()) {
      llvm::SmallString<256> Path(InFile);
      llvm::sys::path::replace_extension(Path, "used_files.txt");
      OutputFile = (std::string)Path;
    }

    std::error_code EC;
    auto OS = std::make_unique<llvm::raw_fd_ostream>(OutputFile, EC);
    if (EC)
      return nullptr;
    
    return OS;
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    std::unique_ptr<raw_pwrite_stream> OS = getOutputFile();
    if (!OS)
      return;

    auto &SMgr = Ctx.getSourceManager();
    
    for (const auto &FID : Querier.FileIDSet) {
      const FileEntry *FE = SMgr.getFileEntryForID(FID);
      if (!FE)
        continue;
      
      *OS << FE->getName() << "\n";
    }
  }
};

void PrintHelp();

class DeclsQueryAction : public PluginASTAction {
  std::string OutputFile;

public:
  DeclsQueryAction(StringRef OutputFile) : OutputFile(OutputFile) {}
  DeclsQueryAction() = default;

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<DeclsQuerierConsumer>(CI, InFile, OutputFile);
  }

  ActionType getActionType() override { return AddAfterMainAction; }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &Args) override {
    for (auto &Arg : Args) {
      if (StringRef(Arg).startswith("output=")) {
        OutputFile = StringRef(Arg).split('=').second.str();
      } else {
        PrintHelp();
        return false;
      }
    }

    return true;
  }
};

void PrintHelp() {
  llvm::outs() << R"cpp(
To get used decls from modules.

If you're using plugin, use -fplugin-arg-get_used_files_from_modules-output=<output-file>
to specify the output path of used files.
  )cpp";
}
}

static clang::FrontendPluginRegistry::Add<clang::DeclsQueryAction>
X("get_used_files_from_modules", "get used files from modules");
