//===--- tools/extra/clang-rename/RenamingAction.cpp - Clang rename tool --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides an action to rename every symbol at a point.
///
//===----------------------------------------------------------------------===//

#include "RenamingAction.h"
#include "USRLocFinder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include <string>
#include <vector>

using namespace llvm;

namespace clang {
namespace rename {

class RenamingASTConsumer : public ASTConsumer {
public:
  RenamingASTConsumer(
      const std::string &NewName, const std::string &PrevName,
      const std::vector<std::string> &USRs,
      std::map<std::string, tooling::Replacements> &FileToReplaces,
      bool PrintLocations)
      : NewName(NewName), PrevName(PrevName), USRs(USRs),
        FileToReplaces(FileToReplaces), PrintLocations(PrintLocations) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    const auto &SourceMgr = Context.getSourceManager();
    std::vector<SourceLocation> RenamingCandidates;
    std::vector<SourceLocation> NewCandidates;

    NewCandidates =
        getLocationsOfUSRs(USRs, PrevName, Context.getTranslationUnitDecl());
    RenamingCandidates.insert(RenamingCandidates.end(), NewCandidates.begin(),
                              NewCandidates.end());

    auto PrevNameLen = PrevName.length();
    for (const auto &Loc : RenamingCandidates) {
      if (PrintLocations) {
        FullSourceLoc FullLoc(Loc, SourceMgr);
        errs() << "clang-rename: renamed at: " << SourceMgr.getFilename(Loc)
               << ":" << FullLoc.getSpellingLineNumber() << ":"
               << FullLoc.getSpellingColumnNumber() << "\n";
      }
      // FIXME: better error handling.
      auto Replace = tooling::Replacement(SourceMgr, Loc, PrevNameLen, NewName);
      auto Err = FileToReplaces[Replace.getFilePath()].add(Replace);
      if (Err)
        llvm::errs() << "Renaming failed in " << Replace.getFilePath() << "! "
                     << llvm::toString(std::move(Err)) << "\n";
    }
  }

private:
  const std::string &NewName, &PrevName;
  const std::vector<std::string> &USRs;
  std::map<std::string, tooling::Replacements> &FileToReplaces;
  bool PrintLocations;
};

std::unique_ptr<ASTConsumer> RenamingAction::newASTConsumer() {
  return llvm::make_unique<RenamingASTConsumer>(NewName, PrevName, USRs,
                                                FileToReplaces, PrintLocations);
}

} // namespace rename
} // namespace clang
