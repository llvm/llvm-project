//===--- tools/extra/clang-rename/USRFindingAction.cpp - Clang rename tool ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides an action to find USR for the symbol at <offset>, as well as
/// all additional USRs.
///
//===----------------------------------------------------------------------===//

#include "USRFindingAction.h"
#include "USRFinder.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include <algorithm>
#include <string>
#include <set>
#include <vector>


using namespace llvm;
using namespace clang::ast_matchers;

namespace clang {
namespace rename {

namespace {
// \brief NamedDeclFindingConsumer should delegate finding USRs of given Decl to
// AdditionalUSRFinder. AdditionalUSRFinder adds USRs of ctor and dtor if given
// Decl refers to class and adds USRs of all overridden methods if Decl refers
// to virtual method.
class AdditionalUSRFinder : public MatchFinder::MatchCallback {
public:
  explicit AdditionalUSRFinder(const Decl *FoundDecl, ASTContext &Context,
                             std::vector<std::string> *USRs)
    : FoundDecl(FoundDecl), Context(Context), USRs(USRs), USRSet(), Finder() {}

  void Find() {
    USRSet.insert(getUSRForDecl(FoundDecl));
    addUSRsFromOverrideSetsAndCtorDtors();
    addMatchers();
    Finder.matchAST(Context);
    USRs->insert(USRs->end(), USRSet.begin(), USRSet.end());
  }

private:
  void addMatchers() {
    const auto CXXMethodDeclMatcher =
        cxxMethodDecl(isVirtual()).bind("cxxMethodDecl");
    Finder.addMatcher(CXXMethodDeclMatcher, this);
  }

  // FIXME: Implement hasOverriddenMethod and matchesUSR matchers to make
  // lookups more efficient.
  virtual void run(const MatchFinder::MatchResult &Result) {
    const auto *VirtualMethod =
        Result.Nodes.getNodeAs<CXXMethodDecl>("cxxMethodDecl");
    bool Found = false;
    for (const auto &OverriddenMethod : VirtualMethod->overridden_methods()) {
      if (USRSet.find(getUSRForDecl(OverriddenMethod)) != USRSet.end()) {
        Found = true;
      }
    }
    if (Found) {
      USRSet.insert(getUSRForDecl(VirtualMethod));
    }
  }

  void addUSRsFromOverrideSetsAndCtorDtors() {
    // If D is CXXRecordDecl we should add all USRs of its constructors.
    if (const auto *RecordDecl = dyn_cast<CXXRecordDecl>(FoundDecl)) {
      RecordDecl = RecordDecl->getDefinition();
      for (const auto *CtorDecl : RecordDecl->ctors()) {
        USRSet.insert(getUSRForDecl(CtorDecl));
      }
      USRSet.insert(getUSRForDecl(RecordDecl->getDestructor()));
    }
    // If D is CXXMethodDecl we should add all USRs of its overriden methods.
    if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(FoundDecl)) {
      for (auto &OverriddenMethod : MethodDecl->overridden_methods()) {
        USRSet.insert(getUSRForDecl(OverriddenMethod));
      }
    }
  }

  const Decl *FoundDecl;
  ASTContext &Context;
  std::vector<std::string> *USRs;
  std::set<std::string> USRSet;
  MatchFinder Finder;
};
} // namespace

struct NamedDeclFindingConsumer : public ASTConsumer {
  void HandleTranslationUnit(ASTContext &Context) override {
    const auto &SourceMgr = Context.getSourceManager();
    // The file we look for the USR in will always be the main source file.
    const auto Point = SourceMgr.getLocForStartOfFile(SourceMgr.getMainFileID())
                           .getLocWithOffset(SymbolOffset);
    if (!Point.isValid())
      return;
    const NamedDecl *FoundDecl = nullptr;
    if (OldName.empty()) {
      FoundDecl = getNamedDeclAt(Context, Point);
    } else {
      FoundDecl = getNamedDeclFor(Context, OldName);
    }
    if (FoundDecl == nullptr) {
      FullSourceLoc FullLoc(Point, SourceMgr);
      errs() << "clang-rename: could not find symbol at "
             << SourceMgr.getFilename(Point) << ":"
             << FullLoc.getSpellingLineNumber() << ":"
             << FullLoc.getSpellingColumnNumber() << " (offset " << SymbolOffset
             << ").\n";
      return;
    }

    // If FoundDecl is a constructor or destructor, we want to instead take the
    // Decl of the corresponding class.
    if (const auto *CtorDecl = dyn_cast<CXXConstructorDecl>(FoundDecl)) {
      FoundDecl = CtorDecl->getParent();
    } else if (const auto *DtorDecl = dyn_cast<CXXDestructorDecl>(FoundDecl)) {
      FoundDecl = DtorDecl->getParent();
    }
    *SpellingName = FoundDecl->getNameAsString();

    AdditionalUSRFinder Finder(FoundDecl, Context, USRs);
    Finder.Find();
  }

  unsigned SymbolOffset;
  std::string OldName;
  std::string *SpellingName;
  std::vector<std::string> *USRs;
};

std::unique_ptr<ASTConsumer> USRFindingAction::newASTConsumer() {
  std::unique_ptr<NamedDeclFindingConsumer> Consumer(
      new NamedDeclFindingConsumer);
  SpellingName = "";
  Consumer->SymbolOffset = SymbolOffset;
  Consumer->OldName = OldName;
  Consumer->USRs = &USRs;
  Consumer->SpellingName = &SpellingName;
  return std::move(Consumer);
}

} // namespace rename
} // namespace clang
