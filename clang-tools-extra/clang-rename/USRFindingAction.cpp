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
/// \brief Provides an action to rename every symbol at a point.
///
//===----------------------------------------------------------------------===//

#include "USRFindingAction.h"
#include "USRFinder.h"
#include "clang/AST/AST.h"
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

// Get the USRs for the constructors of the class.
static std::vector<std::string> getAllConstructorUSRs(
    const CXXRecordDecl *Decl) {
  std::vector<std::string> USRs;

  // We need to get the definition of the record (as opposed to any forward
  // declarations) in order to find the constructor and destructor.
  const auto *RecordDecl = Decl->getDefinition();

  // Iterate over all the constructors and add their USRs.
  for (const auto *CtorDecl : RecordDecl->ctors())
    USRs.push_back(getUSRForDecl(CtorDecl));

  // Ignore destructors. GetLocationsOfUSR will find the declaration of and
  // explicit calls to a destructor through TagTypeLoc (and it is better for the
  // purpose of renaming).
  //
  // For example, in the following code segment,
  //  1  class C {
  //  2    ~C();
  //  3  };
  // At line 3, there is a NamedDecl starting from '~' and a TagTypeLoc starting
  // from 'C'.

  return USRs;
}

struct NamedDeclFindingConsumer : public ASTConsumer {
  void HandleTranslationUnit(ASTContext &Context) override {
    const auto &SourceMgr = Context.getSourceManager();
    // The file we look for the USR in will always be the main source file.
    const auto Point = SourceMgr.getLocForStartOfFile(
        SourceMgr.getMainFileID()).getLocWithOffset(SymbolOffset);
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

    // If the decl is a constructor or destructor, we want to instead take the
    // decl of the parent record.
    if (const auto *CtorDecl = dyn_cast<CXXConstructorDecl>(FoundDecl))
      FoundDecl = CtorDecl->getParent();
    else if (const auto *DtorDecl = dyn_cast<CXXDestructorDecl>(FoundDecl))
      FoundDecl = DtorDecl->getParent();

    // If the decl is in any way relatedpp to a class, we want to make sure we
    // search for the constructor and destructor as well as everything else.
    if (const auto *Record = dyn_cast<CXXRecordDecl>(FoundDecl))
      *USRs = getAllConstructorUSRs(Record);

    USRs->push_back(getUSRForDecl(FoundDecl));
    *SpellingName = FoundDecl->getNameAsString();
  }

  unsigned SymbolOffset;
  std::string OldName;
  std::string *SpellingName;
  std::vector<std::string> *USRs;
};

std::unique_ptr<ASTConsumer>
USRFindingAction::newASTConsumer() {
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
