//===--- tools/extra/clang-rename/USRLocFinder.cpp - Clang rename tool ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Mehtods for finding all instances of a USR. Our strategy is very
/// simple; we just compare the USR at every relevant AST node with the one
/// provided.
///
//===----------------------------------------------------------------------===//

#include "USRLocFinder.h"
#include "USRFinder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

namespace clang {
namespace rename {

namespace {
// \brief This visitor recursively searches for all instances of a USR in a
// translation unit and stores them for later usage.
class USRLocFindingASTVisitor
    : public clang::RecursiveASTVisitor<USRLocFindingASTVisitor> {
public:
  explicit USRLocFindingASTVisitor(const std::vector<std::string> &USRs,
                                   StringRef PrevName,
                                   const ASTContext &Context)
      : USRSet(USRs.begin(), USRs.end()), PrevName(PrevName), Context(Context) {
  }

  // Declaration visitors:

  bool VisitCXXConstructorDecl(clang::CXXConstructorDecl *ConstructorDecl) {
    for (const auto *Initializer : ConstructorDecl->inits()) {
      // Ignore implicit initializers.
      if (!Initializer->isWritten())
        continue;
      if (const clang::FieldDecl *FieldDecl = Initializer->getMember()) {
        if (USRSet.find(getUSRForDecl(FieldDecl)) != USRSet.end())
          LocationsFound.push_back(Initializer->getSourceLocation());
      }
    }
    return true;
  }

  bool VisitNamedDecl(const NamedDecl *Decl) {
    if (USRSet.find(getUSRForDecl(Decl)) != USRSet.end())
      checkAndAddLocation(Decl->getLocation());
    return true;
  }

  // Expression visitors:

  bool VisitDeclRefExpr(const DeclRefExpr *Expr) {
    const NamedDecl *Decl = Expr->getFoundDecl();

    if (USRSet.find(getUSRForDecl(Decl)) != USRSet.end()) {
      const SourceManager &Manager = Decl->getASTContext().getSourceManager();
      SourceLocation Location = Manager.getSpellingLoc(Expr->getLocation());
      checkAndAddLocation(Location);
    }

    return true;
  }

  bool VisitMemberExpr(const MemberExpr *Expr) {
    const NamedDecl *Decl = Expr->getFoundDecl().getDecl();
    if (USRSet.find(getUSRForDecl(Decl)) != USRSet.end()) {
      const SourceManager &Manager = Decl->getASTContext().getSourceManager();
      SourceLocation Location = Manager.getSpellingLoc(Expr->getMemberLoc());
      checkAndAddLocation(Location);
    }
    return true;
  }

  // Other visitors:

  bool VisitTypeLoc(const TypeLoc Loc) {
    if (USRSet.find(getUSRForDecl(Loc.getType()->getAsCXXRecordDecl())) !=
        USRSet.end())
      checkAndAddLocation(Loc.getBeginLoc());
    if (const auto *TemplateTypeParm =
            dyn_cast<TemplateTypeParmType>(Loc.getType())) {
      if (USRSet.find(getUSRForDecl(TemplateTypeParm->getDecl())) !=
          USRSet.end())
        checkAndAddLocation(Loc.getBeginLoc());
    }
    return true;
  }

  // Non-visitors:

  // \brief Returns a list of unique locations. Duplicate or overlapping
  // locations are erroneous and should be reported!
  const std::vector<clang::SourceLocation> &getLocationsFound() const {
    return LocationsFound;
  }

  // Namespace traversal:
  void handleNestedNameSpecifierLoc(NestedNameSpecifierLoc NameLoc) {
    while (NameLoc) {
      const NamespaceDecl *Decl =
          NameLoc.getNestedNameSpecifier()->getAsNamespace();
      if (Decl && USRSet.find(getUSRForDecl(Decl)) != USRSet.end())
        checkAndAddLocation(NameLoc.getLocalBeginLoc());
      NameLoc = NameLoc.getPrefix();
    }
  }

private:
  void checkAndAddLocation(SourceLocation Loc) {
    const SourceLocation BeginLoc = Loc;
    const SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        BeginLoc, 0, Context.getSourceManager(), Context.getLangOpts());
    StringRef TokenName =
        Lexer::getSourceText(CharSourceRange::getTokenRange(BeginLoc, EndLoc),
                             Context.getSourceManager(), Context.getLangOpts());
    size_t Offset = TokenName.find(PrevName);

    // The token of the source location we find actually has the old
    // name.
    if (Offset != StringRef::npos)
      LocationsFound.push_back(BeginLoc.getLocWithOffset(Offset));
  }

  const std::set<std::string> USRSet;
  const std::string PrevName;
  std::vector<clang::SourceLocation> LocationsFound;
  const ASTContext &Context;
};
} // namespace

std::vector<SourceLocation>
getLocationsOfUSRs(const std::vector<std::string> &USRs, StringRef PrevName,
                   Decl *Decl) {
  USRLocFindingASTVisitor Visitor(USRs, PrevName, Decl->getASTContext());
  Visitor.TraverseDecl(Decl);
  NestedNameSpecifierLocFinder Finder(Decl->getASTContext());

  for (const auto &Location : Finder.getNestedNameSpecifierLocations())
    Visitor.handleNestedNameSpecifierLoc(Location);

  return Visitor.getLocationsFound();
}

} // namespace rename
} // namespace clang
