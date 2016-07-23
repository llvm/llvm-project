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
  explicit USRLocFindingASTVisitor(StringRef USR, StringRef PrevName,
                                   const ASTContext &Context)
      : USR(USR), PrevName(PrevName), Context(Context) {}

  // Declaration visitors:

  bool VisitCXXConstructorDecl(clang::CXXConstructorDecl *ConstructorDecl) {
    for (auto &Initializer : ConstructorDecl->inits()) {
      if (Initializer->getSourceOrder() == -1) {
        // Ignore implicit initializers.
        continue;
      }
      if (const clang::FieldDecl *FieldDecl = Initializer->getAnyMember()) {
        if (getUSRForDecl(FieldDecl) == USR) {
          // The initializer refers to a field that is to be renamed.
          SourceLocation Location = Initializer->getSourceLocation();
          StringRef TokenName = Lexer::getSourceText(
              CharSourceRange::getTokenRange(Location),
              Context.getSourceManager(), Context.getLangOpts());
          if (TokenName == PrevName) {
            // The token of the source location we find actually has the old
            // name.
            LocationsFound.push_back(Initializer->getSourceLocation());
          }
        }
      }
    }
    return true;
  }

  bool VisitNamedDecl(const NamedDecl *Decl) {
    if (getUSRForDecl(Decl) == USR) {
      checkAndAddLocation(Decl->getLocation());
    }
    return true;
  }

  // Expression visitors:

  bool VisitDeclRefExpr(const DeclRefExpr *Expr) {
    const auto *Decl = Expr->getFoundDecl();

    if (getUSRForDecl(Decl) == USR) {
      const SourceManager &Manager = Decl->getASTContext().getSourceManager();
      SourceLocation Location = Manager.getSpellingLoc(Expr->getLocation());
      checkAndAddLocation(Location);
    }

    return true;
  }

  bool VisitMemberExpr(const MemberExpr *Expr) {
    const auto *Decl = Expr->getFoundDecl().getDecl();
    if (getUSRForDecl(Decl) == USR) {
      const SourceManager &Manager = Decl->getASTContext().getSourceManager();
      SourceLocation Location = Manager.getSpellingLoc(Expr->getMemberLoc());
      checkAndAddLocation(Location);
    }
    return true;
  }

  bool VisitCXXStaticCastExpr(clang::CXXStaticCastExpr *Expr) {
    return handleCXXNamedCastExpr(Expr);
  }

  bool VisitCXXDynamicCastExpr(clang::CXXDynamicCastExpr *Expr) {
    return handleCXXNamedCastExpr(Expr);
  }

  bool VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *Expr) {
    return handleCXXNamedCastExpr(Expr);
  }

  bool VisitCXXConstCastExpr(clang::CXXConstCastExpr *Expr) {
    return handleCXXNamedCastExpr(Expr);
  }

  // Other visitors:

  bool VisitTypeLoc(const TypeLoc Loc) {
    if (getUSRForDecl(Loc.getType()->getAsCXXRecordDecl()) == USR) {
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
      const auto *Decl = NameLoc.getNestedNameSpecifier()->getAsNamespace();
      if (Decl && getUSRForDecl(Decl) == USR) {
        checkAndAddLocation(NameLoc.getLocalBeginLoc());
      }
      NameLoc = NameLoc.getPrefix();
    }
  }

  bool handleCXXNamedCastExpr(clang::CXXNamedCastExpr *Expr) {
    clang::QualType Type = Expr->getType();
    // See if this a cast of a pointer.
    const RecordDecl *Decl = Type->getPointeeCXXRecordDecl();
    if (!Decl) {
      // See if this is a cast of a reference.
      Decl = Type->getAsCXXRecordDecl();
    }

    if (Decl && getUSRForDecl(Decl) == USR) {
      SourceLocation Location =
          Expr->getTypeInfoAsWritten()->getTypeLoc().getBeginLoc();
      checkAndAddLocation(Location);
    }

    return true;
  }

private:
  void checkAndAddLocation(SourceLocation Loc) {
    const auto BeginLoc = Loc;
    const auto EndLoc = Lexer::getLocForEndOfToken(
                                   BeginLoc, 0, Context.getSourceManager(),
                                   Context.getLangOpts());
    StringRef TokenName =
        Lexer::getSourceText(CharSourceRange::getTokenRange(BeginLoc, EndLoc),
                             Context.getSourceManager(), Context.getLangOpts());
    size_t Offset = TokenName.find(PrevName);
    if (Offset != StringRef::npos) {
      // The token of the source location we find actually has the old
      // name.
      LocationsFound.push_back(BeginLoc.getLocWithOffset(Offset));
    }
  }

  // All the locations of the USR were found.
  const std::string USR;
  // Old name that is renamed.
  const std::string PrevName;
  std::vector<clang::SourceLocation> LocationsFound;
  const ASTContext &Context;
};
} // namespace

std::vector<SourceLocation> getLocationsOfUSR(StringRef USR, StringRef PrevName,
                                              Decl *Decl) {
  USRLocFindingASTVisitor Visitor(USR, PrevName, Decl->getASTContext());
  Visitor.TraverseDecl(Decl);
  NestedNameSpecifierLocFinder Finder(Decl->getASTContext());
  for (const auto &Location : Finder.getNestedNameSpecifierLocations()) {
    Visitor.handleNestedNameSpecifierLoc(Location);
  }
  return Visitor.getLocationsFound();
}

} // namespace rename
} // namespace clang
