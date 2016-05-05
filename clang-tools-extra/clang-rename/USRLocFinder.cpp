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
  explicit USRLocFindingASTVisitor(const std::string USR) : USR(USR) {
  }

  // Declaration visitors:

  bool VisitNamedDecl(const NamedDecl *Decl) {
    if (getUSRForDecl(Decl) == USR) {
      LocationsFound.push_back(Decl->getLocation());
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *Decl) {
    clang::QualType Type = Decl->getType();
    const clang::RecordDecl *RecordDecl = Type->getPointeeCXXRecordDecl();
    if (RecordDecl) {
      if (getUSRForDecl(RecordDecl) == USR) {
        // The declaration refers to a type that is to be renamed.
        LocationsFound.push_back(Decl->getTypeSpecStartLoc());
      }
    }
    return true;
  }

  // Expression visitors:

  bool VisitDeclRefExpr(const DeclRefExpr *Expr) {
    const auto *Decl = Expr->getFoundDecl();

    checkNestedNameSpecifierLoc(Expr->getQualifierLoc());
    if (getUSRForDecl(Decl) == USR) {
      LocationsFound.push_back(Expr->getLocation());
    }

    return true;
  }

  bool VisitMemberExpr(const MemberExpr *Expr) {
    const auto *Decl = Expr->getFoundDecl().getDecl();
    if (getUSRForDecl(Decl) == USR) {
      LocationsFound.push_back(Expr->getMemberLoc());
    }
    return true;
  }

  // Non-visitors:

  // \brief Returns a list of unique locations. Duplicate or overlapping
  // locations are erroneous and should be reported!
  const std::vector<clang::SourceLocation> &getLocationsFound() const {
    return LocationsFound;
  }

private:
  // Namespace traversal:
  void checkNestedNameSpecifierLoc(NestedNameSpecifierLoc NameLoc) {
    while (NameLoc) {
      const auto *Decl = NameLoc.getNestedNameSpecifier()->getAsNamespace();
      if (Decl && getUSRForDecl(Decl) == USR)
        LocationsFound.push_back(NameLoc.getLocalBeginLoc());
      NameLoc = NameLoc.getPrefix();
    }
  }

  // All the locations of the USR were found.
  const std::string USR;
  std::vector<clang::SourceLocation> LocationsFound;
};
} // namespace

std::vector<SourceLocation> getLocationsOfUSR(const std::string USR,
                                              Decl *Decl) {
  USRLocFindingASTVisitor visitor(USR);

  visitor.TraverseDecl(Decl);
  return visitor.getLocationsFound();
}

} // namespace rename
} // namespace clang
