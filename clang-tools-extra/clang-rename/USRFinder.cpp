//===--- tools/extra/clang-rename/USRFinder.cpp - Clang rename tool -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file Implements a recursive AST visitor that finds the USR of a symbol at a
/// point.
///
//===----------------------------------------------------------------------===//

#include "USRFinder.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

namespace clang {
namespace rename {

// NamedDeclFindingASTVisitor recursively visits each AST node to find the
// symbol underneath the cursor.
// FIXME: move to seperate .h/.cc file if this gets too large.
namespace {
class NamedDeclFindingASTVisitor
    : public clang::RecursiveASTVisitor<NamedDeclFindingASTVisitor> {
public:
  // \brief Finds the NamedDecl at a point in the source.
  // \param Point the location in the source to search for the NamedDecl.
  explicit NamedDeclFindingASTVisitor(const SourceLocation Point,
                                      const ASTContext &Context)
      : Result(nullptr), Point(Point), Context(Context) {}

  // \brief Finds the NamedDecl for a name in the source.
  // \param Name the fully qualified name.
  explicit NamedDeclFindingASTVisitor(const std::string &Name,
                                      const ASTContext &Context)
      : Result(nullptr), Name(Name), Context(Context) {}

  // Declaration visitors:

  // \brief Checks if the point falls within the NameDecl. This covers every
  // declaration of a named entity that we may come across. Usually, just
  // checking if the point lies within the length of the name of the declaration
  // and the start location is sufficient.
  bool VisitNamedDecl(const NamedDecl *Decl) {
    return dyn_cast<CXXConversionDecl>(Decl)
               ? true
               : setResult(Decl, Decl->getLocation(),
                           Decl->getNameAsString().length());
  }

  // Expression visitors:

  bool VisitDeclRefExpr(const DeclRefExpr *Expr) {
    const NamedDecl *Decl = Expr->getFoundDecl();
    return setResult(Decl, Expr->getLocation(),
                     Decl->getNameAsString().length());
  }

  bool VisitMemberExpr(const MemberExpr *Expr) {
    const NamedDecl *Decl = Expr->getFoundDecl().getDecl();
    return setResult(Decl, Expr->getMemberLoc(),
                     Decl->getNameAsString().length());
  }

  // Other visitors:

  bool VisitTypeLoc(const TypeLoc Loc) {
    const SourceLocation TypeBeginLoc = Loc.getBeginLoc();
    const SourceLocation TypeEndLoc = Lexer::getLocForEndOfToken(
        TypeBeginLoc, 0, Context.getSourceManager(), Context.getLangOpts());
    if (const auto *TemplateTypeParm =
            dyn_cast<TemplateTypeParmType>(Loc.getType()))
      return setResult(TemplateTypeParm->getDecl(), TypeBeginLoc, TypeEndLoc);
    if (const auto *TemplateSpecType =
            dyn_cast<TemplateSpecializationType>(Loc.getType())) {
      return setResult(TemplateSpecType->getTemplateName().getAsTemplateDecl(),
                       TypeBeginLoc, TypeEndLoc);
    }
    return setResult(Loc.getType()->getAsCXXRecordDecl(), TypeBeginLoc,
                     TypeEndLoc);
  }

  bool VisitCXXConstructorDecl(clang::CXXConstructorDecl *ConstructorDecl) {
    for (const auto *Initializer : ConstructorDecl->inits()) {
      // Ignore implicit initializers.
      if (!Initializer->isWritten())
        continue;
      if (const clang::FieldDecl *FieldDecl = Initializer->getMember()) {
        const SourceLocation InitBeginLoc = Initializer->getSourceLocation(),
                             InitEndLoc = Lexer::getLocForEndOfToken(
                                 InitBeginLoc, 0, Context.getSourceManager(),
                                 Context.getLangOpts());
        if (!setResult(FieldDecl, InitBeginLoc, InitEndLoc))
          return false;
      }
    }
    return true;
  }

  // Other:

  const NamedDecl *getNamedDecl() { return Result; }

  // \brief Determines if a namespace qualifier contains the point.
  // \returns false on success and sets Result.
  void handleNestedNameSpecifierLoc(NestedNameSpecifierLoc NameLoc) {
    while (NameLoc) {
      const NamespaceDecl *Decl =
          NameLoc.getNestedNameSpecifier()->getAsNamespace();
      setResult(Decl, NameLoc.getLocalBeginLoc(), NameLoc.getLocalEndLoc());
      NameLoc = NameLoc.getPrefix();
    }
  }

private:
  // \brief Sets Result to Decl if the Point is within Start and End.
  // \returns false on success.
  bool setResult(const NamedDecl *Decl, SourceLocation Start,
                 SourceLocation End) {
    if (!Decl)
      return true;
    if (Name.empty()) {
      // Offset is used to find the declaration.
      if (!Start.isValid() || !Start.isFileID() || !End.isValid() ||
          !End.isFileID() || !isPointWithin(Start, End))
        return true;
    } else {
      // Fully qualified name is used to find the declaration.
      if (Name != Decl->getQualifiedNameAsString())
        return true;
    }
    Result = Decl;
    return false;
  }

  // \brief Sets Result to Decl if Point is within Loc and Loc + Offset.
  // \returns false on success.
  bool setResult(const NamedDecl *Decl, SourceLocation Loc, unsigned Offset) {
    // FIXME: Add test for Offset == 0. Add test for Offset - 1 (vs -2 etc).
    return Offset == 0 ||
           setResult(Decl, Loc, Loc.getLocWithOffset(Offset - 1));
  }

  // \brief Determines if the Point is within Start and End.
  bool isPointWithin(const SourceLocation Start, const SourceLocation End) {
    // FIXME: Add tests for Point == End.
    return Point == Start || Point == End ||
           (Context.getSourceManager().isBeforeInTranslationUnit(Start,
                                                                 Point) &&
            Context.getSourceManager().isBeforeInTranslationUnit(Point, End));
  }

  const NamedDecl *Result;
  const SourceLocation Point; // The location to find the NamedDecl.
  const std::string Name;
  const ASTContext &Context;
};
} // namespace

const NamedDecl *getNamedDeclAt(const ASTContext &Context,
                                const SourceLocation Point) {
  const SourceManager &SM = Context.getSourceManager();
  NamedDeclFindingASTVisitor Visitor(Point, Context);

  // Try to be clever about pruning down the number of top-level declarations we
  // see. If both start and end is either before or after the point we're
  // looking for the point cannot be inside of this decl. Don't even look at it.
  for (auto *CurrDecl : Context.getTranslationUnitDecl()->decls()) {
    SourceLocation StartLoc = CurrDecl->getLocStart();
    SourceLocation EndLoc = CurrDecl->getLocEnd();
    if (StartLoc.isValid() && EndLoc.isValid() &&
        SM.isBeforeInTranslationUnit(StartLoc, Point) !=
            SM.isBeforeInTranslationUnit(EndLoc, Point))
      Visitor.TraverseDecl(CurrDecl);
  }

  NestedNameSpecifierLocFinder Finder(const_cast<ASTContext &>(Context));
  for (const auto &Location : Finder.getNestedNameSpecifierLocations())
    Visitor.handleNestedNameSpecifierLoc(Location);

  return Visitor.getNamedDecl();
}

const NamedDecl *getNamedDeclFor(const ASTContext &Context,
                                 const std::string &Name) {
  NamedDeclFindingASTVisitor Visitor(Name, Context);
  Visitor.TraverseDecl(Context.getTranslationUnitDecl());

  return Visitor.getNamedDecl();
}

std::string getUSRForDecl(const Decl *Decl) {
  llvm::SmallVector<char, 128> Buff;

  // FIXME: Add test for the nullptr case.
  if (Decl == nullptr || index::generateUSRForDecl(Decl, Buff))
    return "";

  return std::string(Buff.data(), Buff.size());
}

} // namespace rename
} // namespace clang
