//===--- SymbolOccurrenceFinder.cpp - Clang refactoring library -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Methods for finding all instances of a USR. Our strategy is very
/// simple; we just compare the USR at every relevant AST node with the one
/// provided.
///
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/SymbolOccurrenceFinder.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/Lexer.h"
#include "clang/Sema/DependentASTVisitor.h"
#include "clang/Tooling/Refactor/USRFinder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

namespace clang {
namespace tooling {
namespace rename {

namespace {
// \brief This visitor recursively searches for all instances of a USR in a
// translation unit and stores them for later usage.
class SymbolOccurrenceFinderASTVisitor
    : public DependentASTVisitor<SymbolOccurrenceFinderASTVisitor> {
public:
  explicit SymbolOccurrenceFinderASTVisitor(
      const SymbolOperation &Operation, const ASTContext &Context,
      std::vector<OldSymbolOccurrence> &Occurrences)
      : Operation(Operation), Context(Context), Occurrences(Occurrences) {}

  /// Returns a \c Symbol if the given declaration corresponds to the symbol
  /// that we're looking for.
  const Symbol *symbolForDecl(const Decl *D) const {
    if (!D)
      return nullptr;
    std::string USR = getUSRForDecl(D);
    return Operation.getSymbolForUSR(USR);
  }

  void checkDecl(const Decl *D, SourceLocation Loc,
                 OldSymbolOccurrence::OccurrenceKind Kind =
                     OldSymbolOccurrence::MatchingSymbol) {
    if (!D)
      return;
    std::string USR = getUSRForDecl(D);
    if (const Symbol *S = Operation.getSymbolForUSR(USR))
      checkAndAddLocations(S->SymbolIndex, Loc, Kind);
  }

  // Declaration visitors:

  bool VisitCXXConstructorDecl(clang::CXXConstructorDecl *ConstructorDecl) {
    for (const auto *Initializer : ConstructorDecl->inits()) {
      // Ignore implicit initializers.
      if (!Initializer->isWritten())
        continue;
      if (const clang::FieldDecl *FieldDecl = Initializer->getMember())
        checkDecl(FieldDecl, Initializer->getSourceLocation());
    }
    return true;
  }

  bool VisitNamedDecl(const NamedDecl *Decl) {
    checkDecl(Decl, Decl->getLocation());
    return true;
  }

  bool WalkUpFromTypedefNameDecl(const TypedefNameDecl *D) {
    // Don't visit the NamedDecl for TypedefNameDecl.
    return VisitTypedefNamedDecl(D);
  }

  bool VisitTypedefNamedDecl(const TypedefNameDecl *D) {
    if (D->isTransparentTag()) {
      if (const auto *Underlying = D->getUnderlyingType()->getAsTagDecl()) {
        checkDecl(Underlying, D->getLocation());
        return true;
      }
    }
    return VisitNamedDecl(D);
  }

  bool WalkUpFromUsingDecl(const UsingDecl *D) {
    // Don't visit the NamedDecl for UsingDecl.
    return VisitUsingDecl(D);
  }

  bool VisitUsingDecl(const UsingDecl *D) {
    for (const auto *Shadow : D->shadows()) {
      const NamedDecl *UD = Shadow->getUnderlyingDecl();
      if (UD->isImplicit() || UD == D)
        continue;
      if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(UD)) {
        UD = FTD->getTemplatedDecl();
        if (!UD)
          continue;
      }
      checkDecl(UD, D->getLocation());
    }
    return true;
  }

  bool WalkUpFromUsingDirectiveDecl(const UsingDirectiveDecl *D) {
    // Don't visit the NamedDecl for UsingDirectiveDecl.
    return VisitUsingDirectiveDecl(D);
  }

  bool VisitUsingDirectiveDecl(const UsingDirectiveDecl *D) {
    checkDecl(D->getNominatedNamespaceAsWritten(), D->getLocation());
    return true;
  }

  bool WalkUpFromUnresolvedUsingValueDecl(const UnresolvedUsingValueDecl *D) {
    // Don't visit the NamedDecl for UnresolvedUsingValueDecl.
    // FIXME: Can we try to lookup the name?
    return true;
  }

  bool
  WalkUpFromUnresolvedUsingTypenameDecl(const UnresolvedUsingTypenameDecl *D) {
    // Don't visit the NamedDecl for UnresolvedUsingTypenameDecl.
    // FIXME: Can we try to lookup the name?
    return true;
  }

  bool WalkUpFromObjCMethodDecl(const ObjCMethodDecl *Decl) {
    // Don't visit the NamedDecl for Objective-C methods.
    return VisitObjCMethodDecl(Decl);
  }

  bool VisitObjCMethodDecl(const ObjCMethodDecl *Decl) {
    const Symbol *S = symbolForDecl(Decl);
    if (!S)
      return true;
    SmallVector<SourceLocation, 8> SelectorLocs;
    Decl->getSelectorLocs(SelectorLocs);
    checkAndAddLocations(S->SymbolIndex, SelectorLocs);
    return true;
  }

  bool handleObjCProtocolList(const ObjCProtocolList &Protocols) {
    for (auto It : enumerate(Protocols))
      checkDecl(It.value(), Protocols.loc_begin()[It.index()]);
    return true;
  }

  bool VisitObjCInterfaceDecl(const ObjCInterfaceDecl *Decl) {
    if (!Decl->hasDefinition())
      return true;
    return handleObjCProtocolList(Decl->getReferencedProtocols());
  }

  bool VisitObjCProtocolDecl(const ObjCProtocolDecl *Decl) {
    if (!Decl->hasDefinition())
      return true;
    return handleObjCProtocolList(Decl->getReferencedProtocols());
  }

  bool VisitObjCCategoryDecl(const ObjCCategoryDecl *Decl) {
    checkDecl(Decl, Decl->getCategoryNameLoc());
    // The location of the class name is the location of the declaration.
    checkDecl(Decl->getClassInterface(), Decl->getLocation());
    return handleObjCProtocolList(Decl->getReferencedProtocols());
  }

  bool VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *Decl) {
    checkDecl(Decl, Decl->getCategoryNameLoc());
    // The location of the class name is the location of the declaration.
    checkDecl(Decl->getClassInterface(), Decl->getLocation());
    return true;
  }

  bool VisitObjCCompatibleAliasDecl(const ObjCCompatibleAliasDecl *Decl) {
    checkDecl(Decl->getClassInterface(), Decl->getClassInterfaceLoc());
    return true;
  }

  bool VisitObjCPropertyDecl(const ObjCPropertyDecl *Decl) {
    if (Decl->hasExplicitGetterName())
      checkDecl(Decl->getGetterMethodDecl(), Decl->getGetterNameLoc());
    if (Decl->hasExplicitSetterName())
      checkDecl(Decl->getSetterMethodDecl(), Decl->getSetterNameLoc());
    return true;
  }

  bool VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *Decl) {
    checkDecl(Decl->getPropertyDecl(), Decl->getLocation());
    if (Decl->isIvarNameSpecified())
      checkDecl(Decl->getPropertyIvarDecl(), Decl->getPropertyIvarDeclLoc());
    return true;
  }

  // Expression visitors:

  bool VisitDeclRefExpr(const DeclRefExpr *Expr) {
    checkDecl(Expr->getFoundDecl(), Expr->getLocation());
    return true;
  }

  bool VisitMemberExpr(const MemberExpr *Expr) {
    checkDecl(Expr->getFoundDecl().getDecl(), Expr->getMemberLoc());
    return true;
  }

  bool VisitObjCMessageExpr(const ObjCMessageExpr *Expr) {
    const Symbol *S = symbolForDecl(Expr->getMethodDecl());
    if (!S)
      return true;
    SmallVector<SourceLocation, 8> SelectorLocs;
    Expr->getSelectorLocs(SelectorLocs);
    checkAndAddLocations(S->SymbolIndex, SelectorLocs);
    return true;
  }

  bool VisitObjCProtocolExpr(const ObjCProtocolExpr *Expr) {
    checkDecl(Expr->getProtocol(), Expr->getProtocolIdLoc());
    return true;
  }

  bool VisitObjCIvarRefExpr(const ObjCIvarRefExpr *Expr) {
    checkDecl(Expr->getDecl(), Expr->getLocation());
    return true;
  }

  bool VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *Expr) {
    if (Expr->isClassReceiver())
      checkDecl(Expr->getClassReceiver(), Expr->getReceiverLocation());
    if (Expr->isImplicitProperty()) {
      // Class properties that are explicitly defined using @property
      // declarations are represented implicitly as there is no ivar for class
      // properties.
      if (const ObjCMethodDecl *Getter = Expr->getImplicitPropertyGetter()) {
        if (Getter->isClassMethod())
          if (const auto *PD = Getter->getCanonicalDecl()->findPropertyDecl()) {
            checkDecl(PD, Expr->getLocation());
            return true;
          }
      }

      checkDecl(Expr->getImplicitPropertyGetter(), Expr->getLocation(),
                OldSymbolOccurrence::MatchingImplicitProperty);
      // Add a manual location for a setter since a token like 'property' won't
      // match the the name of the renamed symbol like 'setProperty'.
      if (const auto *S = symbolForDecl(Expr->getImplicitPropertySetter()))
        addLocation(S->SymbolIndex, Expr->getLocation(),
                    OldSymbolOccurrence::MatchingImplicitProperty);
      return true;
    }
    checkDecl(Expr->getExplicitProperty(), Expr->getLocation());
    return true;
  }

  // Other visitors:

  bool VisitTypeLoc(const TypeLoc Loc) {
    TypedefTypeLoc TTL = Loc.getAs<TypedefTypeLoc>();
    if (TTL) {
      const auto *TND = TTL.getTypedefNameDecl();
      if (TND->isTransparentTag()) {
        if (const auto *Underlying = TND->getUnderlyingType()->getAsTagDecl()) {
          checkDecl(Underlying, TTL.getNameLoc());
          return true;
        }
      }
      checkDecl(TND, TTL.getNameLoc());
      return true;
    }
    TypeSpecTypeLoc TSTL = Loc.getAs<TypeSpecTypeLoc>();
    if (TSTL) {
      checkDecl(Loc.getType()->getAsTagDecl(), TSTL.getNameLoc());
    }
    if (const auto *TemplateTypeParm =
            dyn_cast<TemplateTypeParmType>(Loc.getType())) {
      checkDecl(TemplateTypeParm->getDecl(), Loc.getBeginLoc());
    }
    if (const auto *TemplateSpecType =
            dyn_cast<TemplateSpecializationType>(Loc.getType())) {
      checkDecl(TemplateSpecType->getTemplateName().getAsTemplateDecl(),
                Loc.getBeginLoc());
    }
    return true;
  }

  bool VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc Loc) {
    checkDecl(Loc.getIFaceDecl(), Loc.getNameLoc());
    return true;
  }

  bool VisitObjCObjectTypeLoc(ObjCObjectTypeLoc Loc) {
    for (unsigned I = 0, E = Loc.getNumProtocols(); I < E; ++I)
      checkDecl(Loc.getProtocol(I), Loc.getProtocolLoc(I));
    return true;
  }

  bool VisitDependentSymbolReference(const NamedDecl *Symbol,
                                     SourceLocation SymbolNameLoc) {
    checkDecl(Symbol, SymbolNameLoc);
    return true;
  }

  // Non-visitors:

  // Namespace traversal:
  void handleNestedNameSpecifierLoc(NestedNameSpecifierLoc NameLoc) {
    while (NameLoc) {
      checkDecl(NameLoc.getNestedNameSpecifier()->getAsNamespace(),
                NameLoc.getLocalBeginLoc());
      NameLoc = NameLoc.getPrefix();
    }
  }

private:
  size_t getOffsetForString(SourceLocation Loc, StringRef PrevNameString) {
    const SourceLocation BeginLoc = Loc;
    const SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        BeginLoc, 0, Context.getSourceManager(), Context.getLangOpts());
    StringRef TokenName =
        Lexer::getSourceText(CharSourceRange::getTokenRange(BeginLoc, EndLoc),
                             Context.getSourceManager(), Context.getLangOpts());
    return TokenName.find(PrevNameString);
  }

  void checkAndAddLocations(unsigned SymbolIndex,
                            ArrayRef<SourceLocation> Locations,
                            OldSymbolOccurrence::OccurrenceKind Kind =
                                OldSymbolOccurrence::MatchingSymbol) {
    if (Locations.size() !=
        Operation.symbols()[SymbolIndex].Name.getNamePieces().size())
      return;

    SmallVector<SourceLocation, 4> StringLocations;
    for (size_t I = 0, E = Locations.size(); I != E; ++I) {
      SourceLocation Loc = Locations[I];
      bool IsMacroExpansion = Loc.isMacroID();
      if (IsMacroExpansion) {
        const SourceManager &SM = Context.getSourceManager();
        if (SM.isMacroArgExpansion(Loc)) {
          Loc = SM.getSpellingLoc(Loc);
          IsMacroExpansion = false;
        } else
          Loc = SM.getExpansionLoc(Loc);
      }
      if (IsMacroExpansion) {
        Occurrences.push_back(OldSymbolOccurrence(
            Kind, /*IsMacroExpansion=*/true, SymbolIndex, Loc));
        return;
      }
      size_t Offset = getOffsetForString(
          Loc, Operation.symbols()[SymbolIndex].Name.getNamePieces()[I]);
      if (Offset == StringRef::npos)
        return;
      StringLocations.push_back(Loc.getLocWithOffset(Offset));
    }

    Occurrences.push_back(OldSymbolOccurrence(Kind, /*IsMacroExpansion=*/false,
                                              SymbolIndex, StringLocations));
  }

  /// Adds a location without checking if the name is actually there.
  void addLocation(unsigned SymbolIndex, SourceLocation Location,
                   OldSymbolOccurrence::OccurrenceKind Kind) {
    if (1 != Operation.symbols()[SymbolIndex].Name.getNamePieces().size())
      return;
    bool IsMacroExpansion = Location.isMacroID();
    if (IsMacroExpansion) {
      const SourceManager &SM = Context.getSourceManager();
      if (SM.isMacroArgExpansion(Location)) {
        Location = SM.getSpellingLoc(Location);
        IsMacroExpansion = false;
      } else
        Location = SM.getExpansionLoc(Location);
    }
    Occurrences.push_back(
        OldSymbolOccurrence(Kind, IsMacroExpansion, SymbolIndex, Location));
  }

  const SymbolOperation &Operation;
  const ASTContext &Context;
  std::vector<OldSymbolOccurrence> &Occurrences;
};
} // namespace

std::vector<OldSymbolOccurrence>
findSymbolOccurrences(const SymbolOperation &Operation, Decl *Decl) {
  std::vector<OldSymbolOccurrence> Occurrences;
  SymbolOccurrenceFinderASTVisitor Visitor(Operation, Decl->getASTContext(),
                                           Occurrences);
  Visitor.TraverseDecl(Decl);
  NestedNameSpecifierLocFinder Finder(Decl->getASTContext());

  for (const auto &Location : Finder.getNestedNameSpecifierLocations())
    Visitor.handleNestedNameSpecifierLoc(Location);

  return Occurrences;
}

} // end namespace rename
} // end namespace tooling
} // end namespace clang
