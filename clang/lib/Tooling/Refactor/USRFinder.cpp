//===--- USRFinder.cpp - Clang refactoring library ------------------------===//
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

#include "clang/Tooling/Refactor/USRFinder.h"
#include "SourceLocationUtilities.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DependentASTVisitor.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Refactoring/RefactoringDiagnostic.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

using namespace llvm;

namespace clang {
namespace tooling {
namespace rename {

typedef std::function<bool(const NamedDecl *, SourceLocation, SourceLocation)>
    OccurrenceCheckerType;

// NamedDeclFindingASTVisitor recursively visits each AST node to find the
// symbol underneath the cursor.
// FIXME: move to seperate .h/.cc file if this gets too large.
namespace {
class NamedDeclFindingASTVisitor
    : public DependentASTVisitor<NamedDeclFindingASTVisitor> {
public:
  // \brief Finds the NamedDecl at a point in the source.
  // \param Point the location in the source to search for the NamedDecl.
  explicit NamedDeclFindingASTVisitor(
      const OccurrenceCheckerType &OccurrenceChecker, const ASTContext &Context)
      : Result(nullptr), OccurrenceChecker(OccurrenceChecker),
        Context(Context) {}

  // Declaration visitors:

  // \brief Checks if the point falls within the NameDecl. This covers every
  // declaration of a named entity that we may come across. Usually, just
  // checking if the point lies within the length of the name of the declaration
  // and the start location is sufficient.
  bool VisitNamedDecl(const NamedDecl *Decl) {
    return dyn_cast<CXXConversionDecl>(Decl)
               ? true
               : checkOccurrence(Decl, Decl->getLocation(),
                                 Decl->getNameAsString().length());
  }

  bool WalkUpFromTypedefNameDecl(const TypedefNameDecl *D) {
    // Don't visit the NamedDecl for TypedefNameDecl.
    return VisitTypedefNamedDecl(D);
  }

  bool VisitTypedefNamedDecl(const TypedefNameDecl *D) {
    if (D->isTransparentTag()) {
      if (const auto *Underlying = D->getUnderlyingType()->getAsTagDecl())
        return checkOccurrence(Underlying, D->getLocation(),
                               D->getNameAsString().size());
    }
    return VisitNamedDecl(D);
  }

  bool WalkUpFromUsingDecl(const UsingDecl *D) {
    // Don't visit the NamedDecl for UsingDecl.
    return VisitUsingDecl(D);
  }

  bool VisitUsingDecl(const UsingDecl *D) {
    for (const auto *Shadow : D->shadows()) {
      // Currently we always find the first declaration, but is this the right
      // behaviour?
      const NamedDecl *UD = Shadow->getUnderlyingDecl();
      if (UD->isImplicit() || UD == D)
        continue;
      if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(UD)) {
        UD = FTD->getTemplatedDecl();
        if (!UD)
          continue;
      }
      if (!checkOccurrence(UD, D->getLocation()))
        return false;
    }
    return true;
  }

  bool WalkUpFromUsingDirectiveDecl(const UsingDirectiveDecl *D) {
    // Don't visit the NamedDecl for UsingDirectiveDecl.
    return VisitUsingDirectiveDecl(D);
  }

  bool VisitUsingDirectiveDecl(const UsingDirectiveDecl *D) {
    return checkOccurrence(D->getNominatedNamespaceAsWritten(),
                           D->getLocation());
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
    // Check all of the selector source ranges.
    for (unsigned I = 0, E = Decl->getNumSelectorLocs(); I != E; ++I) {
      SourceLocation Loc = Decl->getSelectorLoc(I);
      if (!checkOccurrence(Decl, Loc,
                           Loc.getLocWithOffset(
                               Decl->getSelector().getNameForSlot(I).size())))
        return false;
    }
    return true;
  }

  bool VisitObjCProtocolList(const ObjCProtocolList &Protocols) {
    for (unsigned I = 0, E = Protocols.size(); I != E; ++I) {
      if (!checkOccurrence(Protocols[I], Protocols.loc_begin()[I]))
        return false;
    }
    return true;
  }

  bool VisitObjCInterfaceDecl(const ObjCInterfaceDecl *Decl) {
    if (!Decl->hasDefinition())
      return true;
    return VisitObjCProtocolList(Decl->getReferencedProtocols());
  }

  bool VisitObjCProtocolDecl(const ObjCProtocolDecl *Decl) {
    if (!Decl->hasDefinition())
      return true;
    return VisitObjCProtocolList(Decl->getReferencedProtocols());
  }

  bool WalkUpFromObjCCategoryDecl(const ObjCCategoryDecl *Decl) {
    // Don't visit the NamedDecl for Objective-C categories because the location
    // of the name refers to the interface declaration.
    return VisitObjCCategoryDecl(Decl);
  }

  bool VisitObjCCategoryDecl(const ObjCCategoryDecl *Decl) {
    if (!checkOccurrence(Decl, Decl->getCategoryNameLoc()))
      return false;
    if (const auto *Class = Decl->getClassInterface()) {
      // The location of the class name is the location of the declaration.
      if (!checkOccurrence(Class, Decl->getLocation()))
        return false;
    }
    return VisitObjCProtocolList(Decl->getReferencedProtocols());
  }

  bool WalkUpFromObjCCategoryImplDecl(const ObjCCategoryImplDecl *Decl) {
    // Don't visit the NamedDecl for Objective-C categories because the location
    // of the name refers to the interface declaration.
    return VisitObjCCategoryImplDecl(Decl);
  }

  bool VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *Decl) {
    if (!checkOccurrence(Decl, Decl->getCategoryNameLoc()))
      return false;
    if (const auto *Class = Decl->getClassInterface()) {
      // The location of the class name is the location of the declaration.
      if (!checkOccurrence(Class, Decl->getLocation()))
        return false;
    }
    return true;
  }

  bool VisitObjCCompatibleAliasDecl(const ObjCCompatibleAliasDecl *Decl) {
    return checkOccurrence(Decl->getClassInterface(),
                           Decl->getClassInterfaceLoc());
  }

  bool WalkUpFromObjCIvarDecl(ObjCIvarDecl *Decl) {
    // Don't visit the NamedDecl for automatically synthesized ivars as the
    // implicit ivars have the same location as the property declarations, and
    // we want to find the property declarations.
    if (Decl->getSynthesize())
      return true;
    return RecursiveASTVisitor::WalkUpFromObjCIvarDecl(Decl);
  }

  bool VisitObjCPropertyDecl(const ObjCPropertyDecl *Decl) {
    if (Decl->hasExplicitGetterName()) {
      if (const auto *Getter = Decl->getGetterMethodDecl())
        if (!checkOccurrence(Getter, Decl->getGetterNameLoc(),
                             Decl->getGetterName().getNameForSlot(0).size()))
          return false;
    }
    if (Decl->hasExplicitSetterName()) {
      if (const auto *Setter = Decl->getSetterMethodDecl())
        return checkOccurrence(Setter, Decl->getSetterNameLoc(),
                               Decl->getSetterName().getNameForSlot(0).size());
    }
    return true;
  }

  bool VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *Decl) {
    if (!checkOccurrence(Decl->getPropertyDecl(), Decl->getLocation()))
      return false;
    if (Decl->isIvarNameSpecified())
      return checkOccurrence(Decl->getPropertyIvarDecl(),
                             Decl->getPropertyIvarDeclLoc());
    return true;
  }

  // Expression visitors:

  bool VisitDeclRefExpr(const DeclRefExpr *Expr) {
    const NamedDecl *Decl = Expr->getFoundDecl();
    return checkOccurrence(Decl, Expr->getLocation(),
                           Decl->getNameAsString().length());
  }

  bool VisitMemberExpr(const MemberExpr *Expr) {
    const NamedDecl *Decl = Expr->getFoundDecl().getDecl();
    return checkOccurrence(Decl, Expr->getMemberLoc(),
                           Decl->getNameAsString().length());
  }

  bool VisitObjCMessageExpr(const ObjCMessageExpr *Expr) {
    const ObjCMethodDecl *Decl = Expr->getMethodDecl();
    if (Decl == nullptr)
      return true;

    // Check all of the selector source ranges.
    for (unsigned I = 0, E = Expr->getNumSelectorLocs(); I != E; ++I) {
      SourceLocation Loc = Expr->getSelectorLoc(I);
      if (!checkOccurrence(Decl, Loc,
                           Loc.getLocWithOffset(
                               Decl->getSelector().getNameForSlot(I).size())))
        return false;
    }
    return true;
  }

  bool VisitObjCProtocolExpr(const ObjCProtocolExpr *Expr) {
    return checkOccurrence(Expr->getProtocol(), Expr->getProtocolIdLoc());
  }

  bool VisitObjCIvarRefExpr(const ObjCIvarRefExpr *Expr) {
    return checkOccurrence(Expr->getDecl(), Expr->getLocation());
  }

  bool VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *Expr) {
    if (Expr->isClassReceiver())
      checkOccurrence(Expr->getClassReceiver(), Expr->getReceiverLocation());
    if (Expr->isImplicitProperty()) {
      // Class properties that are explicitly defined using @property
      // declarations are represented implicitly as there is no ivar for class
      // properties.
      if (const ObjCMethodDecl *Getter = Expr->getImplicitPropertyGetter()) {
        if (Getter->isClassMethod()) {
          if (const auto *PD = Getter->getCanonicalDecl()->findPropertyDecl())
            return checkOccurrence(PD, Expr->getLocation());
        }
      }

      if (Expr->isMessagingGetter()) {
        if (const ObjCMethodDecl *Getter = Expr->getImplicitPropertyGetter())
          return checkOccurrence(Getter, Expr->getLocation());
      } else if (const ObjCMethodDecl *Setter =
                     Expr->getImplicitPropertySetter()) {
        return checkOccurrence(Setter, Expr->getLocation());
      }

      return true;
    }
    return checkOccurrence(Expr->getExplicitProperty(), Expr->getLocation());
  }

  // Other visitors:

  bool VisitTypeLoc(const TypeLoc Loc) {
    const SourceLocation TypeBeginLoc = Loc.getBeginLoc();
    const SourceLocation TypeEndLoc = Lexer::getLocForEndOfToken(
        TypeBeginLoc, 0, Context.getSourceManager(), Context.getLangOpts());
    if (const auto *TemplateTypeParm =
            dyn_cast<TemplateTypeParmType>(Loc.getType()))
      return checkOccurrence(TemplateTypeParm->getDecl(), TypeBeginLoc,
                             TypeEndLoc);
    if (const auto *TemplateSpecType =
            dyn_cast<TemplateSpecializationType>(Loc.getType())) {
      return checkOccurrence(
          TemplateSpecType->getTemplateName().getAsTemplateDecl(), TypeBeginLoc,
          TypeEndLoc);
    }
    TypedefTypeLoc TTL = Loc.getAs<TypedefTypeLoc>();
    if (TTL) {
      const auto *TND = TTL.getTypedefNameDecl();
      if (TND->isTransparentTag()) {
        if (const auto *Underlying = TND->getUnderlyingType()->getAsTagDecl())
          return checkOccurrence(Underlying, TTL.getNameLoc());
      }
      return checkOccurrence(TND, TTL.getNameLoc());
    }
    TypeSpecTypeLoc TSTL = Loc.getAs<TypeSpecTypeLoc>();
    if (TSTL) {
      return checkOccurrence(Loc.getType()->getAsTagDecl(), TSTL.getNameLoc());
    }
    return true;
  }

  bool VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc Loc) {
    return checkOccurrence(Loc.getIFaceDecl(), Loc.getNameLoc());
  }

  bool VisitObjCObjectTypeLoc(ObjCObjectTypeLoc Loc) {
    for (unsigned I = 0, E = Loc.getNumProtocols(); I < E; ++I) {
      if (!checkOccurrence(Loc.getProtocol(I), Loc.getProtocolLoc(I)))
        return false;
    }
    return true;
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
        if (!checkOccurrence(FieldDecl, InitBeginLoc, InitEndLoc))
          return false;
      }
    }
    return true;
  }

  bool VisitDependentSymbolReference(const NamedDecl *Symbol,
                                     SourceLocation SymbolNameLoc) {
    return checkOccurrence(Symbol, SymbolNameLoc);
  }

  // Other:

  const NamedDecl *getNamedDecl() { return Result; }

  bool isDone() const { return Result; }

  // \brief Determines if a namespace qualifier contains the point.
  // \returns false on success and sets Result.
  void handleNestedNameSpecifierLoc(NestedNameSpecifierLoc NameLoc) {
    while (NameLoc) {
      const NamespaceDecl *Decl =
          NameLoc.getNestedNameSpecifier()->getAsNamespace();
      checkOccurrence(Decl, NameLoc.getLocalBeginLoc(),
                      NameLoc.getLocalEndLoc());
      NameLoc = NameLoc.getPrefix();
    }
  }

private:
  /// \brief Sets Result to \p Decl if the occurrence checker returns true.
  ///
  /// \returns false on success.
  bool checkRange(const NamedDecl *Decl, SourceLocation Start,
                  SourceLocation End) {
    assert(!Start.isMacroID() && !End.isMacroID() && "Macro location?");
    if (!Decl)
      return true;
    if (isa<ImplicitParamDecl>(Decl))
      return true;
    if (const auto *FD = dyn_cast<FunctionDecl>(Decl)) {
      // Don't match operators.
      if (FD->isOverloadedOperator())
        return true;
    }
    if (!OccurrenceChecker(Decl, Start, End))
      return true;
    Result = Decl;
    return false;
  }

  /// Checks if the given declaration is valid, and if it is, sets Result to
  /// \p Decl if the occurrence checker returns true.
  ///
  /// \returns false if the point of interest is inside the range that
  /// corresponds the occurrence of this declaration.
  bool checkOccurrence(const NamedDecl *Decl, SourceLocation Loc) {
    if (!Decl)
      return true;
    return checkOccurrence(Decl, Loc, Decl->getNameAsString().size());
  }

  /// \brief Sets Result to \p Decl if the occurrence checker returns true.
  ///
  /// \returns false on success.
  bool checkOccurrence(const NamedDecl *Decl, SourceLocation Loc,
                       unsigned Length) {
    if (Loc.isMacroID()) {
      const SourceManager &SM = Context.getSourceManager();
      if (SM.isMacroArgExpansion(Loc))
        Loc = SM.getSpellingLoc(Loc);
      else
        return true;
    }

    return Length == 0 ||
           checkRange(Decl, Loc, Loc.getLocWithOffset(Length - 1));
  }

  bool checkOccurrence(const NamedDecl *ND, SourceLocation Start,
                       SourceLocation End) {
    const SourceManager &SM = Context.getSourceManager();
    if (Start.isMacroID()) {
      if (SM.isMacroArgExpansion(Start))
        Start = SM.getSpellingLoc(Start);
      else
        return true;
    }
    if (End.isMacroID()) {
      if (SM.isMacroArgExpansion(End))
        End = SM.getSpellingLoc(End);
      else
        return true;
    }
    return checkRange(ND, Start, End);
  }

  const NamedDecl *Result;
  const OccurrenceCheckerType &OccurrenceChecker;
  const ASTContext &Context;
};

} // namespace

static const ExternalSourceSymbolAttr *getExternalSymAttr(const Decl *D) {
  if (const auto *A = D->getAttr<ExternalSourceSymbolAttr>())
    return A;
  if (const auto *DCD = dyn_cast<Decl>(D->getDeclContext())) {
    if (const auto *A = DCD->getAttr<ExternalSourceSymbolAttr>())
      return A;
  }
  return nullptr;
}

static bool overridesSystemMethod(const ObjCMethodDecl *MD,
                                  const SourceManager &SM) {
  SmallVector<const ObjCMethodDecl *, 8> Overrides;
  MD->getOverriddenMethods(Overrides);
  for (const auto *Override : Overrides) {
    SourceLocation Loc = Override->getBeginLoc();
    if (Loc.isValid()) {
      if (SM.getFileCharacteristic(Loc) != SrcMgr::C_User)
        return true;
    }
    if (overridesSystemMethod(Override, SM))
      return true;
  }
  return false;
}

// TODO: Share with the indexer?
static bool isTemplateImplicitInstantiation(const Decl *D) {
  TemplateSpecializationKind TKind = TSK_Undeclared;
  if (const ClassTemplateSpecializationDecl *SD =
          dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    TKind = SD->getSpecializationKind();
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    TKind = FD->getTemplateSpecializationKind();
  } else if (auto *VD = dyn_cast<VarDecl>(D)) {
    TKind = VD->getTemplateSpecializationKind();
  } else if (const auto *RD = dyn_cast<CXXRecordDecl>(D)) {
    if (RD->getInstantiatedFromMemberClass())
      TKind = RD->getTemplateSpecializationKind();
  } else if (const auto *ED = dyn_cast<EnumDecl>(D)) {
    if (ED->getInstantiatedFromMemberEnum())
      TKind = ED->getTemplateSpecializationKind();
  } else if (isa<FieldDecl>(D) || isa<TypedefNameDecl>(D) ||
             isa<EnumConstantDecl>(D)) {
    if (const auto *Parent = dyn_cast<Decl>(D->getDeclContext()))
      return isTemplateImplicitInstantiation(Parent);
  }
  switch (TKind) {
  case TSK_Undeclared:
  case TSK_ExplicitSpecialization:
    return false;
  case TSK_ImplicitInstantiation:
  case TSK_ExplicitInstantiationDeclaration:
  case TSK_ExplicitInstantiationDefinition:
    return true;
  }
  llvm_unreachable("invalid TemplateSpecializationKind");
}

static const CXXRecordDecl *
getDeclContextForTemplateInstationPattern(const Decl *D) {
  if (const auto *CTSD =
          dyn_cast<ClassTemplateSpecializationDecl>(D->getDeclContext()))
    return CTSD->getTemplateInstantiationPattern();
  else if (const auto *RD = dyn_cast<CXXRecordDecl>(D->getDeclContext()))
    return RD->getInstantiatedFromMemberClass();
  return nullptr;
}

static const NamedDecl *
adjustTemplateImplicitInstantiation(const NamedDecl *D) {
  if (const ClassTemplateSpecializationDecl *SD =
          dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    return SD->getTemplateInstantiationPattern();
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    return FD->getTemplateInstantiationPattern();
  } else if (auto *VD = dyn_cast<VarDecl>(D)) {
    return VD->getTemplateInstantiationPattern();
  } else if (const auto *RD = dyn_cast<CXXRecordDecl>(D)) {
    return RD->getInstantiatedFromMemberClass();
  } else if (const auto *ED = dyn_cast<EnumDecl>(D)) {
    return ED->getInstantiatedFromMemberEnum();
  } else if (isa<FieldDecl>(D) || isa<TypedefNameDecl>(D)) {
    const auto *ND = cast<NamedDecl>(D);
    if (const CXXRecordDecl *Pattern =
            getDeclContextForTemplateInstationPattern(ND)) {
      for (const NamedDecl *BaseND : Pattern->lookup(ND->getDeclName())) {
        if (BaseND->isImplicit())
          continue;
        if (BaseND->getKind() == ND->getKind())
          return BaseND;
      }
    }
  } else if (const auto *ECD = dyn_cast<EnumConstantDecl>(D)) {
    if (const auto *ED = dyn_cast<EnumDecl>(ECD->getDeclContext())) {
      if (const EnumDecl *Pattern = ED->getInstantiatedFromMemberEnum()) {
        for (const NamedDecl *BaseECD : Pattern->lookup(ECD->getDeclName()))
          return BaseECD;
      }
    }
  }
  return D;
}

const NamedDecl *getNamedDeclAt(const ASTContext &Context,
                                SourceLocation Point) {
  if (Point.isMacroID())
    Point = Context.getSourceManager().getSpellingLoc(Point);
  // FIXME: If point is in a system header, return early here.

  OccurrenceCheckerType PointChecker = [Point, &Context](
      const NamedDecl *Decl, SourceLocation Start, SourceLocation End) -> bool {
    return Start.isValid() && Start.isFileID() && End.isValid() &&
           End.isFileID() &&
           isPointWithin(Point, Start, End, Context.getSourceManager());
  };
  NamedDeclFindingASTVisitor Visitor(PointChecker, Context);

  // We only want to search the decls that exist in the same file as the point.
  FileID InitiationFile = Context.getSourceManager().getFileID(Point);
  for (auto *CurrDecl : Context.getTranslationUnitDecl()->decls()) {
    const SourceRange DeclRange = CurrDecl->getSourceRange();
    SourceLocation FileLoc;
    if (DeclRange.getBegin().isMacroID() && !DeclRange.getEnd().isMacroID())
      FileLoc = DeclRange.getEnd();
    else
      FileLoc = Context.getSourceManager().getSpellingLoc(DeclRange.getBegin());
    // FIXME: Add test.
    if (Context.getSourceManager().getFileID(FileLoc) == InitiationFile)
      Visitor.TraverseDecl(CurrDecl);
    if (Visitor.isDone())
      break;
  }

  if (!Visitor.isDone()) {
    NestedNameSpecifierLocFinder Finder(const_cast<ASTContext &>(Context));
    for (const auto &Location : Finder.getNestedNameSpecifierLocations()) {
      Visitor.handleNestedNameSpecifierLoc(Location);
      if (Visitor.isDone())
        break;
    }
  }

  const auto Diag = [&](unsigned DiagID) -> DiagnosticBuilder {
    return Context.getDiagnostics().Report(Point, DiagID);
  };
  const auto *ND = Visitor.getNamedDecl();
  if (!ND)
    return nullptr;

  // Canonicalize the found declaration.
  //
  // If FoundDecl is a constructor or destructor, we want to instead take
  // the Decl of the corresponding class.
  if (const auto *CtorDecl = dyn_cast<CXXConstructorDecl>(ND))
    ND = CtorDecl->getParent();
  else if (const auto *DtorDecl = dyn_cast<CXXDestructorDecl>(ND))
    ND = DtorDecl->getParent();

  if (isTemplateImplicitInstantiation(ND))
    ND = adjustTemplateImplicitInstantiation(ND);

  // Builtins can't be renamed.
  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    if (FD->getBuiltinID()) {
      Diag(diag::err_rename_builtin_function) << ND->getDeclName();
      return nullptr;
    }
  }
  // Declarations with invalid locations are probably implicit.
  if (ND->getBeginLoc().isInvalid())
    return nullptr;
  // Declarations in system headers can't be renamed.
  auto CheckSystemLoc = [&](SourceLocation Loc) -> bool {
    if (Context.getSourceManager().getFileCharacteristic(Loc) !=
        SrcMgr::C_User) {
      Diag(diag::err_rename_sys_header) << ND->getDeclName();
      return true;
    }
    return false;
  };
  if (CheckSystemLoc(ND->getBeginLoc()))
    return nullptr;
  if (const auto *TD = dyn_cast<TypedefNameDecl>(ND)) {
    if (const TypedefNameDecl *CTD = TD->getCanonicalDecl()) {
      if (CheckSystemLoc(CTD->getBeginLoc()))
        return nullptr;
    }
  } else if (const auto *TD = dyn_cast<TagDecl>(ND)) {
    if (const TagDecl *CTD = TD->getCanonicalDecl()) {
      if (CheckSystemLoc(CTD->getBeginLoc()))
        return nullptr;
    }
  } else if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    if (const FunctionDecl *CFD = FD->getCanonicalDecl()) {
      if (CheckSystemLoc(CFD->getBeginLoc()))
        return nullptr;
    }
  } else if (const auto *VD = dyn_cast<VarDecl>(ND)) {
    if (const VarDecl *CVD = VD->getCanonicalDecl()) {
      if (CheckSystemLoc(CVD->getBeginLoc()))
        return nullptr;
    }
  }
  // Declarations from other languages can't be renamed.
  if (const ExternalSourceSymbolAttr *ESSA = getExternalSymAttr(ND)) {
    Diag(diag::err_rename_external_source_symbol) << ND->getDeclName()
                                                  << ESSA->getLanguage();
    return nullptr;
  }
  // Methods that override methods from system headers can't be renamed.
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(ND)) {
    if (overridesSystemMethod(MD, Context.getSourceManager())) {
      Diag(diag::err_method_rename_override_sys_framework) << ND->getDeclName();
      return nullptr;
    }
  }
  return ND;
}

const NamedDecl *getNamedDeclWithUSR(const ASTContext &Context, StringRef USR) {
  // TODO: Remove in favour of the new converter.
  OccurrenceCheckerType USRChecker =
      [USR](const NamedDecl *Decl, SourceLocation Start, SourceLocation End) {
        return USR == getUSRForDecl(Decl);
      };
  NamedDeclFindingASTVisitor Visitor(USRChecker, Context);

  for (auto *CurrDecl : Context.getTranslationUnitDecl()->decls()) {
    Visitor.TraverseDecl(CurrDecl);
    if (Visitor.isDone())
      break;
  }

  // Don't need to visit nested name specifiers as they refer to previously
  // declared declarations that we've already seen.
  return Visitor.getNamedDecl();
}

std::string getUSRForDecl(const Decl *Decl) {
  llvm::SmallVector<char, 128> Buff;

  // FIXME: Add test for the nullptr case.
  if (Decl == nullptr || index::generateUSRForDecl(Decl, Buff))
    return "";

  return std::string(Buff.data(), Buff.size());
}

} // end namespace rename
} // end namespace tooling
} // end namespace clang
