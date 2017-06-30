//===--- USRLocFinder.cpp - Clang refactoring library ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Methods for finding all instances of a USR. Our strategy is very
/// simple; we just compare the USR at every relevant AST node with the one
/// provided.
///
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/Rename/USRLocFinder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Lookup.h"
#include "clang/Tooling/Refactoring/Rename/USRFinder.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <cstddef>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

namespace clang {
namespace tooling {

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

SourceLocation StartLocationForType(TypeLoc TL) {
  // For elaborated types (e.g. `struct a::A`) we want the portion after the
  // `struct` but including the namespace qualifier, `a::`.
  if (auto ElaboratedTypeLoc = TL.getAs<clang::ElaboratedTypeLoc>()) {
    NestedNameSpecifierLoc NestedNameSpecifier =
        ElaboratedTypeLoc.getQualifierLoc();
    if (NestedNameSpecifier.getNestedNameSpecifier())
      return NestedNameSpecifier.getBeginLoc();
    TL = TL.getNextTypeLoc();
  }
  return TL.getLocStart();
}

SourceLocation EndLocationForType(TypeLoc TL) {
  // Dig past any namespace or keyword qualifications.
  while (TL.getTypeLocClass() == TypeLoc::Elaborated ||
         TL.getTypeLocClass() == TypeLoc::Qualified)
    TL = TL.getNextTypeLoc();

  // The location for template specializations (e.g. Foo<int>) includes the
  // templated types in its location range.  We want to restrict this to just
  // before the `<` character.
  if (TL.getTypeLocClass() == TypeLoc::TemplateSpecialization) {
    return TL.castAs<TemplateSpecializationTypeLoc>()
        .getLAngleLoc()
        .getLocWithOffset(-1);
  }
  return TL.getEndLoc();
}

NestedNameSpecifier *GetNestedNameForType(TypeLoc TL) {
  // Dig past any keyword qualifications.
  while (TL.getTypeLocClass() == TypeLoc::Qualified)
    TL = TL.getNextTypeLoc();

  // For elaborated types (e.g. `struct a::A`) we want the portion after the
  // `struct` but including the namespace qualifier, `a::`.
  if (auto ElaboratedTypeLoc = TL.getAs<clang::ElaboratedTypeLoc>())
    return ElaboratedTypeLoc.getQualifierLoc().getNestedNameSpecifier();
  return nullptr;
}

// Find all locations identified by the given USRs for rename.
//
// This class will traverse the AST and find every AST node whose USR is in the
// given USRs' set.
class RenameLocFinder : public RecursiveASTVisitor<RenameLocFinder> {
public:
  RenameLocFinder(llvm::ArrayRef<std::string> USRs, ASTContext &Context)
      : USRSet(USRs.begin(), USRs.end()), Context(Context) {}

  // A structure records all information of a symbol reference being renamed.
  // We try to add as few prefix qualifiers as possible.
  struct RenameInfo {
    // The begin location of a symbol being renamed.
    SourceLocation Begin;
    // The end location of a symbol being renamed.
    SourceLocation End;
    // The declaration of a symbol being renamed (can be nullptr).
    const NamedDecl *FromDecl;
    // The declaration in which the nested name is contained (can be nullptr).
    const Decl *Context;
    // The nested name being replaced (can be nullptr).
    const NestedNameSpecifier *Specifier;
  };

  // FIXME: Currently, prefix qualifiers will be added to the renamed symbol
  // definition (e.g. "class Foo {};" => "class b::Bar {};" when renaming
  // "a::Foo" to "b::Bar").
  // For renaming declarations/definitions, prefix qualifiers should be filtered
  // out.
  bool VisitNamedDecl(const NamedDecl *Decl) {
    // UsingDecl has been handled in other place.
    if (llvm::isa<UsingDecl>(Decl))
      return true;

    // DestructorDecl has been handled in Typeloc.
    if (llvm::isa<CXXDestructorDecl>(Decl))
      return true;

    if (Decl->isImplicit())
      return true;

    if (isInUSRSet(Decl)) {
      RenameInfo Info = {Decl->getLocation(), Decl->getLocation(), nullptr,
                         nullptr, nullptr};
      RenameInfos.push_back(Info);
    }
    return true;
  }

  bool VisitDeclRefExpr(const DeclRefExpr *Expr) {
    const NamedDecl *Decl = Expr->getFoundDecl();
    if (isInUSRSet(Decl)) {
      RenameInfo Info = {Expr->getSourceRange().getBegin(),
                         Expr->getSourceRange().getEnd(), Decl,
                         getClosestAncestorDecl(*Expr), Expr->getQualifier()};
      RenameInfos.push_back(Info);
    }

    return true;
  }

  bool VisitUsingDecl(const UsingDecl *Using) {
    for (const auto *UsingShadow : Using->shadows()) {
      if (isInUSRSet(UsingShadow->getTargetDecl())) {
        UsingDecls.push_back(Using);
        break;
      }
    }
    return true;
  }

  bool VisitNestedNameSpecifierLocations(NestedNameSpecifierLoc NestedLoc) {
    if (!NestedLoc.getNestedNameSpecifier()->getAsType())
      return true;
    if (IsTypeAliasWhichWillBeRenamedElsewhere(NestedLoc.getTypeLoc()))
      return true;

    if (const auto *TargetDecl =
            getSupportedDeclFromTypeLoc(NestedLoc.getTypeLoc())) {
      if (isInUSRSet(TargetDecl)) {
        RenameInfo Info = {NestedLoc.getBeginLoc(),
                           EndLocationForType(NestedLoc.getTypeLoc()),
                           TargetDecl, getClosestAncestorDecl(NestedLoc),
                           NestedLoc.getNestedNameSpecifier()->getPrefix()};
        RenameInfos.push_back(Info);
      }
    }
    return true;
  }

  bool VisitTypeLoc(TypeLoc Loc) {
    if (IsTypeAliasWhichWillBeRenamedElsewhere(Loc))
      return true;

    auto Parents = Context.getParents(Loc);
    TypeLoc ParentTypeLoc;
    if (!Parents.empty()) {
      // Handle cases of nested name specificier locations.
      //
      // The VisitNestedNameSpecifierLoc interface is not impelmented in
      // RecursiveASTVisitor, we have to handle it explicitly.
      if (const auto *NSL = Parents[0].get<NestedNameSpecifierLoc>()) {
        VisitNestedNameSpecifierLocations(*NSL);
        return true;
      }

      if (const auto *TL = Parents[0].get<TypeLoc>())
        ParentTypeLoc = *TL;
    }

    // Handle the outermost TypeLoc which is directly linked to the interesting
    // declaration and don't handle nested name specifier locations.
    if (const auto *TargetDecl = getSupportedDeclFromTypeLoc(Loc)) {
      if (isInUSRSet(TargetDecl)) {
        // Only handle the outermost typeLoc.
        //
        // For a type like "a::Foo", there will be two typeLocs for it.
        // One ElaboratedType, the other is RecordType:
        //
        //   ElaboratedType 0x33b9390 'a::Foo' sugar
        //   `-RecordType 0x338fef0 'class a::Foo'
        //     `-CXXRecord 0x338fe58 'Foo'
        //
        // Skip if this is an inner typeLoc.
        if (!ParentTypeLoc.isNull() &&
            isInUSRSet(getSupportedDeclFromTypeLoc(ParentTypeLoc)))
          return true;
        RenameInfo Info = {StartLocationForType(Loc), EndLocationForType(Loc),
                           TargetDecl, getClosestAncestorDecl(Loc),
                           GetNestedNameForType(Loc)};
        RenameInfos.push_back(Info);
        return true;
      }
    }

    // Handle specific template class specialiation cases.
    if (const auto *TemplateSpecType =
            dyn_cast<TemplateSpecializationType>(Loc.getType())) {
      TypeLoc TargetLoc = Loc;
      if (!ParentTypeLoc.isNull()) {
        if (llvm::isa<ElaboratedType>(ParentTypeLoc.getType()))
          TargetLoc = ParentTypeLoc;
      }

      if (isInUSRSet(TemplateSpecType->getTemplateName().getAsTemplateDecl())) {
        TypeLoc TargetLoc = Loc;
        // FIXME: Find a better way to handle this case.
        // For the qualified template class specification type like
        // "ns::Foo<int>" in "ns::Foo<int>& f();", we want the parent typeLoc
        // (ElaboratedType) of the TemplateSpecializationType in order to
        // catch the prefix qualifiers "ns::".
        if (!ParentTypeLoc.isNull() &&
            llvm::isa<ElaboratedType>(ParentTypeLoc.getType()))
          TargetLoc = ParentTypeLoc;
        RenameInfo Info = {
            StartLocationForType(TargetLoc), EndLocationForType(TargetLoc),
            TemplateSpecType->getTemplateName().getAsTemplateDecl(),
            getClosestAncestorDecl(
                ast_type_traits::DynTypedNode::create(TargetLoc)),
            GetNestedNameForType(TargetLoc)};
        RenameInfos.push_back(Info);
      }
    }
    return true;
  }

  // Returns a list of RenameInfo.
  const std::vector<RenameInfo> &getRenameInfos() const { return RenameInfos; }

  // Returns a list of using declarations which are needed to update.
  const std::vector<const UsingDecl *> &getUsingDecls() const {
    return UsingDecls;
  }

private:
  // FIXME: This method may not be suitable for renaming other types like alias
  // types. Need to figure out a way to handle it.
  bool IsTypeAliasWhichWillBeRenamedElsewhere(TypeLoc TL) const {
    while (!TL.isNull()) {
      // SubstTemplateTypeParm is the TypeLocation class for a substituted type
      // inside a template expansion so we ignore these.  For example:
      //
      // template<typename T> struct S {
      //   T t;  // <-- this T becomes a TypeLoc(int) with class
      //         //     SubstTemplateTypeParm when S<int> is instantiated
      // }
      if (TL.getTypeLocClass() == TypeLoc::SubstTemplateTypeParm)
        return true;

      // Typedef is the TypeLocation class for a type which is a typedef to the
      // type we want to replace.  We ignore the use of the typedef as we will
      // replace the definition of it.  For example:
      //
      // typedef int T;
      // T a;  // <---  This T is a TypeLoc(int) with class Typedef.
      if (TL.getTypeLocClass() == TypeLoc::Typedef)
        return true;
      TL = TL.getNextTypeLoc();
    }
    return false;
  }

  // Get the supported declaration from a given typeLoc. If the declaration type
  // is not supported, returns nullptr.
  //
  // FIXME: support more types, e.g. enum, type alias.
  const NamedDecl *getSupportedDeclFromTypeLoc(TypeLoc Loc) {
    if (const auto *RD = Loc.getType()->getAsCXXRecordDecl())
      return RD;
    return nullptr;
  }

  // Get the closest ancester which is a declaration of a given AST node.
  template <typename ASTNodeType>
  const Decl *getClosestAncestorDecl(const ASTNodeType &Node) {
    auto Parents = Context.getParents(Node);
    // FIXME: figure out how to handle it when there are multiple parents.
    if (Parents.size() != 1)
      return nullptr;
    if (ast_type_traits::ASTNodeKind::getFromNodeKind<Decl>().isBaseOf(
            Parents[0].getNodeKind()))
      return Parents[0].template get<Decl>();
    return getClosestAncestorDecl(Parents[0]);
  }

  // Get the parent typeLoc of a given typeLoc. If there is no such parent,
  // return nullptr.
  const TypeLoc *getParentTypeLoc(TypeLoc Loc) const {
    auto Parents = Context.getParents(Loc);
    // FIXME: figure out how to handle it when there are multiple parents.
    if (Parents.size() != 1)
      return nullptr;
    return Parents[0].get<TypeLoc>();
  }

  // Check whether the USR of a given Decl is in the USRSet.
  bool isInUSRSet(const Decl *Decl) const {
    auto USR = getUSRForDecl(Decl);
    if (USR.empty())
      return false;
    return llvm::is_contained(USRSet, USR);
  }

  const std::set<std::string> USRSet;
  ASTContext &Context;
  std::vector<RenameInfo> RenameInfos;
  // Record all interested using declarations which contains the using-shadow
  // declarations of the symbol declarations being renamed.
  std::vector<const UsingDecl *> UsingDecls;
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

std::vector<tooling::AtomicChange>
createRenameAtomicChanges(llvm::ArrayRef<std::string> USRs,
                          llvm::StringRef NewName, Decl *TranslationUnitDecl) {
  RenameLocFinder Finder(USRs, TranslationUnitDecl->getASTContext());
  Finder.TraverseDecl(TranslationUnitDecl);

  const SourceManager &SM =
      TranslationUnitDecl->getASTContext().getSourceManager();

  std::vector<tooling::AtomicChange> AtomicChanges;
  auto Replace = [&](SourceLocation Start, SourceLocation End,
                     llvm::StringRef Text) {
    tooling::AtomicChange ReplaceChange = tooling::AtomicChange(SM, Start);
    llvm::Error Err = ReplaceChange.replace(
        SM, CharSourceRange::getTokenRange(Start, End), Text);
    if (Err) {
      llvm::errs() << "Faile to add replacement to AtomicChange: "
                   << llvm::toString(std::move(Err)) << "\n";
      return;
    }
    AtomicChanges.push_back(std::move(ReplaceChange));
  };

  for (const auto &RenameInfo : Finder.getRenameInfos()) {
    std::string ReplacedName = NewName.str();
    if (RenameInfo.FromDecl && RenameInfo.Context) {
      if (!llvm::isa<clang::TranslationUnitDecl>(
              RenameInfo.Context->getDeclContext())) {
        ReplacedName = tooling::replaceNestedName(
            RenameInfo.Specifier, RenameInfo.Context->getDeclContext(),
            RenameInfo.FromDecl,
            NewName.startswith("::") ? NewName.str() : ("::" + NewName).str());
      }
    }
    // If the NewName contains leading "::", add it back.
    if (NewName.startswith("::") && NewName.substr(2) == ReplacedName)
      ReplacedName = NewName.str();
    Replace(RenameInfo.Begin, RenameInfo.End, ReplacedName);
  }

  // Hanlde using declarations explicitly as "using a::Foo" don't trigger
  // typeLoc for "a::Foo".
  for (const auto *Using : Finder.getUsingDecls())
    Replace(Using->getLocStart(), Using->getLocEnd(), "using " + NewName.str());

  return AtomicChanges;
}

} // end namespace tooling
} // end namespace clang
