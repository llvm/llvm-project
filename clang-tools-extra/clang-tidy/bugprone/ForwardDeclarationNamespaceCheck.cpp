//===--- ForwardDeclarationNamespaceCheck.cpp - clang-tidy ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ForwardDeclarationNamespaceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include <stack>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void ForwardDeclarationNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  // Match all class declarations/definitions *EXCEPT*
  // 1. implicit classes, e.g. `class A {};` has implicit `class A` inside `A`.
  // 2. nested classes declared/defined inside another class.
  // 3. template class declaration, template instantiation or
  //    specialization (NOTE: extern specialization is filtered out by
  //    `unless(hasAncestor(cxxRecordDecl()))`).
  auto IsInSpecialization = hasAncestor(
      decl(anyOf(cxxRecordDecl(isExplicitTemplateSpecialization()),
                 functionDecl(isExplicitTemplateSpecialization()))));
  Finder->addMatcher(
      cxxRecordDecl(
          hasParent(decl(anyOf(namespaceDecl(), translationUnitDecl()))),
          unless(isImplicit()), unless(hasAncestor(cxxRecordDecl())),
          unless(isInstantiated()), unless(IsInSpecialization),
          unless(classTemplateSpecializationDecl()))
          .bind("record_decl"),
      this);

  // Match all friend declarations. Classes used in friend declarations are not
  // marked as referenced in AST. We need to record all record classes used in
  // friend declarations.
  Finder->addMatcher(friendDecl().bind("friend_decl"), this);
}

void ForwardDeclarationNamespaceCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *RecordDecl =
          Result.Nodes.getNodeAs<CXXRecordDecl>("record_decl")) {
    StringRef DeclName = RecordDecl->getName();
    if (RecordDecl->isThisDeclarationADefinition()) {
      DeclNameToDefinitions[DeclName].push_back(RecordDecl);
    } else {
      // If a declaration has no definition, the definition could be in another
      // namespace (a wrong namespace).
      // NOTE: even a declaration does have definition, we still need it to
      // compare with other declarations.
      DeclNameToDeclarations[DeclName].push_back(RecordDecl);
    }
  } else {
    const auto *Decl = Result.Nodes.getNodeAs<FriendDecl>("friend_decl");
    assert(Decl && "Decl is neither record_decl nor friend decl!");

    // Classes used in friend declarations are not marked referenced in AST,
    // so we need to check classes used in friend declarations manually to
    // reduce the rate of false positive.
    // For example, in
    //    \code
    //      struct A;
    //      struct B { friend A; };
    //    \endcode
    // `A` will not be marked as "referenced" in the AST.
    if (const TypeSourceInfo *Tsi = Decl->getFriendType()) {
      QualType Desugared = Tsi->getType().getDesugaredType(*Result.Context);
      FriendTypes.insert(Desugared.getTypePtr());
    }
  }
}

static bool haveSameNamespaceOrTranslationUnit(const CXXRecordDecl *Decl1,
                                               const CXXRecordDecl *Decl2) {
  const DeclContext *ParentDecl1 = Decl1->getLexicalParent();
  const DeclContext *ParentDecl2 = Decl2->getLexicalParent();

  // Since we only matched declarations whose parent is Namespace or
  // TranslationUnit declaration, the parent should be either a translation unit
  // or namespace.
  if (ParentDecl1->getDeclKind() == Decl::TranslationUnit ||
      ParentDecl2->getDeclKind() == Decl::TranslationUnit) {
    return ParentDecl1 == ParentDecl2;
  }
  assert(ParentDecl1->getDeclKind() == Decl::Namespace &&
         "ParentDecl1 declaration must be a namespace");
  assert(ParentDecl2->getDeclKind() == Decl::Namespace &&
         "ParentDecl2 declaration must be a namespace");
  auto *Ns1 = NamespaceDecl::castFromDeclContext(ParentDecl1);
  auto *Ns2 = NamespaceDecl::castFromDeclContext(ParentDecl2);
  return Ns1->getOriginalNamespace() == Ns2->getOriginalNamespace();
}

static std::string getNameOfNamespace(const CXXRecordDecl *Decl) {
  const auto *ParentDecl = Decl->getLexicalParent();
  if (ParentDecl->getDeclKind() == Decl::TranslationUnit) {
    return "(global)";
  }
  const auto *NsDecl = cast<NamespaceDecl>(ParentDecl);
  std::string Ns;
  llvm::raw_string_ostream OStream(Ns);
  NsDecl->printQualifiedName(OStream);
  OStream.flush();
  return Ns.empty() ? "(global)" : Ns;
}

void ForwardDeclarationNamespaceCheck::onEndOfTranslationUnit() {
  // Iterate each group of declarations by name.
  for (const auto &KeyValuePair : DeclNameToDeclarations) {
    const auto &Declarations = KeyValuePair.second;
    // If more than 1 declaration exists, we check if all are in the same
    // namespace.
    for (const auto *CurDecl : Declarations) {
      if (CurDecl->hasDefinition() || CurDecl->isReferenced()) {
        continue; // Skip forward declarations that are used/referenced.
      }
      if (FriendTypes.contains(CurDecl->getTypeForDecl())) {
        continue; // Skip forward declarations referenced as friend.
      }
      if (CurDecl->getLocation().isMacroID() ||
          CurDecl->getLocation().isInvalid()) {
        continue;
      }
      // Compare with all other declarations with the same name.
      for (const auto *Decl : Declarations) {
        if (Decl == CurDecl) {
          continue; // Don't compare with self.
        }
        if (!CurDecl->hasDefinition() &&
            !haveSameNamespaceOrTranslationUnit(CurDecl, Decl)) {
          diag(CurDecl->getLocation(),
               "declaration %0 is never referenced, but a declaration with "
               "the same name found in another namespace '%1'")
              << CurDecl << getNameOfNamespace(Decl);
          diag(Decl->getLocation(), "a declaration of %0 is found here",
               DiagnosticIDs::Note)
              << Decl;
          break; // FIXME: We only generate one warning for each declaration.
        }
      }
      // Check if a definition in another namespace exists.
      const auto DeclName = CurDecl->getName();
      if (!DeclNameToDefinitions.contains(DeclName)) {
        continue; // No definition in this translation unit, we can skip it.
      }
      // Make a warning for each definition with the same name (in other
      // namespaces).
      const auto &Definitions = DeclNameToDefinitions[DeclName];
      for (const auto *Def : Definitions) {
        diag(CurDecl->getLocation(),
             "no definition found for %0, but a definition with "
             "the same name %1 found in another namespace '%2'")
            << CurDecl << Def << getNameOfNamespace(Def);
        diag(Def->getLocation(), "a definition of %0 is found here",
             DiagnosticIDs::Note)
            << Def;
      }
    }
  }
}

} // namespace clang::tidy::bugprone
