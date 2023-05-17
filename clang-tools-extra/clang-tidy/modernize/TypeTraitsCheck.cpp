//===--- TypeTraitsCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeTraitsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static const llvm::StringSet<> ValueTraits = {
    "alignment_of",
    "conjunction",
    "disjunction",
    "extent",
    "has_unique_object_representations",
    "has_virtual_destructor",
    "is_abstract",
    "is_aggregate",
    "is_arithmetic",
    "is_array",
    "is_assignable",
    "is_base_of",
    "is_bounded_array",
    "is_class",
    "is_compound",
    "is_const",
    "is_constructible",
    "is_convertible",
    "is_copy_assignable",
    "is_copy_constructible",
    "is_default_constructible",
    "is_destructible",
    "is_empty",
    "is_enum",
    "is_final",
    "is_floating_point",
    "is_function",
    "is_fundamental",
    "is_integral",
    "is_invocable",
    "is_invocable_r",
    "is_layout_compatible",
    "is_lvalue_reference",
    "is_member_function_pointer",
    "is_member_object_pointer",
    "is_member_pointer",
    "is_move_assignable",
    "is_move_constructible",
    "is_nothrow_assignable",
    "is_nothrow_constructible",
    "is_nothrow_convertible",
    "is_nothrow_copy_assignable",
    "is_nothrow_copy_constructible",
    "is_nothrow_default_constructible",
    "is_nothrow_destructible",
    "is_nothrow_invocable",
    "is_nothrow_invocable_r",
    "is_nothrow_move_assignable",
    "is_nothrow_move_constructible",
    "is_nothrow_swappable",
    "is_nothrow_swappable_with",
    "is_null_pointer",
    "is_object",
    "is_pointer",
    "is_pointer_interconvertible_base_of",
    "is_polymorphic",
    "is_reference",
    "is_rvalue_reference",
    "is_same",
    "is_scalar",
    "is_scoped_enum",
    "is_signed",
    "is_standard_layout",
    "is_swappable",
    "is_swappable_with",
    "is_trivial",
    "is_trivially_assignable",
    "is_trivially_constructible",
    "is_trivially_copy_assignable",
    "is_trivially_copy_constructible",
    "is_trivially_copyable",
    "is_trivially_default_constructible",
    "is_trivially_destructible",
    "is_trivially_move_assignable",
    "is_trivially_move_constructible",
    "is_unbounded_array",
    "is_union",
    "is_unsigned",
    "is_void",
    "is_volatile",
    "negation",
    "rank",
    "reference_constructs_from_temporary",
    "reference_converts_from_temporary",
};

static const llvm::StringSet<> TypeTraits = {
    "remove_cv",
    "remove_const",
    "remove_volatile",
    "add_cv",
    "add_const",
    "add_volatile",
    "remove_reference",
    "add_lvalue_reference",
    "add_rvalue_reference",
    "remove_pointer",
    "add_pointer",
    "make_signed",
    "make_unsigned",
    "remove_extent",
    "remove_all_extents",
    "aligned_storage",
    "aligned_union",
    "decay",
    "remove_cvref",
    "enable_if",
    "conditional",
    "common_type",
    "common_reference",
    "underlying_type",
    "result_of",
    "invoke_result",
    "type_identity",
};

static DeclarationName getName(const DependentScopeDeclRefExpr &D) {
  return D.getDeclName();
}

static DeclarationName getName(const DeclRefExpr &D) {
  return D.getDecl()->getDeclName();
}

static bool isNamedType(const ElaboratedTypeLoc &ETL) {
  if (const auto *TFT =
          ETL.getNamedTypeLoc().getTypePtr()->getAs<TypedefType>()) {
    const TypedefNameDecl *Decl = TFT->getDecl();
    return Decl->getDeclName().isIdentifier() && Decl->getName() == "type";
  }
  return false;
}

static bool isNamedType(const DependentNameTypeLoc &DTL) {
  return DTL.getTypePtr()->getIdentifier()->getName() == "type";
}

namespace {
AST_POLYMORPHIC_MATCHER(isValue, AST_POLYMORPHIC_SUPPORTED_TYPES(
                                     DeclRefExpr, DependentScopeDeclRefExpr)) {
  const IdentifierInfo *Ident = getName(Node).getAsIdentifierInfo();
  return Ident && Ident->isStr("value");
}

AST_POLYMORPHIC_MATCHER(isType,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(ElaboratedTypeLoc,
                                                        DependentNameTypeLoc)) {
  return Node.getBeginLoc().isValid() && isNamedType(Node);
}
} // namespace

static constexpr char Bind[] = "";

void TypeTraitsCheck::registerMatchers(MatchFinder *Finder) {
  const ast_matchers::internal::VariadicDynCastAllOfMatcher<
      Stmt,
      DependentScopeDeclRefExpr>
      dependentScopeDeclRefExpr; // NOLINT(readability-identifier-naming)
  const ast_matchers::internal::VariadicDynCastAllOfMatcher<
      TypeLoc,
      DependentNameTypeLoc>
      dependentNameTypeLoc; // NOLINT(readability-identifier-naming)

  // Only register matchers for trait<...>::value in c++17 mode.
  if (getLangOpts().CPlusPlus17) {
    Finder->addMatcher(mapAnyOf(declRefExpr, dependentScopeDeclRefExpr)
                           .with(isValue())
                           .bind(Bind),
                       this);
  }
  Finder->addMatcher(mapAnyOf(elaboratedTypeLoc, dependentNameTypeLoc)
                         .with(isType())
                         .bind(Bind),
                     this);
}

static bool isNamedDeclInStdTraitsSet(const NamedDecl *ND,
                                      const llvm::StringSet<> &Set) {
  return ND->isInStdNamespace() && ND->getDeclName().isIdentifier() &&
         Set.contains(ND->getName());
}

static bool checkTemplatedDecl(const NestedNameSpecifier *NNS,
                               const llvm::StringSet<> &Set) {
  if (!NNS)
    return false;
  const Type *NNST = NNS->getAsType();
  if (!NNST)
    return false;
  const auto *TST = NNST->getAs<TemplateSpecializationType>();
  if (!TST)
    return false;
  if (const TemplateDecl *TD = TST->getTemplateName().getAsTemplateDecl()) {
    return isNamedDeclInStdTraitsSet(TD, Set);
  }
  return false;
}

TypeTraitsCheck::TypeTraitsCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", false)) {}

void TypeTraitsCheck::check(const MatchFinder::MatchResult &Result) {
  auto EmitValueWarning = [this, &Result](const NestedNameSpecifierLoc &QualLoc,
                                          SourceLocation EndLoc) {
    SourceLocation TemplateNameEndLoc;
    if (auto TSTL = QualLoc.getTypeLoc().getAs<TemplateSpecializationTypeLoc>();
        !TSTL.isNull())
      TemplateNameEndLoc = Lexer::getLocForEndOfToken(
          TSTL.getTemplateNameLoc(), 0, *Result.SourceManager,
          Result.Context->getLangOpts());
    else
      return;

    if (EndLoc.isMacroID() || QualLoc.getEndLoc().isMacroID() ||
        TemplateNameEndLoc.isMacroID()) {
      if (IgnoreMacros)
        return;
      diag(QualLoc.getBeginLoc(), "use c++17 style variable templates");
      return;
    }
    diag(QualLoc.getBeginLoc(), "use c++17 style variable templates")
        << FixItHint::CreateInsertion(TemplateNameEndLoc, "_v")
        << FixItHint::CreateRemoval({QualLoc.getEndLoc(), EndLoc});
  };

  auto EmitTypeWarning = [this, &Result](const NestedNameSpecifierLoc &QualLoc,
                                         SourceLocation EndLoc,
                                         SourceLocation TypenameLoc) {
    SourceLocation TemplateNameEndLoc;
    if (auto TSTL = QualLoc.getTypeLoc().getAs<TemplateSpecializationTypeLoc>();
        !TSTL.isNull())
      TemplateNameEndLoc = Lexer::getLocForEndOfToken(
          TSTL.getTemplateNameLoc(), 0, *Result.SourceManager,
          Result.Context->getLangOpts());
    else
      return;

    if (EndLoc.isMacroID() || QualLoc.getEndLoc().isMacroID() ||
        TemplateNameEndLoc.isMacroID() || TypenameLoc.isMacroID()) {
      if (IgnoreMacros)
        return;
      diag(QualLoc.getBeginLoc(), "use c++14 style type templates");
      return;
    }
    auto Diag = diag(QualLoc.getBeginLoc(), "use c++14 style type templates");

    if (TypenameLoc.isValid())
      Diag << FixItHint::CreateRemoval(TypenameLoc);
    Diag << FixItHint::CreateInsertion(TemplateNameEndLoc, "_t")
         << FixItHint::CreateRemoval({QualLoc.getEndLoc(), EndLoc});
  };

  if (const auto *DRE = Result.Nodes.getNodeAs<DeclRefExpr>(Bind)) {
    if (!DRE->hasQualifier())
      return;
    if (const auto *CTSD = dyn_cast_if_present<ClassTemplateSpecializationDecl>(
            DRE->getQualifier()->getAsRecordDecl())) {
      if (isNamedDeclInStdTraitsSet(CTSD, ValueTraits))
        EmitValueWarning(DRE->getQualifierLoc(), DRE->getEndLoc());
    }
    return;
  }

  if (const auto *ETL = Result.Nodes.getNodeAs<ElaboratedTypeLoc>(Bind)) {
    const NestedNameSpecifierLoc QualLoc = ETL->getQualifierLoc();
    const auto *NNS = QualLoc.getNestedNameSpecifier();
    if (!NNS)
      return;
    if (const auto *CTSD = dyn_cast_if_present<ClassTemplateSpecializationDecl>(
            NNS->getAsRecordDecl())) {
      if (isNamedDeclInStdTraitsSet(CTSD, TypeTraits))
        EmitTypeWarning(ETL->getQualifierLoc(), ETL->getEndLoc(),
                        ETL->getElaboratedKeywordLoc());
    }
    return;
  }

  if (const auto *DSDRE =
          Result.Nodes.getNodeAs<DependentScopeDeclRefExpr>(Bind)) {
    if (checkTemplatedDecl(DSDRE->getQualifier(), ValueTraits))
      EmitValueWarning(DSDRE->getQualifierLoc(), DSDRE->getEndLoc());
    return;
  }

  if (const auto *DNTL = Result.Nodes.getNodeAs<DependentNameTypeLoc>(Bind)) {
    NestedNameSpecifierLoc QualLoc = DNTL->getQualifierLoc();
    if (checkTemplatedDecl(QualLoc.getNestedNameSpecifier(), TypeTraits))
      EmitTypeWarning(QualLoc, DNTL->getEndLoc(),
                      DNTL->getElaboratedKeywordLoc());
    return;
  }
}

void TypeTraitsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}
} // namespace clang::tidy::modernize
