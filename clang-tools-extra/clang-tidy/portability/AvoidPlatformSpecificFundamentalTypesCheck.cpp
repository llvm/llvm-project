//===--- AvoidPlatformSpecificFundamentalTypesCheck.cpp - clang-tidy ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidPlatformSpecificFundamentalTypesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::portability {

namespace {

AST_MATCHER(clang::TypeLoc, hasValidBeginLoc) {
  return Node.getBeginLoc().isValid();
}

AST_MATCHER_P(clang::TypeLoc, hasType,
              clang::ast_matchers::internal::Matcher<clang::Type>,
              InnerMatcher) {
  const clang::Type *TypeNode = Node.getTypePtr();
  return TypeNode != nullptr &&
         InnerMatcher.matches(*TypeNode, Finder, Builder);
}

} // namespace

AvoidPlatformSpecificFundamentalTypesCheck::AvoidPlatformSpecificFundamentalTypesCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

bool AvoidPlatformSpecificFundamentalTypesCheck::isFundamentalIntegerType(
    const Type *T) const {
  if (!T->isBuiltinType())
    return false;

  const auto *BT = T->getAs<BuiltinType>();
  if (!BT)
    return false;

  switch (BT->getKind()) {
  case BuiltinType::Int:
  case BuiltinType::UInt:
  case BuiltinType::Short:
  case BuiltinType::UShort:
  case BuiltinType::Long:
  case BuiltinType::ULong:
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
    return true;
  default:
    return false;
  }
}

bool AvoidPlatformSpecificFundamentalTypesCheck::isSemanticType(const Type *T) const {
  if (!T->isBuiltinType())
    return false;

  const auto *BT = T->getAs<BuiltinType>();
  if (!BT)
    return false;

  switch (BT->getKind()) {
  case BuiltinType::Bool:
  case BuiltinType::Char_S:
  case BuiltinType::Char_U:
  case BuiltinType::SChar:
  case BuiltinType::UChar:
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
  case BuiltinType::Char8:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
    return true;
  default:
    return false;
  }
}

void AvoidPlatformSpecificFundamentalTypesCheck::registerMatchers(MatchFinder *Finder) {
  // Match variable declarations with fundamental integer types
  Finder->addMatcher(
      varDecl().bind("var_decl"),
      this);

  // Match function declarations with fundamental integer return types
  Finder->addMatcher(
      functionDecl().bind("func_decl"),
      this);

  // Match function parameters with fundamental integer types
  Finder->addMatcher(
      parmVarDecl().bind("param_decl"),
      this);

  // Match field declarations with fundamental integer types
  Finder->addMatcher(
      fieldDecl().bind("field_decl"),
      this);

  // Match typedef declarations to check their underlying types
  Finder->addMatcher(
      typedefDecl().bind("typedef_decl"),
      this);

  Finder->addMatcher(
      typeAliasDecl().bind("alias_decl"),
      this);
}

void AvoidPlatformSpecificFundamentalTypesCheck::check(
    const MatchFinder::MatchResult &Result) {
  SourceLocation Loc;
  QualType QT;
  std::string DeclType;

  if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var_decl")) {
    Loc = VD->getLocation();
    QT = VD->getType();
    DeclType = "variable";
  } else if (const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("func_decl")) {
    Loc = FD->getLocation();
    QT = FD->getReturnType();
    DeclType = "function return type";
  } else if (const auto *PD = Result.Nodes.getNodeAs<ParmVarDecl>("param_decl")) {
    Loc = PD->getLocation();
    QT = PD->getType();
    DeclType = "function parameter";
  } else if (const auto *FD = Result.Nodes.getNodeAs<FieldDecl>("field_decl")) {
    Loc = FD->getLocation();
    QT = FD->getType();
    DeclType = "field";
  } else if (const auto *TD = Result.Nodes.getNodeAs<TypedefDecl>("typedef_decl")) {
    Loc = TD->getLocation();
    QT = TD->getUnderlyingType();
    DeclType = "typedef";
  } else if (const auto *AD = Result.Nodes.getNodeAs<TypeAliasDecl>("alias_decl")) {
    Loc = AD->getLocation();
    QT = AD->getUnderlyingType();
    DeclType = "type alias";
  } else {
    return;
  }

  if (Loc.isInvalid() || QT.isNull())
    return;

  // Check if the type is already a typedef - if so, don't warn
  // since the user is already using a typedef (which is what we want)
  if (QT->getAs<TypedefType>()) {
    return;
  }

  const Type *T = QT.getCanonicalType().getTypePtr();
  if (!T)
    return;

  // Skip if not a fundamental integer type
  if (!isFundamentalIntegerType(T))
    return;

  // Skip semantic types
  if (isSemanticType(T))
    return;

  // Get the type name for the diagnostic
  std::string TypeName = QT.getAsString();

  diag(Loc, "avoid using platform-dependent fundamental integer type '%0'; "
            "consider using a typedef or fixed-width type instead")
      << TypeName;
}

} // namespace clang::tidy::portability
