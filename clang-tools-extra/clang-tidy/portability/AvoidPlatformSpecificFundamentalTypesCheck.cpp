//===--- AvoidPlatformSpecificFundamentalTypesCheck.cpp - clang-tidy
//---------------===//
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

AvoidPlatformSpecificFundamentalTypesCheck::
    AvoidPlatformSpecificFundamentalTypesCheck(StringRef Name,
                                               ClangTidyContext *Context)
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

bool AvoidPlatformSpecificFundamentalTypesCheck::isSemanticType(
    const Type *T) const {
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

void AvoidPlatformSpecificFundamentalTypesCheck::registerMatchers(
    MatchFinder *Finder) {
  // Create a matcher for platform-specific fundamental integer types
  // This should only match direct uses of builtin types, not typedefs
  auto PlatformSpecificFundamentalType = qualType(
      allOf(
          // Must be a builtin type directly (not through typedef)
          builtinType(),
          // Only match the specific fundamental integer types we care about
          anyOf(
              asString("int"),
              asString("unsigned int"),
              asString("short"),
              asString("unsigned short"),
              asString("long"),
              asString("unsigned long"),
              asString("long long"),
              asString("unsigned long long")
          )
      )
  );

  // Match variable declarations with platform-specific fundamental integer types
  Finder->addMatcher(
      varDecl(hasType(PlatformSpecificFundamentalType)).bind("var_decl"),
      this);

  // Match function declarations with platform-specific fundamental integer return types
  Finder->addMatcher(
      functionDecl(returns(PlatformSpecificFundamentalType)).bind("func_decl"),
      this);

  // Match function parameters with platform-specific fundamental integer types
  Finder->addMatcher(
      parmVarDecl(hasType(PlatformSpecificFundamentalType)).bind("param_decl"),
      this);

  // Match field declarations with platform-specific fundamental integer types
  Finder->addMatcher(
      fieldDecl(hasType(PlatformSpecificFundamentalType)).bind("field_decl"),
      this);

  // Match typedef declarations with platform-specific fundamental underlying types
  Finder->addMatcher(
      typedefDecl(hasUnderlyingType(PlatformSpecificFundamentalType)).bind("typedef_decl"),
      this);

  // Match type alias declarations with platform-specific fundamental underlying types
  Finder->addMatcher(
      typeAliasDecl(hasType(PlatformSpecificFundamentalType)).bind("alias_decl"),
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
  } else if (const auto *FD =
                 Result.Nodes.getNodeAs<FunctionDecl>("func_decl")) {
    Loc = FD->getLocation();
    QT = FD->getReturnType();
    DeclType = "function return type";
  } else if (const auto *PD =
                 Result.Nodes.getNodeAs<ParmVarDecl>("param_decl")) {
    Loc = PD->getLocation();
    QT = PD->getType();
    DeclType = "function parameter";
  } else if (const auto *FD = Result.Nodes.getNodeAs<FieldDecl>("field_decl")) {
    Loc = FD->getLocation();
    QT = FD->getType();
    DeclType = "field";
  } else if (const auto *TD =
                 Result.Nodes.getNodeAs<TypedefDecl>("typedef_decl")) {
    Loc = TD->getLocation();
    QT = TD->getUnderlyingType();
    DeclType = "typedef";
  } else if (const auto *AD =
                 Result.Nodes.getNodeAs<TypeAliasDecl>("alias_decl")) {
    Loc = AD->getLocation();
    QT = AD->getUnderlyingType();
    DeclType = "type alias";
  } else {
    return;
  }

  if (Loc.isInvalid() || QT.isNull())
    return;

  // Get the type name for the diagnostic
  std::string TypeName = QT.getAsString();

  diag(Loc, "avoid using platform-dependent fundamental integer type '%0'; "
            "consider using a typedef or fixed-width type instead")
      << TypeName;
}

} // namespace clang::tidy::portability
