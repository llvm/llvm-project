//===--- AvoidPlatformSpecificFundamentalTypesCheck.cpp - clang-tidy ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidPlatformSpecificFundamentalTypesCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang::ast_matchers;

namespace clang::tidy::portability {

AvoidPlatformSpecificFundamentalTypesCheck::
    AvoidPlatformSpecificFundamentalTypesCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnFloats(Options.get("WarnOnFloats", true)),
      WarnOnInts(Options.get("WarnOnInts", true)),
      WarnOnChars(Options.get("WarnOnChars", true)),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

void AvoidPlatformSpecificFundamentalTypesCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void AvoidPlatformSpecificFundamentalTypesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnFloats", WarnOnFloats);
  Options.store(Opts, "WarnOnInts", WarnOnInts);
  Options.store(Opts, "WarnOnChars", WarnOnChars);
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
}

std::string AvoidPlatformSpecificFundamentalTypesCheck::getFloatReplacement(
    const BuiltinType *BT, ASTContext &Context) const {
  const TargetInfo &Target = Context.getTargetInfo();

  auto GetReplacementType = [](unsigned Width) {
    switch (Width) {
    // This is ambiguous by default since it could be bfloat16 or float16
    case 16U:
      return "";
    case 32U:
      return "float32_t";
    case 64U:
      return "float64_t";
    case 128U:
      return "float128_t";
    default:
      return "";
    }
  };

  switch (BT->getKind()) {
  // Not an ambiguous type
  case BuiltinType::BFloat16:
    return "bfloat16_t";
  case BuiltinType::Half:
    return GetReplacementType(Target.getHalfWidth());
  case BuiltinType::Float:
    return GetReplacementType(Target.getFloatWidth());
  case BuiltinType::Double:
    return GetReplacementType(Target.getDoubleWidth());
  default:
    return "";
  }
}

void AvoidPlatformSpecificFundamentalTypesCheck::registerMatchers(
    MatchFinder *Finder) {
  // Build the list of type strings to match
  std::vector<std::string> TypeStrings;

  // Add integer types if the option is enabled
  if (WarnOnInts) {
    TypeStrings.insert(TypeStrings.end(), {"short",
                                           "short int",
                                           "signed short",
                                           "signed short int",
                                           "unsigned short",
                                           "unsigned short int",
                                           "int",
                                           "signed",
                                           "signed int",
                                           "unsigned",
                                           "unsigned int",
                                           "long",
                                           "long int",
                                           "signed long",
                                           "signed long int",
                                           "unsigned long",
                                           "unsigned long int",
                                           "long long",
                                           "long long int",
                                           "signed long long",
                                           "signed long long int",
                                           "unsigned long long",
                                           "unsigned long long int"});
  }

  // Add float types if the option is enabled
  if (WarnOnFloats) {
    TypeStrings.insert(TypeStrings.end(),
                       {"half", "__bf16", "float", "double", "long double"});
  }

  // Add char types if the option is enabled
  if (WarnOnChars) {
    TypeStrings.insert(TypeStrings.end(),
                       {"char", "signed char", "unsigned char"});
  }

  // If no types are enabled, return early
  if (TypeStrings.empty())
    return;

  // Create the matcher dynamically
  auto TypeMatcher = asString(TypeStrings.front());
  for (const auto &TypeString : TypeStrings)
    TypeMatcher = anyOf(TypeMatcher, asString(TypeString));

  auto PlatformSpecificFundamentalType = qualType(allOf(
      // Must be a builtin type directly (not through typedef)
      builtinType(),
      // Match the specific fundamental types we care about
      TypeMatcher));

  // Match variable declarations with platform-specific fundamental types
  Finder->addMatcher(
      varDecl(hasType(PlatformSpecificFundamentalType)).bind("var_decl"), this);

  // Match function declarations with platform-specific fundamental return types
  Finder->addMatcher(
      functionDecl(returns(PlatformSpecificFundamentalType)).bind("func_decl"),
      this);

  // Match function parameters with platform-specific fundamental types
  Finder->addMatcher(
      parmVarDecl(hasType(PlatformSpecificFundamentalType)).bind("param_decl"),
      this);

  // Match field declarations with platform-specific fundamental types
  Finder->addMatcher(
      fieldDecl(hasType(PlatformSpecificFundamentalType)).bind("field_decl"),
      this);

  // Match typedef declarations with platform-specific fundamental underlying
  // types
  Finder->addMatcher(
      typedefDecl(hasUnderlyingType(PlatformSpecificFundamentalType))
          .bind("typedef_decl"),
      this);

  // Match type alias declarations with platform-specific fundamental underlying
  // types
  Finder->addMatcher(typeAliasDecl(hasType(PlatformSpecificFundamentalType))
                         .bind("alias_decl"),
                     this);
}

void AvoidPlatformSpecificFundamentalTypesCheck::check(
    const MatchFinder::MatchResult &Result) {
  SourceLocation Loc;
  QualType QT;
  SourceRange TypeRange;

  auto SetTypeRange = [&TypeRange](auto Decl) {
    if (Decl->getTypeSourceInfo())
      TypeRange = Decl->getTypeSourceInfo()->getTypeLoc().getSourceRange();
  };

  if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var_decl")) {
    Loc = VD->getLocation();
    QT = VD->getType();
    SetTypeRange(VD);
  } else if (const auto *FD =
                 Result.Nodes.getNodeAs<FunctionDecl>("func_decl")) {
    Loc = FD->getLocation();
    QT = FD->getReturnType();
    SetTypeRange(FD);
  } else if (const auto *PD =
                 Result.Nodes.getNodeAs<ParmVarDecl>("param_decl")) {
    Loc = PD->getLocation();
    QT = PD->getType();
    SetTypeRange(PD);
  } else if (const auto *FD = Result.Nodes.getNodeAs<FieldDecl>("field_decl")) {
    Loc = FD->getLocation();
    QT = FD->getType();
    SetTypeRange(FD);
  } else if (const auto *TD =
                 Result.Nodes.getNodeAs<TypedefDecl>("typedef_decl")) {
    Loc = TD->getLocation();
    QT = TD->getUnderlyingType();
    SetTypeRange(TD);
  } else if (const auto *AD =
                 Result.Nodes.getNodeAs<TypeAliasDecl>("alias_decl")) {
    Loc = AD->getLocation();
    QT = AD->getUnderlyingType();
    SetTypeRange(AD);
  } else {
    return;
  }

  // Get the type name for the diagnostic
  const std::string TypeName = QT.getAsString();

  // Check the type category
  const auto *BT = QT->getAs<BuiltinType>();

  if (BT->isFloatingPoint()) {
    // Handle floating point types
    const std::string Replacement = getFloatReplacement(BT, *Result.Context);
    if (!Replacement.empty()) {
      auto Diag =
          diag(Loc, "avoid using platform-dependent floating point type '%0'; "
                    "consider using '%1' instead")
          << TypeName << Replacement;

      if (TypeRange.isValid())
        Diag << FixItHint::CreateReplacement(TypeRange, Replacement);

      if (auto IncludeFixit = IncludeInserter.createIncludeInsertion(
              Result.SourceManager->getFileID(Loc), "<stdfloat>")) {
        Diag << *IncludeFixit;
      }
    } else {
      diag(Loc, "avoid using platform-dependent floating point type '%0'; "
                "consider using a typedef or fixed-width type instead")
          << TypeName;
    }
  } else if (BT->getKind() == BuiltinType::Char_S ||
             BT->getKind() == BuiltinType::Char_U ||
             BT->getKind() == BuiltinType::SChar ||
             BT->getKind() == BuiltinType::UChar) {
    // Handle char types
    diag(Loc, "avoid using platform-dependent character type '%0'; "
              "consider using char8_t for text or std::byte for bytes")
        << TypeName;
  } else {
    // Handle integer types
    diag(Loc, "avoid using platform-dependent fundamental integer type '%0'; "
              "consider using a typedef or fixed-width type instead")
        << TypeName;
  }
}

} // namespace clang::tidy::portability
