//===--- AvoidPlatformSpecificFundamentalTypesCheck.cpp - clang-tidy ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidPlatformSpecificFundamentalTypesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang::ast_matchers;

namespace {
AST_MATCHER(clang::QualType, isBuiltinInt) {
  const auto *BT = Node->getAs<clang::BuiltinType>();
  if (!BT)
    return false;

  switch (BT->getKind()) {
  case clang::BuiltinType::Short:
  case clang::BuiltinType::UShort:
  case clang::BuiltinType::Int:
  case clang::BuiltinType::UInt:
  case clang::BuiltinType::Long:
  case clang::BuiltinType::ULong:
  case clang::BuiltinType::LongLong:
  case clang::BuiltinType::ULongLong:
    return true;
  default:
    return false;
  }
}

AST_MATCHER(clang::QualType, isBuiltinFloat) {
  const auto *BT = Node->getAs<clang::BuiltinType>();
  if (!BT)
    return false;

  switch (BT->getKind()) {
  case clang::BuiltinType::Half:
  case clang::BuiltinType::BFloat16:
  case clang::BuiltinType::Float:
  case clang::BuiltinType::Double:
  case clang::BuiltinType::LongDouble:
    return true;
  default:
    return false;
  }
}

AST_MATCHER(clang::QualType, isBuiltinChar) {
  const auto *BT = Node->getAs<clang::BuiltinType>();
  if (!BT)
    return false;

  switch (BT->getKind()) {
  case clang::BuiltinType::Char_S:
  case clang::BuiltinType::Char_U:
  case clang::BuiltinType::SChar:
  case clang::BuiltinType::UChar:
    return true;
  default:
    return false;
  }
}
} // namespace

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
  auto PlatformSpecificFundamentalType = qualType(
      allOf(builtinType(),
            anyOf(WarnOnInts ? isBuiltinInt() : unless(anything()),
                  WarnOnFloats ? isBuiltinFloat() : unless(anything()),
                  WarnOnChars ? isBuiltinChar() : unless(anything()))));

  if (!WarnOnInts && !WarnOnFloats && !WarnOnChars)
    return;

  Finder->addMatcher(
      varDecl(hasType(PlatformSpecificFundamentalType)).bind("var_decl"), this);

  Finder->addMatcher(
      functionDecl(returns(PlatformSpecificFundamentalType)).bind("func_decl"),
      this);

  Finder->addMatcher(
      parmVarDecl(hasType(PlatformSpecificFundamentalType)).bind("param_decl"),
      this);

  Finder->addMatcher(
      fieldDecl(hasType(PlatformSpecificFundamentalType)).bind("field_decl"),
      this);

  Finder->addMatcher(
      typedefDecl(hasUnderlyingType(PlatformSpecificFundamentalType))
          .bind("typedef_decl"),
      this);

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

  const std::string TypeName = QT.getAsString();

  const auto *BT = QT->getAs<BuiltinType>();

  if (BT->isFloatingPoint()) {
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
    diag(Loc, "avoid using platform-dependent character type '%0'; "
              "consider using char8_t for text or std::byte for bytes")
        << TypeName;
  } else {
    diag(Loc, "avoid using platform-dependent fundamental integer type '%0'; "
              "consider using a typedef or fixed-width type instead")
        << TypeName;
  }
}

} // namespace clang::tidy::portability
