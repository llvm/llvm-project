//===----------------------------------------------------------------------===//
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
  assert(BT);

  // BT->isInteger() would detect char and bool
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
  assert(BT);

  return BT->isFloatingPoint();
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
                      areDiagsSelfContained()) {
  if (!WarnOnFloats && !WarnOnInts && !WarnOnChars)
    this->configurationDiag(
        "The check 'portability-avoid-platform-specific-fundamental-types' "
        "will not perform any analysis because 'WarnOnFloats', 'WarnOnInts' "
        "and 'WarnOnChars' are all false.");
}

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

static std::optional<std::string> getFloatReplacement(const BuiltinType *BT,
                                                      ASTContext &Context) {
  const TargetInfo &Target = Context.getTargetInfo();

  auto GetReplacementType = [](unsigned Width) -> std::optional<std::string> {
    switch (Width) {
    // This is ambiguous by default since it could be bfloat16 or float16
    case 16U:
      return std::nullopt;
    case 32U:
      return "float32_t";
    case 64U:
      return "float64_t";
    case 128U:
      return "float128_t";
    default:
      return std::nullopt;
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
  case BuiltinType::LongDouble:
    return GetReplacementType(Target.getLongDoubleWidth());
  case BuiltinType::Float128:
    return "float128_t";
  default:
    return std::nullopt;
  }
}

void AvoidPlatformSpecificFundamentalTypesCheck::registerMatchers(
    MatchFinder *Finder) {
  auto PlatformSpecificFundamentalType = qualType(allOf(
      builtinType(), anyOf(WarnOnInts ? isBuiltinInt() : unless(anything()),
                           WarnOnFloats ? isBuiltinFloat() : unless(anything()),
                           WarnOnChars ? isChar() : unless(anything()),
                           WarnOnChars ? isWideChar() : unless(anything()))));

  if (!WarnOnInts && !WarnOnFloats && !WarnOnChars)
    return;

  Finder->addMatcher(typeLoc(loc(PlatformSpecificFundamentalType)).bind("type"),
                     this);
}

void AvoidPlatformSpecificFundamentalTypesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *TL = Result.Nodes.getNodeAs<TypeLoc>("type");
  assert(TL);

  const SourceLocation Loc = TL->getBeginLoc();
  const QualType QT = TL->getType();
  const SourceRange TypeRange = TL->getSourceRange();

  // Skip implicit type locations, such as literals
  if (!Loc.isValid() || !TypeRange.isValid())
    return;

  const std::string TypeName = QT.getUnqualifiedType().getAsString();

  const auto *BT = QT->getAs<BuiltinType>();

  assert(BT);
  if (BT->isFloatingPoint()) {
    const auto Replacement = getFloatReplacement(BT, *Result.Context);

    if (!Replacement.has_value()) {
      diag(Loc, "avoid using platform-dependent floating point type '%0'; "
                "consider using a type alias or fixed-width type instead")
          << TypeName;
      return;
    }

    auto Diag =
        diag(Loc, "avoid using platform-dependent floating point type '%0'; "
                  "consider using '%1' instead")
        << TypeName << Replacement.value();

    if (TypeRange.isValid())
      Diag << FixItHint::CreateReplacement(TypeRange, Replacement.value());

    if (auto IncludeFixit = IncludeInserter.createIncludeInsertion(
            Result.SourceManager->getFileID(Loc), "<stdfloat>")) {
      Diag << *IncludeFixit;
    }
  } else if (QT->isCharType() || QT->isWideCharType()) {
    diag(Loc, "avoid using platform-dependent character type '%0'; "
              "consider using 'char8_t' for text or 'std::byte' for bytes")
        << TypeName;
  } else {
    diag(Loc, "avoid using platform-dependent fundamental integer type '%0'; "
              "consider using a type alias or fixed-width type instead")
        << TypeName;
  }
}

} // namespace clang::tidy::portability
