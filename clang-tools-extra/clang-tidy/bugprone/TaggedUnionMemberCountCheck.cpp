//===--- TaggedUnionMemberCountCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TaggedUnionMemberCountCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

const char StrictModeOptionName[] = "StrictMode";
const char EnableCountingEnumHeuristicOptionName[] =
    "EnableCountingEnumHeuristic";
const char CountingEnumPrefixesOptionName[] = "CountingEnumPrefixes";
const char CountingEnumSuffixesOptionName[] = "CountingEnumSuffixes";

TaggedUnionMemberCountCheck::TaggedUnionMemberCountCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.get(StrictModeOptionName, false)),
      EnableCountingEnumHeuristic(
          Options.get(EnableCountingEnumHeuristicOptionName, true)),
      RawCountingEnumPrefixes(Options.get(CountingEnumPrefixesOptionName, "")),
      RawCountingEnumSuffixes(
          Options.get(CountingEnumSuffixesOptionName, "count")),
      ParsedCountingEnumPrefixes(
          utils::options::parseStringList(RawCountingEnumPrefixes)),
      ParsedCountingEnumSuffixes(
          utils::options::parseStringList(RawCountingEnumSuffixes)),
      CountingEnumPrefixesSet(
          Options.get(CountingEnumPrefixesOptionName).has_value()),
      CountingEnumSuffixesSet(
          Options.get(CountingEnumSuffixesOptionName).has_value()),
      CountingEnumConstantDecl(nullptr) {
  if (!EnableCountingEnumHeuristic) {
    if (CountingEnumPrefixesSet)
      configurationDiag("%0: Counting enum heuristic is disabled but "
                        "CountingEnumPrefixes is set")
          << Name;
    if (CountingEnumSuffixesSet)
      configurationDiag("%0: Counting enum heuristic is disabled but "
                        "CountingEnumSuffixes is set")
          << Name;
  }
}

void TaggedUnionMemberCountCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, StrictModeOptionName, StrictMode);
  Options.store(Opts, EnableCountingEnumHeuristicOptionName,
                EnableCountingEnumHeuristic);
  Options.store(Opts, CountingEnumPrefixesOptionName, RawCountingEnumPrefixes);
  Options.store(Opts, CountingEnumSuffixesOptionName, RawCountingEnumSuffixes);
}

void TaggedUnionMemberCountCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      recordDecl(
          anyOf(isStruct(), isClass()),
          has(fieldDecl(hasType(qualType(hasCanonicalType(recordType()))))
                  .bind("union")),
          has(fieldDecl(hasType(qualType(hasCanonicalType(enumType()))))
                  .bind("tags")))
          .bind("root"),
      this);
}

static bool isUnion(const FieldDecl *R) {
  return R->getType().getCanonicalType().getTypePtr()->isUnionType();
}

static bool isEnum(const FieldDecl *R) {
  return R->getType().getCanonicalType().getTypePtr()->isEnumeralType();
}

static bool hasMultipleUnionsOrEnums(const RecordDecl *Rec) {
  return llvm::count_if(Rec->fields(), isUnion) > 1 ||
         llvm::count_if(Rec->fields(), isEnum) > 1;
}

static bool signEquals(const llvm::APSInt &A, const llvm::APSInt &B) {
  return (A.isNegative() && B.isNegative()) ||
         (A.isStrictlyPositive() && B.isStrictlyPositive()) ||
         (A.isZero() && B.isZero());
}

static bool greaterBySign(const llvm::APSInt &A, const llvm::APSInt &B) {
  return (A.isNonNegative() && B.isNegative()) ||
         (A.isStrictlyPositive() && B.isNonPositive());
}

bool TaggedUnionMemberCountCheck::isCountingEnumLikeName(
    StringRef Name) const noexcept {
  if (llvm::any_of(ParsedCountingEnumPrefixes,
                   [&Name](const StringRef &Prefix) -> bool {
                     return Name.starts_with_insensitive(Prefix);
                   }))
    return true;
  if (llvm::any_of(ParsedCountingEnumSuffixes,
                   [&Name](const StringRef &Suffix) -> bool {
                     return Name.ends_with_insensitive(Suffix);
                   }))
    return true;
  return false;
}

size_t TaggedUnionMemberCountCheck::getNumberOfValidEnumValues(
    const EnumDecl *Ed) noexcept {
  bool FoundMax = false;
  llvm::APSInt MaxTagValue;
  llvm::SmallSet<llvm::APSInt, 32> EnumValues;

  // Heuristic for counter enum constants.
  //
  //   enum tag_with_counter {
  //     tag1,
  //     tag2,
  //     tag_count, <-- Searching for these enum constants
  //   };
  //
  // The final tag count is decreased by 1 if and only if:
  // 1. There is only one counting enum constant,
  // 2. The counting enum constant is the last enum constant that is defined,
  // 3. The value of the counting enum constant is the largest out of every enum
  //    constant.
  // The 'ce' prefix is a shorthand for 'counting enum'.
  size_t CeCount = 0;
  bool CeIsLast = false;
  llvm::APSInt CeValue = llvm::APSInt::get(0);

  for (const auto &Enumerator : Ed->enumerators()) {
    const llvm::APSInt Val = Enumerator->getInitVal();
    EnumValues.insert(Val);
    if (FoundMax) {
      if (greaterBySign(Val, MaxTagValue) ||
          (signEquals(Val, MaxTagValue) && Val > MaxTagValue))
        MaxTagValue = Val;
    } else {
      MaxTagValue = Val;
      FoundMax = true;
    }

    if (EnableCountingEnumHeuristic) {
      if (isCountingEnumLikeName(Enumerator->getName())) {
        CeIsLast = true;
        CeValue = Val;
        CeCount += 1;
        CountingEnumConstantDecl = Enumerator;
      } else {
        CeIsLast = false;
      }
    }
  }

  size_t ValidValuesCount = EnumValues.size();
  if (CeCount == 1 && CeIsLast && CeValue == MaxTagValue) {
    ValidValuesCount -= 1;
  } else {
    CountingEnumConstantDecl = nullptr;
  }

  return ValidValuesCount;
}

void TaggedUnionMemberCountCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Root = Result.Nodes.getNodeAs<RecordDecl>("root");
  const auto *UnionField = Result.Nodes.getNodeAs<FieldDecl>("union");
  const auto *TagField = Result.Nodes.getNodeAs<FieldDecl>("tags");

  // The matcher can only narrow down the type to recordType()
  if (!isUnion(UnionField))
    return;

  if (hasMultipleUnionsOrEnums(Root))
    return;

  const auto *UnionDef =
      UnionField->getType().getCanonicalType().getTypePtr()->getAsRecordDecl();
  const auto *EnumDef = llvm::dyn_cast_or_null<EnumDecl>(
      TagField->getType().getCanonicalType().getTypePtr()->getAsTagDecl());

  if (EnumDef) {
    const size_t UnionMemberCount = llvm::range_size(UnionDef->fields());
    const size_t TagCount = getNumberOfValidEnumValues(EnumDef);

    if (UnionMemberCount > TagCount) {
      diag(Root->getLocation(),
           "tagged union has more data members (%0) than tags (%1)!")
          << UnionMemberCount << TagCount;
    } else if (StrictMode && UnionMemberCount < TagCount) {
      diag(Root->getLocation(),
           "tagged union has fewer data members (%0) than tags (%1)!")
          << UnionMemberCount << TagCount;
    }

    if (CountingEnumConstantDecl) {
      diag(CountingEnumConstantDecl->getLocation(),
           "assuming that this constant is just an auxiliary value and not "
           "used for indicating a valid union data member",
           DiagnosticIDs::Note);
    }
  }
}

} // namespace clang::tidy::bugprone
