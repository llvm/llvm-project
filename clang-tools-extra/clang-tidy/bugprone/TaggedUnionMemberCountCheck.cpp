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

static constexpr llvm::StringLiteral StrictModeOptionName = "StrictMode";
static constexpr llvm::StringLiteral EnableCountingEnumHeuristicOptionName =
    "EnableCountingEnumHeuristic";
static constexpr llvm::StringLiteral CountingEnumPrefixesOptionName =
    "CountingEnumPrefixes";
static constexpr llvm::StringLiteral CountingEnumSuffixesOptionName =
    "CountingEnumSuffixes";

static constexpr bool StrictModeOptionDefaultValue = false;
static constexpr bool EnableCountingEnumHeuristicOptionDefaultValue = true;
static constexpr llvm::StringLiteral CountingEnumPrefixesOptionDefaultValue =
    "";
static constexpr llvm::StringLiteral CountingEnumSuffixesOptionDefaultValue =
    "count";

static constexpr llvm::StringLiteral RootMatchBindName = "root";
static constexpr llvm::StringLiteral UnionMatchBindName = "union";
static constexpr llvm::StringLiteral TagMatchBindName = "tags";

namespace {
AST_MATCHER(FieldDecl, isUnion) {
  const Type *T = Node.getType().getCanonicalType().getTypePtr();
  assert(T);
  return T->isUnionType();
}

AST_MATCHER(FieldDecl, isEnum) {
  const Type *T = Node.getType().getCanonicalType().getTypePtr();
  assert(T);
  return T->isEnumeralType();
}

AST_MATCHER_P2(RecordDecl, fieldCountOfKindIsGT,
               ast_matchers::internal::Matcher<FieldDecl>, InnerMatcher,
               unsigned, N) {
  unsigned matchCount = 0;
  for (const auto field : Node.fields()) {
    if (InnerMatcher.matches(*field, Finder, Builder)) {
      matchCount += 1;
    }
  }
  return matchCount > N;
}
}; // namespace

TaggedUnionMemberCountCheck::TaggedUnionMemberCountCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(
          Options.get(StrictModeOptionName, StrictModeOptionDefaultValue)),
      EnableCountingEnumHeuristic(
          Options.get(EnableCountingEnumHeuristicOptionName,
                      EnableCountingEnumHeuristicOptionDefaultValue)),
      CountingEnumPrefixes(utils::options::parseStringList(
          Options.get(CountingEnumPrefixesOptionName,
                      CountingEnumPrefixesOptionDefaultValue))),
      CountingEnumSuffixes(utils::options::parseStringList(
          Options.get(CountingEnumSuffixesOptionName,
                      CountingEnumSuffixesOptionDefaultValue))) {
  if (!EnableCountingEnumHeuristic) {
    if (Options.get(CountingEnumPrefixesOptionName).has_value())
      configurationDiag("%0: Counting enum heuristic is disabled but "
                        "%1 is set")
          << Name << CountingEnumPrefixesOptionName;
    if (Options.get(CountingEnumSuffixesOptionName).has_value())
      configurationDiag("%0: Counting enum heuristic is disabled but "
                        "%1 is set")
          << Name << CountingEnumSuffixesOptionName;
  }
}

void TaggedUnionMemberCountCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, StrictModeOptionName, StrictMode);
  Options.store(Opts, EnableCountingEnumHeuristicOptionName,
                EnableCountingEnumHeuristic);
  Options.store(Opts, CountingEnumPrefixesOptionName,
                utils::options::serializeStringList(CountingEnumPrefixes));
  Options.store(Opts, CountingEnumSuffixesOptionName,
                utils::options::serializeStringList(CountingEnumSuffixes));
}

void TaggedUnionMemberCountCheck::registerMatchers(MatchFinder *Finder) {

  static const auto hasMultipleUnionsOrEnums = anyOf(
      fieldCountOfKindIsGT(isUnion(), 1), fieldCountOfKindIsGT(isEnum(), 1));

  Finder->addMatcher(
      recordDecl(anyOf(isStruct(), isClass()),
                 unless(anyOf(isImplicit(), hasMultipleUnionsOrEnums)),
                 has(fieldDecl(isUnion()).bind(UnionMatchBindName)),
                 has(fieldDecl(isEnum()).bind(TagMatchBindName)))
          .bind(RootMatchBindName),
      this);
}

static bool signEquals(const llvm::APSInt &A, const llvm::APSInt &B) {
  return (A.isNegative() == B.isNegative()) &&
         (A.isStrictlyPositive() == B.isStrictlyPositive()) &&
         (A.isZero() == B.isZero());
}

static bool greaterBySign(const llvm::APSInt &A, const llvm::APSInt &B) {
  return (A.isNonNegative() && B.isNegative()) ||
         (A.isStrictlyPositive() && B.isNonPositive());
}

bool TaggedUnionMemberCountCheck::isCountingEnumLikeName(
    StringRef Name) const {
  if (llvm::any_of(CountingEnumPrefixes, [Name](StringRef Prefix) -> bool {
        return Name.starts_with_insensitive(Prefix);
      }))
    return true;
  if (llvm::any_of(CountingEnumSuffixes, [Name](StringRef Suffix) -> bool {
        return Name.ends_with_insensitive(Suffix);
      }))
    return true;
  return false;
}

std::size_t TaggedUnionMemberCountCheck::getNumberOfValidEnumValues(
    const EnumDecl *ED,
    const EnumConstantDecl *&OutCountingEnumConstantDecl) {
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
  std::size_t CeCount = 0;
  bool CeIsLast = false;
  llvm::APSInt CeValue = llvm::APSInt::get(0);

  for (const auto &Enumerator : ED->enumerators()) {
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
        OutCountingEnumConstantDecl = Enumerator;
      } else {
        CeIsLast = false;
      }
    }
  }

  std::size_t ValidValuesCount = EnumValues.size();
  if (CeCount == 1 && CeIsLast && CeValue == MaxTagValue) {
    ValidValuesCount -= 1;
  } else {
    OutCountingEnumConstantDecl = nullptr;
  }

  return ValidValuesCount;
}

void TaggedUnionMemberCountCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Root = Result.Nodes.getNodeAs<RecordDecl>(RootMatchBindName);
  const auto *UnionField =
      Result.Nodes.getNodeAs<FieldDecl>(UnionMatchBindName);
  const auto *TagField = Result.Nodes.getNodeAs<FieldDecl>(TagMatchBindName);
  assert(Root && UnionField && TagField);

  const auto *UnionDef =
      UnionField->getType().getCanonicalType().getTypePtr()->getAsRecordDecl();
  const auto *EnumDef = llvm::dyn_cast<EnumDecl>(
      TagField->getType().getCanonicalType().getTypePtr()->getAsTagDecl());
  assert(UnionDef && EnumDef && "Declarations missing!");

  const EnumConstantDecl *CountingEnumConstantDecl = nullptr;
  const std::size_t UnionMemberCount = llvm::range_size(UnionDef->fields());
  const std::size_t TagCount =
      getNumberOfValidEnumValues(EnumDef, CountingEnumConstantDecl);

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

} // namespace clang::tidy::bugprone
