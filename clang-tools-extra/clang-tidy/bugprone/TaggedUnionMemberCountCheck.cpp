//===----------------------------------------------------------------------===//
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

AST_MATCHER_P2(RecordDecl, fieldCountOfKindIsOne,
               ast_matchers::internal::Matcher<FieldDecl>, InnerMatcher,
               StringRef, BindName) {
  // BoundNodesTreeBuilder resets itself when a match occurs.
  // So to avoid losing previously saved binds, a temporary instance
  // is used for matching.
  //
  // For precedence, see commit: 5b07de1a5faf4a22ae6fd982b877c5e7e3a76559
  clang::ast_matchers::internal::BoundNodesTreeBuilder TempBuilder;

  const FieldDecl *FirstMatch = nullptr;
  for (const FieldDecl *Field : Node.fields()) {
    if (InnerMatcher.matches(*Field, Finder, &TempBuilder)) {
      if (FirstMatch) {
        return false;
      }
      FirstMatch = Field;
    }
  }

  if (FirstMatch) {
    Builder->setBinding(BindName, clang::DynTypedNode::create(*FirstMatch));
    return true;
  }
  return false;
}

} // namespace

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
    if (Options.get(CountingEnumPrefixesOptionName))
      configurationDiag("%0: Counting enum heuristic is disabled but "
                        "%1 is set")
          << Name << CountingEnumPrefixesOptionName;
    if (Options.get(CountingEnumSuffixesOptionName))
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

  auto NotFromSystemHeaderOrStdNamespace =
      unless(anyOf(isExpansionInSystemHeader(), isInStdNamespace()));

  auto UnionField =
      fieldDecl(hasType(qualType(hasCanonicalType(recordType(hasDeclaration(
          recordDecl(isUnion(), NotFromSystemHeaderOrStdNamespace)))))));

  auto EnumField = fieldDecl(hasType(qualType(hasCanonicalType(
      enumType(hasDeclaration(enumDecl(NotFromSystemHeaderOrStdNamespace)))))));

  auto HasOneUnionField = fieldCountOfKindIsOne(UnionField, UnionMatchBindName);
  auto HasOneEnumField = fieldCountOfKindIsOne(EnumField, TagMatchBindName);

  Finder->addMatcher(recordDecl(anyOf(isStruct(), isClass()), HasOneUnionField,
                                HasOneEnumField, unless(isImplicit()))
                         .bind(RootMatchBindName),
                     this);
}

bool TaggedUnionMemberCountCheck::isCountingEnumLikeName(StringRef Name) const {
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

std::pair<const std::size_t, const EnumConstantDecl *>
TaggedUnionMemberCountCheck::getNumberOfEnumValues(const EnumDecl *ED) {
  llvm::SmallSet<llvm::APSInt, 16> EnumValues;

  const EnumConstantDecl *LastEnumConstant = nullptr;
  for (const EnumConstantDecl *Enumerator : ED->enumerators()) {
    EnumValues.insert(Enumerator->getInitVal());
    LastEnumConstant = Enumerator;
  }

  if (EnableCountingEnumHeuristic && LastEnumConstant &&
      isCountingEnumLikeName(LastEnumConstant->getName()) &&
      llvm::APSInt::isSameValue(LastEnumConstant->getInitVal(),
                                llvm::APSInt::get(EnumValues.size() - 1))) {
    return {EnumValues.size() - 1, LastEnumConstant};
  }

  return {EnumValues.size(), nullptr};
}

void TaggedUnionMemberCountCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Root = Result.Nodes.getNodeAs<RecordDecl>(RootMatchBindName);
  const auto *UnionField =
      Result.Nodes.getNodeAs<FieldDecl>(UnionMatchBindName);
  const auto *TagField = Result.Nodes.getNodeAs<FieldDecl>(TagMatchBindName);

  assert(Root && "Root is missing!");
  assert(UnionField && "UnionField is missing!");
  assert(TagField && "TagField is missing!");
  if (!Root || !UnionField || !TagField)
    return;

  const auto *UnionDef = UnionField->getType()->castAsRecordDecl();
  const auto *EnumDef = TagField->getType()->castAsEnumDecl();

  const std::size_t UnionMemberCount = llvm::range_size(UnionDef->fields());
  auto [TagCount, CountingEnumConstantDecl] = getNumberOfEnumValues(EnumDef);

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
