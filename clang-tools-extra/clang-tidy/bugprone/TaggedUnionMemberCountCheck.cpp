//===--- TaggedUnionMemberCountCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TaggedUnionMemberCountCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <limits>
#include <iostream>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

TaggedUnionMemberCountCheck::TaggedUnionMemberCountCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        EnumCounterHeuristicIsEnabled(Options.get("EnumCounterHeuristicIsEnabled", true)),
        EnumCounterSuffix(Options.get("EnumCounterSuffix", "count")),
        StrictMode(Options.get("StrictMode", true)) { }

void TaggedUnionMemberCountCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "EnumCounterHeuristicIsEnabled", EnumCounterHeuristicIsEnabled);
  Options.store(Opts, "EnumCounterSuffix", EnumCounterSuffix);
}

void TaggedUnionMemberCountCheck::registerMatchers(MatchFinder *Finder) {
Finder->addMatcher(
      recordDecl(
          allOf(isStruct(),
				has(fieldDecl(hasType(qualType(hasCanonicalType(recordType())))).bind("union")),
				has(fieldDecl(hasType(qualType(hasCanonicalType(enumType())))).bind("tags"))))
          .bind("root"),
      this);
}

static bool isUnion(const FieldDecl *R) {
	return R->getType().getCanonicalType().getTypePtr()->isUnionType();
}

static bool isEnum(const FieldDecl *R) {
	return R->getType().getCanonicalType().getTypePtr()->isEnumeralType();
}

static bool hasMultipleUnionsOrEnums(const RecordDecl *rec) {
  return llvm::count_if(rec->fields(), isUnion) > 1 ||
         llvm::count_if(rec->fields(), isEnum) > 1;
}

static size_t getNumberOfValidEnumValues(const EnumDecl *ed, bool EnumCounterHeuristicIsEnabled, StringRef EnumCounterSuffix) {
  int64_t maxTagValue = std::numeric_limits<int64_t>::min();
  int64_t minTagValue = std::numeric_limits<int64_t>::max();

  // Heuristic for counter enum constants.
  //
  //   enum tag_with_counter {
  //     tag1,
  //     tag2,
  //     tag_count, <-- Searching for these enum constants
  //   };
  //
  // The 'ce' prefix is used to abbreviate counterEnum.
  // The final tag count is decreased by 1 if and only if:
  // 1. The number of counting enum constants = 1,
  int ceCount = 0;
  // 2. The counting enum constant is the last enum constant that is defined,
  int ceFirstIndex = 0;
  // 3. The value of the counting enum constant is the largest out of every enum constant.
  int64_t ceValue = 0;

  int64_t enumConstantsCount = 0;
  for (auto En : llvm::enumerate(ed->enumerators())) {
    enumConstantsCount += 1;

    int64_t enumValue = En.value()->getInitVal().getExtValue();
    StringRef enumName = En.value()->getName();

    if (enumValue > maxTagValue)
      maxTagValue = enumValue;
    if (enumValue < minTagValue)
      minTagValue = enumValue;

    if (enumName.ends_with_insensitive(EnumCounterSuffix)) {
      if (ceCount == 0) {
        ceFirstIndex = En.index();
      }
      ceValue = enumValue;
      ceCount += 1;
    }
  }

  int64_t validValuesCount = maxTagValue - minTagValue + 1;
  if (EnumCounterHeuristicIsEnabled &&
      ceCount == 1 &&
      ceFirstIndex == enumConstantsCount - 1 &&
      ceValue == maxTagValue) {
    validValuesCount -= 1;
  }
  return validValuesCount;
}

// Feladatok:
// - typedef tesztelés
// - template tesztelés
// - névtelen union tesztelés
// - "count" paraméterezése
void TaggedUnionMemberCountCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *root = Result.Nodes.getNodeAs<RecordDecl>("root");
  const auto *unionField = Result.Nodes.getNodeAs<FieldDecl>("union");
  const auto *tagField = Result.Nodes.getNodeAs<FieldDecl>("tags");

  // The matcher can only narrow down the type to recordType()
  if (!isUnion(unionField))
    return;

  if (hasMultipleUnionsOrEnums(root))
    return;

  const auto *unionDef = unionField->getType().getCanonicalType().getTypePtr()->getAsRecordDecl();
  const auto *enumDef = static_cast<EnumDecl*>(tagField->getType().getCanonicalType().getTypePtr()->getAsTagDecl());

  size_t unionMemberCount = llvm::range_size(unionDef->fields());
  size_t tagCount = getNumberOfValidEnumValues(enumDef, EnumCounterHeuristicIsEnabled, EnumCounterSuffix);

  // FIXME: Maybe a emit a note when a counter enum constant was found.
  if (unionMemberCount > tagCount) {
    diag(root->getLocation(), "Tagged union has more data members (%0) than tags (%1)!")
        << unionMemberCount << tagCount;
  } else if (StrictMode && unionMemberCount < tagCount) {
    diag(root->getLocation(), "Tagged union has fewer data members (%0) than tags (%1)!")
        << unionMemberCount << tagCount;
  }
}

} // namespace clang::tidy::bugprone
