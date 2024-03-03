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

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void TaggedUnionMemberCountCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      recordDecl(
          allOf(isStruct(),
                has(fieldDecl(hasType(recordDecl(isUnion()).bind("union")))),
                has(fieldDecl(hasType(enumDecl().bind("tags"))))))
          .bind("root"),
      this);
}

static bool hasMultipleUnionsOrEnums(const RecordDecl *rec) {
  int tags = 0;
  int unions = 0;
  for (const FieldDecl *r : rec->fields()) {
    TypeSourceInfo *info = r->getTypeSourceInfo();
    QualType qualtype = info->getType();
    const Type *type = qualtype.getTypePtr();
    if (type->isUnionType())
      unions += 1;
    else if (type->isEnumeralType())
      tags += 1;
    if (tags > 1 || unions > 1)
      return true;
  }
  return false;
}

static int64_t getNumberOfValidEnumValues(const EnumDecl *ed) {
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

    if (enumName.ends_with_insensitive("count")) {
      if (ceCount == 0) {
        ceFirstIndex = En.index();
      }
      ceValue = enumValue;
      ceCount += 1;
    }
  }

  int64_t validValuesCount = maxTagValue - minTagValue + 1;
  if (ceCount == 1 &&
      ceFirstIndex == enumConstantsCount - 1 &&
      ceValue == maxTagValue) {
    validValuesCount -= 1;
  }
  return validValuesCount;
}

void TaggedUnionMemberCountCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *root = Result.Nodes.getNodeAs<RecordDecl>("root");
  const auto *unionMatch = Result.Nodes.getNodeAs<RecordDecl>("union");
  const auto *tagMatch = Result.Nodes.getNodeAs<EnumDecl>("tags");

  if (hasMultipleUnionsOrEnums(root))
    return;

  int64_t unionMemberCount = llvm::range_size(unionMatch->fields());
  int64_t tagCount = getNumberOfValidEnumValues(tagMatch);

  // FIXME: Maybe a emit a note when a counter enum constant was found.
  if (unionMemberCount > tagCount) {
    diag(root->getLocation(), "Tagged union has more data members than tags! "
                              "Data members: %0 Tags: %1")
        << unionMemberCount << tagCount;
  } else if (unionMemberCount < tagCount) {
    diag(root->getLocation(), "Tagged union has fewer data members than tags! "
                              "Data members: %0 Tags: %1")
        << unionMemberCount << tagCount;
  }
}

} // namespace clang::tidy::bugprone
