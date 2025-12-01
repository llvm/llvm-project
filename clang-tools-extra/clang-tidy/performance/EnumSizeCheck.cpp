//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EnumSizeCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <limits>
#include <utility>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {

AST_MATCHER(EnumDecl, hasEnumerators) { return !Node.enumerators().empty(); }

const std::uint64_t Min8 =
    std::imaxabs(std::numeric_limits<std::int8_t>::min());
const std::uint64_t Max8 = std::numeric_limits<std::int8_t>::max();
const std::uint64_t Min16 =
    std::imaxabs(std::numeric_limits<std::int16_t>::min());
const std::uint64_t Max16 = std::numeric_limits<std::int16_t>::max();
const std::uint64_t Min32 =
    std::imaxabs(std::numeric_limits<std::int32_t>::min());
const std::uint64_t Max32 = std::numeric_limits<std::int32_t>::max();

} // namespace
static std::pair<const char *, std::uint32_t>
getNewType(std::size_t Size, std::uint64_t Min, std::uint64_t Max) noexcept {
  if (Min) {
    if (Min <= Min8 && Max <= Max8) {
      return {"std::int8_t", sizeof(std::int8_t)};
    }

    if (Min <= Min16 && Max <= Max16 && Size > sizeof(std::int16_t)) {
      return {"std::int16_t", sizeof(std::int16_t)};
    }

    if (Min <= Min32 && Max <= Max32 && Size > sizeof(std::int32_t)) {
      return {"std::int32_t", sizeof(std::int32_t)};
    }

    return {};
  }

  if (Max) {
    if (Max <= std::numeric_limits<std::uint8_t>::max()) {
      return {"std::uint8_t", sizeof(std::uint8_t)};
    }

    if (Max <= std::numeric_limits<std::uint16_t>::max() &&
        Size > sizeof(std::uint16_t)) {
      return {"std::uint16_t", sizeof(std::uint16_t)};
    }

    if (Max <= std::numeric_limits<std::uint32_t>::max() &&
        Size > sizeof(std::uint32_t)) {
      return {"std::uint32_t", sizeof(std::uint32_t)};
    }

    return {};
  }

  // Zero case
  return {"std::uint8_t", sizeof(std::uint8_t)};
}

EnumSizeCheck::EnumSizeCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      EnumIgnoreList(
          utils::options::parseStringList(Options.get("EnumIgnoreList", ""))) {}

void EnumSizeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "EnumIgnoreList",
                utils::options::serializeStringList(EnumIgnoreList));
}

bool EnumSizeCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus11;
}

void EnumSizeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      enumDecl(unless(isExpansionInSystemHeader()), isDefinition(),
               hasEnumerators(),
               unless(matchers::matchesAnyListedName(EnumIgnoreList)))
          .bind("e"),
      this);
}

void EnumSizeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<EnumDecl>("e");
  const QualType BaseType = MatchedDecl->getIntegerType().getCanonicalType();
  if (!BaseType->isIntegerType())
    return;

  const std::uint32_t Size = Result.Context->getTypeSize(BaseType) / 8U;
  if (1U == Size)
    return;

  std::uint64_t MinV = 0U;
  std::uint64_t MaxV = 0U;

  for (const auto &It : MatchedDecl->enumerators()) {
    const llvm::APSInt &InitVal = It->getInitVal();
    if ((InitVal.isUnsigned() || InitVal.isNonNegative())) {
      MaxV = std::max<std::uint64_t>(MaxV, InitVal.getZExtValue());
    } else {
      MinV = std::max<std::uint64_t>(MinV, InitVal.abs().getZExtValue());
    }
  }

  auto NewType = getNewType(Size, MinV, MaxV);
  if (!NewType.first || Size <= NewType.second)
    return;

  diag(MatchedDecl->getLocation(),
       "enum %0 uses a larger base type (%1, size: %2 %select{byte|bytes}5) "
       "than necessary for its value set, consider using '%3' (%4 "
       "%select{byte|bytes}6) as the base type to reduce its size")
      << MatchedDecl << MatchedDecl->getIntegerType() << Size << NewType.first
      << NewType.second << (Size > 1U) << (NewType.second > 1U);
}

} // namespace clang::tidy::performance
