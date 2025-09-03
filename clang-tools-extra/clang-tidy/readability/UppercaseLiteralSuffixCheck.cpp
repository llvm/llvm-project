//===--- UppercaseLiteralSuffixCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UppercaseLiteralSuffixCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <cctype>
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

struct IntegerLiteralCheck {
  using type = clang::IntegerLiteral;
  static constexpr llvm::StringLiteral Name = llvm::StringLiteral("integer");
  // What should be skipped before looking for the Suffixes? (Nothing here.)
  static constexpr llvm::StringLiteral SkipFirst = llvm::StringLiteral("");
  // Suffix can only consist of 'u', 'l', and 'z' chars, can be a bit-precise
  // integer (wb), and can be a complex number ('i', 'j'). In MS compatibility
  // mode, suffixes like i32 are supported.
  static constexpr llvm::StringLiteral Suffixes =
      llvm::StringLiteral("uUlLzZwWiIjJ");
};

struct FloatingLiteralCheck {
  using type = clang::FloatingLiteral;
  static constexpr llvm::StringLiteral Name =
      llvm::StringLiteral("floating point");
  // C++17 introduced hexadecimal floating-point literals, and 'f' is both a
  // valid hexadecimal digit in a hex float literal and a valid floating-point
  // literal suffix.
  // So we can't just "skip to the chars that can be in the suffix".
  // Since the exponent ('p'/'P') is mandatory for hexadecimal floating-point
  // literals, we first skip everything before the exponent.
  static constexpr llvm::StringLiteral SkipFirst = llvm::StringLiteral("pP");
  // Suffix can only consist of 'f', 'l', "f16", "bf16", "df", "dd", "dl",
  // 'h', 'q' chars, and can be a complex number ('i', 'j').
  static constexpr llvm::StringLiteral Suffixes =
      llvm::StringLiteral("fFlLbBdDhHqQiIjJ");
};

struct NewSuffix {
  SourceRange LiteralLocation;
  StringRef OldSuffix;
  std::optional<FixItHint> FixIt;
};

std::optional<SourceLocation> getMacroAwareLocation(SourceLocation Loc,
                                                    const SourceManager &SM) {
  // Do nothing if the provided location is invalid.
  if (Loc.isInvalid())
    return std::nullopt;
  // Look where the location was *actually* written.
  SourceLocation SpellingLoc = SM.getSpellingLoc(Loc);
  if (SpellingLoc.isInvalid())
    return std::nullopt;
  return SpellingLoc;
}

std::optional<SourceRange> getMacroAwareSourceRange(SourceRange Loc,
                                                    const SourceManager &SM) {
  std::optional<SourceLocation> Begin =
      getMacroAwareLocation(Loc.getBegin(), SM);
  std::optional<SourceLocation> End = getMacroAwareLocation(Loc.getEnd(), SM);
  if (!Begin || !End)
    return std::nullopt;
  return SourceRange(*Begin, *End);
}

std::optional<std::string>
getNewSuffix(llvm::StringRef OldSuffix,
             const std::vector<StringRef> &NewSuffixes) {
  // If there is no config, just uppercase the entirety of the suffix.
  if (NewSuffixes.empty())
    return OldSuffix.upper();
  // Else, find matching suffix, case-*insensitive*ly.
  auto NewSuffix =
      llvm::find_if(NewSuffixes, [OldSuffix](StringRef PotentialNewSuffix) {
        return OldSuffix.equals_insensitive(PotentialNewSuffix);
      });
  // Have a match, return it.
  if (NewSuffix != NewSuffixes.end())
    return NewSuffix->str();
  // Nope, I guess we have to keep it as-is.
  return std::nullopt;
}

template <typename LiteralType>
std::optional<NewSuffix>
shouldReplaceLiteralSuffix(const Expr &Literal,
                           const std::vector<StringRef> &NewSuffixes,
                           const SourceManager &SM, const LangOptions &LO) {
  NewSuffix ReplacementDsc;

  const auto &L = cast<typename LiteralType::type>(Literal);

  // The naive location of the literal. Is always valid.
  ReplacementDsc.LiteralLocation = L.getSourceRange();

  // Was this literal fully spelled or is it a product of macro expansion?
  bool RangeCanBeFixed =
      utils::rangeCanBeFixed(ReplacementDsc.LiteralLocation, &SM);

  // The literal may have macro expansion, we need the final expanded src range.
  std::optional<SourceRange> Range =
      getMacroAwareSourceRange(ReplacementDsc.LiteralLocation, SM);
  if (!Range)
    return std::nullopt;

  if (RangeCanBeFixed)
    ReplacementDsc.LiteralLocation = *Range;
  // Else keep the naive literal location!

  // Get the whole literal from the source buffer.
  bool Invalid = false;
  const StringRef LiteralSourceText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(*Range), SM, LO, &Invalid);
  assert(!Invalid && "Failed to retrieve the source text.");

  // Make sure the first character is actually a digit, instead of
  // something else, like a non-type template parameter.
  if (!std::isdigit(static_cast<unsigned char>(LiteralSourceText.front())))
    return std::nullopt;

  size_t Skip = 0;

  // Do we need to ignore something before actually looking for the suffix?
  if (!LiteralType::SkipFirst.empty()) {
    // E.g. we can't look for 'f' suffix in hexadecimal floating-point literals
    // until after we skip to the exponent (which is mandatory there),
    // because hex-digit-sequence may contain 'f'.
    Skip = LiteralSourceText.find_first_of(LiteralType::SkipFirst);
    // We could be in non-hexadecimal floating-point literal, with no exponent.
    if (Skip == StringRef::npos)
      Skip = 0;
  }

  // Find the beginning of the suffix by looking for the first char that is
  // one of these chars that can be in the suffix, potentially starting looking
  // in the exponent, if we are skipping hex-digit-sequence.
  Skip = LiteralSourceText.find_first_of(LiteralType::Suffixes, /*From=*/Skip);

  // We can't check whether the *Literal has any suffix or not without actually
  // looking for the suffix. So it is totally possible that there is no suffix.
  if (Skip == StringRef::npos)
    return std::nullopt;

  // Move the cursor in the source range to the beginning of the suffix.
  Range->setBegin(Range->getBegin().getLocWithOffset(Skip));
  // And in our textual representation too.
  ReplacementDsc.OldSuffix = LiteralSourceText.drop_front(Skip);
  assert(!ReplacementDsc.OldSuffix.empty() &&
         "We still should have some chars left.");

  // And get the replacement suffix.
  std::optional<std::string> NewSuffix =
      getNewSuffix(ReplacementDsc.OldSuffix, NewSuffixes);
  if (!NewSuffix || ReplacementDsc.OldSuffix == *NewSuffix)
    return std::nullopt; // The suffix was already the way it should be.

  if (RangeCanBeFixed)
    ReplacementDsc.FixIt = FixItHint::CreateReplacement(*Range, *NewSuffix);

  return ReplacementDsc;
}

} // namespace

UppercaseLiteralSuffixCheck::UppercaseLiteralSuffixCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      NewSuffixes(
          utils::options::parseStringList(Options.get("NewSuffixes", ""))),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)) {}

void UppercaseLiteralSuffixCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "NewSuffixes",
                utils::options::serializeStringList(NewSuffixes));
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void UppercaseLiteralSuffixCheck::registerMatchers(MatchFinder *Finder) {
  // Sadly, we can't check whether the literal has suffix or not.
  // E.g. i32 suffix still results in 'BuiltinType::Kind::Int'.
  // And such an info is not stored in the *Literal itself.
  Finder->addMatcher(
      stmt(eachOf(integerLiteral().bind(IntegerLiteralCheck::Name),
                  floatLiteral().bind(FloatingLiteralCheck::Name)),
           unless(anyOf(hasParent(userDefinedLiteral()),
                        hasAncestor(substNonTypeTemplateParmExpr())))),
      this);
}

template <typename LiteralType>
bool UppercaseLiteralSuffixCheck::checkBoundMatch(
    const MatchFinder::MatchResult &Result) {
  const auto *Literal =
      Result.Nodes.getNodeAs<typename LiteralType::type>(LiteralType::Name);
  if (!Literal)
    return false;

  // We won't *always* want to diagnose.
  // We might have a suffix that is already uppercase.
  if (auto Details = shouldReplaceLiteralSuffix<LiteralType>(
          *Literal, NewSuffixes, *Result.SourceManager, getLangOpts())) {
    if (Details->LiteralLocation.getBegin().isMacroID() && IgnoreMacros)
      return true;
    auto Complaint = diag(Details->LiteralLocation.getBegin(),
                          "%0 literal has suffix '%1', which is not uppercase")
                     << LiteralType::Name << Details->OldSuffix;
    if (Details->FixIt) // Similarly, a fix-it is not always possible.
      Complaint << *(Details->FixIt);
  }

  return true;
}

void UppercaseLiteralSuffixCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (checkBoundMatch<IntegerLiteralCheck>(Result))
    return; // If it *was* IntegerLiteral, don't check for FloatingLiteral.
  checkBoundMatch<FloatingLiteralCheck>(Result);
}

} // namespace clang::tidy::readability
