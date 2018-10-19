//===--- UppercaseLiteralSuffixCheck.cpp - clang-tidy ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UppercaseLiteralSuffixCheck.h"
#include "../utils/ASTUtils.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

struct IntegerLiteralCheck {
  using type = clang::IntegerLiteral;
  static constexpr llvm::StringLiteral Name = llvm::StringLiteral("integer");
  // What should be skipped before looking for the Suffixes? (Nothing here.)
  static constexpr llvm::StringLiteral SkipFirst = llvm::StringLiteral("");
  // Suffix can only consist of 'u' and 'l' chars, and can be a complex number
  // ('i', 'j'). In MS compatibility mode, suffixes like i32 are supported.
  static constexpr llvm::StringLiteral Suffixes =
      llvm::StringLiteral("uUlLiIjJ");
};
constexpr llvm::StringLiteral IntegerLiteralCheck::Name;
constexpr llvm::StringLiteral IntegerLiteralCheck::SkipFirst;
constexpr llvm::StringLiteral IntegerLiteralCheck::Suffixes;

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
  // Suffix can only consist of 'f', 'l', "f16", 'h', 'q' chars,
  // and can be a complex number ('i', 'j').
  static constexpr llvm::StringLiteral Suffixes =
      llvm::StringLiteral("fFlLhHqQiIjJ");
};
constexpr llvm::StringLiteral FloatingLiteralCheck::Name;
constexpr llvm::StringLiteral FloatingLiteralCheck::SkipFirst;
constexpr llvm::StringLiteral FloatingLiteralCheck::Suffixes;

struct NewSuffix {
  SourceLocation LiteralLocation;
  StringRef OldSuffix;
  llvm::Optional<FixItHint> FixIt;
};

llvm::Optional<SourceLocation> GetMacroAwareLocation(SourceLocation Loc,
                                                     const SourceManager &SM) {
  // Do nothing if the provided location is invalid.
  if (Loc.isInvalid())
    return llvm::None;
  // Look where the location was *actually* written.
  SourceLocation SpellingLoc = SM.getSpellingLoc(Loc);
  if (SpellingLoc.isInvalid())
    return llvm::None;
  return SpellingLoc;
}

llvm::Optional<SourceRange> GetMacroAwareSourceRange(SourceRange Loc,
                                                     const SourceManager &SM) {
  llvm::Optional<SourceLocation> Begin =
      GetMacroAwareLocation(Loc.getBegin(), SM);
  llvm::Optional<SourceLocation> End = GetMacroAwareLocation(Loc.getEnd(), SM);
  if (!Begin || !End)
    return llvm::None;
  return SourceRange(*Begin, *End);
}

llvm::Optional<std::string>
getNewSuffix(llvm::StringRef OldSuffix,
             const std::vector<std::string> &NewSuffixes) {
  // If there is no config, just uppercase the entirety of the suffix.
  if (NewSuffixes.empty())
    return OldSuffix.upper();
  // Else, find matching suffix, case-*insensitive*ly.
  auto NewSuffix = llvm::find_if(
      NewSuffixes, [OldSuffix](const std::string &PotentialNewSuffix) {
        return OldSuffix.equals_lower(PotentialNewSuffix);
      });
  // Have a match, return it.
  if (NewSuffix != NewSuffixes.end())
    return *NewSuffix;
  // Nope, I guess we have to keep it as-is.
  return llvm::None;
}

template <typename LiteralType>
llvm::Optional<NewSuffix>
shouldReplaceLiteralSuffix(const Expr &Literal,
                           const std::vector<std::string> &NewSuffixes,
                           const SourceManager &SM, const LangOptions &LO) {
  NewSuffix ReplacementDsc;

  const auto &L = cast<typename LiteralType::type>(Literal);

  // The naive location of the literal. Is always valid.
  ReplacementDsc.LiteralLocation = L.getLocation();

  // Was this literal fully spelled or is it a product of macro expansion?
  bool RangeCanBeFixed =
      utils::rangeCanBeFixed(ReplacementDsc.LiteralLocation, &SM);

  // The literal may have macro expansion, we need the final expanded src range.
  llvm::Optional<SourceRange> Range =
      GetMacroAwareSourceRange(ReplacementDsc.LiteralLocation, SM);
  if (!Range)
    return llvm::None;

  if (RangeCanBeFixed)
    ReplacementDsc.LiteralLocation = Range->getBegin();
  // Else keep the naive literal location!

  // Get the whole literal from the source buffer.
  bool Invalid;
  const StringRef LiteralSourceText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(*Range), SM, LO, &Invalid);
  assert(!Invalid && "Failed to retrieve the source text.");

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
    return llvm::None;

  // Move the cursor in the source range to the beginning of the suffix.
  Range->setBegin(Range->getBegin().getLocWithOffset(Skip));
  // And in our textual representation too.
  ReplacementDsc.OldSuffix = LiteralSourceText.drop_front(Skip);
  assert(!ReplacementDsc.OldSuffix.empty() &&
         "We still should have some chars left.");

  // And get the replacement suffix.
  llvm::Optional<std::string> NewSuffix =
      getNewSuffix(ReplacementDsc.OldSuffix, NewSuffixes);
  if (!NewSuffix || ReplacementDsc.OldSuffix == *NewSuffix)
    return llvm::None; // The suffix was already the way it should be.

  if (RangeCanBeFixed)
    ReplacementDsc.FixIt = FixItHint::CreateReplacement(*Range, *NewSuffix);

  return ReplacementDsc;
}

} // namespace

UppercaseLiteralSuffixCheck::UppercaseLiteralSuffixCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      NewSuffixes(
          utils::options::parseStringList(Options.get("NewSuffixes", ""))) {}

void UppercaseLiteralSuffixCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "NewSuffixes",
                utils::options::serializeStringList(NewSuffixes));
}

void UppercaseLiteralSuffixCheck::registerMatchers(MatchFinder *Finder) {
  // Sadly, we can't check whether the literal has sufix or not.
  // E.g. i32 suffix still results in 'BuiltinType::Kind::Int'.
  // And such an info is not stored in the *Literal itself.
  Finder->addMatcher(
      stmt(allOf(eachOf(integerLiteral().bind(IntegerLiteralCheck::Name),
                        floatLiteral().bind(FloatingLiteralCheck::Name)),
                 unless(hasParent(userDefinedLiteral())))),
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
    auto Complaint = diag(Details->LiteralLocation,
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

} // namespace readability
} // namespace tidy
} // namespace clang
