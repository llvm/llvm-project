//===- Tokens.cpp - collect tokens from preprocessing ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Tooling/Syntax/Tokens.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace clang::syntax;

namespace {
// Finds the smallest consecutive subsuquence of Toks that covers R.
llvm::ArrayRef<syntax::Token>
getTokensCovering(llvm::ArrayRef<syntax::Token> Toks, SourceRange R,
                  const SourceManager &SM) {
  if (R.isInvalid())
    return {};
  const syntax::Token *Begin =
      llvm::partition_point(Toks, [&](const syntax::Token &T) {
        return SM.isBeforeInTranslationUnit(T.location(), R.getBegin());
      });
  const syntax::Token *End =
      llvm::partition_point(Toks, [&](const syntax::Token &T) {
        return !SM.isBeforeInTranslationUnit(R.getEnd(), T.location());
      });
  if (Begin > End)
    return {};
  return {Begin, End};
}

// Finds the range within FID corresponding to expanded tokens [First, Last].
// Prev precedes First and Next follows Last, these must *not* be included.
// If no range satisfies the criteria, returns an invalid range.
//
// #define ID(x) x
// ID(ID(ID(a1) a2))
//          ~~       -> a1
//              ~~   -> a2
//       ~~~~~~~~~   -> a1 a2
SourceRange spelledForExpandedSlow(SourceLocation First, SourceLocation Last,
                                   SourceLocation Prev, SourceLocation Next,
                                   FileID TargetFile,
                                   const SourceManager &SM) {
  // There are two main parts to this algorithm:
  //  - identifying which spelled range covers the expanded tokens
  //  - validating that this range doesn't cover any extra tokens (First/Last)
  //
  // We do these in order. However as we transform the expanded range into the
  // spelled one, we adjust First/Last so the validation remains simple.

  assert(SM.getSLocEntry(TargetFile).isFile());
  // In most cases, to select First and Last we must return their expansion
  // range, i.e. the whole of any macros they are included in.
  //
  // When First and Last are part of the *same macro arg* of a macro written
  // in TargetFile, we that slice of the arg, i.e. their spelling range.
  //
  // Unwrap such macro calls. If the target file has A(B(C)), the
  // SourceLocation stack of a token inside C shows us the expansion of A first,
  // then B, then any macros inside C's body, then C itself.
  // (This is the reverse of the order the PP applies the expansions in).
  while (First.isMacroID() && Last.isMacroID()) {
    auto DecFirst = SM.getDecomposedLoc(First);
    auto DecLast = SM.getDecomposedLoc(Last);
    auto &ExpFirst = SM.getSLocEntry(DecFirst.first).getExpansion();
    auto &ExpLast = SM.getSLocEntry(DecLast.first).getExpansion();

    if (!ExpFirst.isMacroArgExpansion() || !ExpLast.isMacroArgExpansion())
      break;
    // Locations are in the same macro arg if they expand to the same place.
    // (They may still have different FileIDs - an arg can have >1 chunks!)
    if (ExpFirst.getExpansionLocStart() != ExpLast.getExpansionLocStart())
      break;
    // Careful, given:
    //   #define HIDE ID(ID(a))
    //   ID(ID(HIDE))
    // The token `a` is wrapped in 4 arg-expansions, we only want to unwrap 2.
    // We distinguish them by whether the macro expands into the target file.
    // Fortunately, the target file ones will always appear first.
    auto ExpFileID = SM.getFileID(ExpFirst.getExpansionLocStart());
    if (ExpFileID == TargetFile)
      break;
    // Replace each endpoint with its spelling inside the macro arg.
    // (This is getImmediateSpellingLoc without repeating lookups).
    First = ExpFirst.getSpellingLoc().getLocWithOffset(DecFirst.second);
    Last = ExpLast.getSpellingLoc().getLocWithOffset(DecLast.second);
  }

  // In all remaining cases we need the full containing macros.
  // If this overlaps Prev or Next, then no range is possible.
  SourceRange Candidate =
      SM.getExpansionRange(SourceRange(First, Last)).getAsRange();
  auto DecFirst = SM.getDecomposedExpansionLoc(Candidate.getBegin());
  auto DecLast = SM.getDecomposedExpansionLoc(Candidate.getEnd());
  // Can end up in the wrong file due to bad input or token-pasting shenanigans.
  if (Candidate.isInvalid() || DecFirst.first != TargetFile ||
      DecLast.first != TargetFile)
    return SourceRange();
  // Check bounds, which may still be inside macros.
  if (Prev.isValid()) {
    auto Dec = SM.getDecomposedLoc(SM.getExpansionRange(Prev).getBegin());
    if (Dec.first != DecFirst.first || Dec.second >= DecFirst.second)
      return SourceRange();
  }
  if (Next.isValid()) {
    auto Dec = SM.getDecomposedLoc(SM.getExpansionRange(Next).getEnd());
    if (Dec.first != DecLast.first || Dec.second <= DecLast.second)
      return SourceRange();
  }
  // Now we know that Candidate is a file range that covers [First, Last]
  // without encroaching on {Prev, Next}. Ship it!
  return Candidate;
}

// Returns the location in the token that the token at Loc was split from.
SourceLocation getLocBeforeSplit(SourceLocation Loc, const SourceManager &SM) {
  // A token can be split more than once, e.g. CUDA's '>>>'. The remainder of a
  // split token can start inside the split token's buffer, so keep the offset.
  while (Loc.isMacroID()) {
    auto [FID, Offset] = SM.getDecomposedLoc(Loc);
    const SrcMgr::ExpansionInfo &Exp = SM.getSLocEntry(FID).getExpansion();
    if (Exp.isExpansionTokenRange())
      break;
    Loc = Exp.getExpansionLocStart().getLocWithOffset(Offset);
  }
  return Loc;
}

// Returns the length of the text in [Loc, Loc + Length) after removing
// trailing escaped newlines.
unsigned lengthWithoutTrailingEscapedNewlines(SourceLocation Loc,
                                              unsigned Length,
                                              const SourceManager &SM,
                                              const LangOptions &LangOpts) {
  const char *Text = SM.getCharacterData(Loc);
  unsigned End = 0;
  while (End < Length) {
    // Size includes the escaped newlines before the character.
    unsigned Size = Lexer::getCharAndSizeNoWarn(Text + End, LangOpts).Size;
    if (End + Size > Length)
      break; // Only escaped newlines are left.
    End += Size;
  }
  return End;
}

// Returns the kind of the token that remains when a token of kind SplitKind is
// split from the front of a token of kind Kind, e.g. returns `equal` for a '>'
// split from a '>='. Returns std::nullopt if the spelling of SplitKind is not a
// prefix of the spelling of Kind, or the rest of the spelling is not a
// punctuator.
std::optional<tok::TokenKind> getRemainderKind(tok::TokenKind Kind,
                                               tok::TokenKind SplitKind) {
  const char *Spelling = tok::getPunctuatorSpelling(Kind);
  const char *SplitSpelling = tok::getPunctuatorSpelling(SplitKind);
  if (!Spelling || !SplitSpelling)
    return std::nullopt;
  llvm::StringRef Rest = Spelling;
  if (!Rest.consume_front(SplitSpelling) || Rest.empty())
    return std::nullopt;
#define PUNCTUATOR(X, Y)                                                       \
  if (Rest == Y)                                                               \
    return tok::X;
#include "clang/Basic/TokenKinds.def"
  return std::nullopt;
}

} // namespace

syntax::Token::Token(SourceLocation Location, unsigned Length,
                     tok::TokenKind Kind)
    : Location(Location), Length(Length), Kind(Kind) {
  assert(Location.isValid());
}

syntax::Token::Token(const clang::Token &T)
    : Token(T.getLocation(), T.getLength(), T.getKind()) {
  assert(!T.isAnnotation());
}

llvm::StringRef syntax::Token::text(const SourceManager &SM) const {
  bool Invalid = false;
  const char *Start = SM.getCharacterData(location(), &Invalid);
  assert(!Invalid);
  return llvm::StringRef(Start, length());
}

FileRange syntax::Token::range(const SourceManager &SM) const {
  assert(location().isFileID() && "must be a spelled token");
  FileID File;
  unsigned StartOffset;
  std::tie(File, StartOffset) = SM.getDecomposedLoc(location());
  return FileRange(File, StartOffset, StartOffset + length());
}

FileRange syntax::Token::range(const SourceManager &SM,
                               const syntax::Token &First,
                               const syntax::Token &Last) {
  auto F = First.range(SM);
  auto L = Last.range(SM);
  assert(F.file() == L.file() && "tokens from different files");
  assert((F == L || F.endOffset() <= L.beginOffset()) &&
         "wrong order of tokens");
  return FileRange(F.file(), F.beginOffset(), L.endOffset());
}

llvm::raw_ostream &syntax::operator<<(llvm::raw_ostream &OS, const Token &T) {
  return OS << T.str();
}

FileRange::FileRange(FileID File, unsigned BeginOffset, unsigned EndOffset)
    : File(File), Begin(BeginOffset), End(EndOffset) {
  assert(File.isValid());
  assert(BeginOffset <= EndOffset);
}

FileRange::FileRange(const SourceManager &SM, SourceLocation BeginLoc,
                     unsigned Length) {
  assert(BeginLoc.isValid());
  assert(BeginLoc.isFileID());

  std::tie(File, Begin) = SM.getDecomposedLoc(BeginLoc);
  End = Begin + Length;
}
FileRange::FileRange(const SourceManager &SM, SourceLocation BeginLoc,
                     SourceLocation EndLoc) {
  assert(BeginLoc.isValid());
  assert(BeginLoc.isFileID());
  assert(EndLoc.isValid());
  assert(EndLoc.isFileID());
  assert(SM.getFileID(BeginLoc) == SM.getFileID(EndLoc));
  assert(SM.getFileOffset(BeginLoc) <= SM.getFileOffset(EndLoc));

  std::tie(File, Begin) = SM.getDecomposedLoc(BeginLoc);
  End = SM.getFileOffset(EndLoc);
}

llvm::raw_ostream &syntax::operator<<(llvm::raw_ostream &OS,
                                      const FileRange &R) {
  return OS << llvm::formatv("FileRange(file = {0}, offsets = {1}-{2})",
                             R.file().getHashValue(), R.beginOffset(),
                             R.endOffset());
}

llvm::StringRef FileRange::text(const SourceManager &SM) const {
  bool Invalid = false;
  StringRef Text = SM.getBufferData(File, &Invalid);
  if (Invalid)
    return "";
  assert(Begin <= Text.size());
  assert(End <= Text.size());
  return Text.substr(Begin, length());
}

void TokenBuffer::indexExpandedTokens() {
  // No-op if the index is already created.
  if (!ExpandedTokIndex.empty())
    return;
  ExpandedTokIndex.reserve(ExpandedTokens.size());
  // Index ExpandedTokens for faster lookups by SourceLocation.
  for (size_t I = 0, E = ExpandedTokens.size(); I != E; ++I) {
    SourceLocation Loc = ExpandedTokens[I].location();
    if (Loc.isValid())
      ExpandedTokIndex[Loc] = I;
  }
}

llvm::ArrayRef<syntax::Token> TokenBuffer::expandedTokens(SourceRange R) const {
  if (R.isInvalid())
    return {};
  if (!ExpandedTokIndex.empty()) {
    // Quick lookup if `R` is a token range.
    // This is a huge win since majority of the users use ranges provided by an
    // AST. Ranges in AST are token ranges from expanded token stream.
    const auto B = ExpandedTokIndex.find(R.getBegin());
    const auto E = ExpandedTokIndex.find(R.getEnd());
    if (B != ExpandedTokIndex.end() && E != ExpandedTokIndex.end()) {
      const Token *L = ExpandedTokens.data() + B->getSecond();
      // Add 1 to End to make a half-open range.
      const Token *R = ExpandedTokens.data() + E->getSecond() + 1;
      if (L > R)
        return {};
      return {L, R};
    }
  }
  // Slow case. Use `isBeforeInTranslationUnit` to binary search for the
  // required range.
  return getTokensCovering(expandedTokens(), R, *SourceMgr);
}

CharSourceRange FileRange::toCharRange(const SourceManager &SM) const {
  return CharSourceRange(
      SourceRange(SM.getComposedLoc(File, Begin), SM.getComposedLoc(File, End)),
      /*IsTokenRange=*/false);
}

std::pair<const syntax::Token *, const TokenBuffer::Mapping *>
TokenBuffer::spelledForExpandedToken(const syntax::Token *Expanded) const {
  assert(Expanded);
  assert(ExpandedTokens.data() <= Expanded &&
         Expanded < ExpandedTokens.data() + ExpandedTokens.size());

  auto FileIt = Files.find(
      SourceMgr->getFileID(SourceMgr->getExpansionLoc(Expanded->location())));
  assert(FileIt != Files.end() && "no file for an expanded token");

  const MarkedFile &File = FileIt->second;

  unsigned ExpandedIndex = Expanded - ExpandedTokens.data();
  // Find the first mapping that produced tokens after \p Expanded.
  auto It = llvm::partition_point(File.Mappings, [&](const Mapping &M) {
    return M.BeginExpanded <= ExpandedIndex;
  });
  // Our token could only be produced by the previous mapping.
  if (It == File.Mappings.begin()) {
    // No previous mapping, no need to modify offsets.
    return {&File.SpelledTokens[ExpandedIndex - File.BeginExpanded],
            /*Mapping=*/nullptr};
  }
  --It; // 'It' now points to last mapping that started before our token.

  // Check if the token is part of the mapping.
  if (ExpandedIndex < It->EndExpanded)
    return {&File.SpelledTokens[It->BeginSpelled], /*Mapping=*/&*It};

  // Not part of the mapping, use the index from previous mapping to compute the
  // corresponding spelled token.
  return {
      &File.SpelledTokens[It->EndSpelled + (ExpandedIndex - It->EndExpanded)],
      /*Mapping=*/nullptr};
}

const TokenBuffer::Mapping *
TokenBuffer::mappingStartingBeforeSpelled(const MarkedFile &F,
                                          const syntax::Token *Spelled) {
  assert(F.SpelledTokens.data() <= Spelled);
  unsigned SpelledI = Spelled - F.SpelledTokens.data();
  assert(SpelledI < F.SpelledTokens.size());

  auto It = llvm::partition_point(F.Mappings, [SpelledI](const Mapping &M) {
    return M.BeginSpelled <= SpelledI;
  });
  if (It == F.Mappings.begin())
    return nullptr;
  --It;
  return &*It;
}

llvm::SmallVector<llvm::ArrayRef<syntax::Token>, 1>
TokenBuffer::expandedForSpelled(llvm::ArrayRef<syntax::Token> Spelled) const {
  if (Spelled.empty())
    return {};
  const auto &File = fileForSpelled(Spelled);

  auto *FrontMapping = mappingStartingBeforeSpelled(File, &Spelled.front());
  unsigned SpelledFrontI = &Spelled.front() - File.SpelledTokens.data();
  assert(SpelledFrontI < File.SpelledTokens.size());
  unsigned ExpandedBegin;
  if (!FrontMapping) {
    // No mapping that starts before the first token of Spelled, we don't have
    // to modify offsets.
    ExpandedBegin = File.BeginExpanded + SpelledFrontI;
  } else if (SpelledFrontI < FrontMapping->EndSpelled) {
    // This mapping applies to Spelled tokens.
    if (SpelledFrontI != FrontMapping->BeginSpelled) {
      // Spelled tokens don't cover the entire mapping, returning empty result.
      return {}; // FIXME: support macro arguments.
    }
    // Spelled tokens start at the beginning of this mapping.
    ExpandedBegin = FrontMapping->BeginExpanded;
  } else {
    // Spelled tokens start after the mapping ends (they start in the hole
    // between 2 mappings, or between a mapping and end of the file).
    ExpandedBegin =
        FrontMapping->EndExpanded + (SpelledFrontI - FrontMapping->EndSpelled);
  }

  auto *BackMapping = mappingStartingBeforeSpelled(File, &Spelled.back());
  unsigned SpelledBackI = &Spelled.back() - File.SpelledTokens.data();
  unsigned ExpandedEnd;
  if (!BackMapping) {
    // No mapping that starts before the last token of Spelled, we don't have to
    // modify offsets.
    ExpandedEnd = File.BeginExpanded + SpelledBackI + 1;
  } else if (SpelledBackI < BackMapping->EndSpelled) {
    // This mapping applies to Spelled tokens.
    if (SpelledBackI + 1 != BackMapping->EndSpelled) {
      // Spelled tokens don't cover the entire mapping, returning empty result.
      return {}; // FIXME: support macro arguments.
    }
    ExpandedEnd = BackMapping->EndExpanded;
  } else {
    // Spelled tokens end after the mapping ends.
    ExpandedEnd =
        BackMapping->EndExpanded + (SpelledBackI - BackMapping->EndSpelled) + 1;
  }

  assert(ExpandedBegin < ExpandedTokens.size());
  assert(ExpandedEnd < ExpandedTokens.size());
  // Avoid returning empty ranges.
  if (ExpandedBegin == ExpandedEnd)
    return {};
  return {llvm::ArrayRef(ExpandedTokens.data() + ExpandedBegin,
                         ExpandedTokens.data() + ExpandedEnd)};
}

llvm::ArrayRef<syntax::Token> TokenBuffer::spelledTokens(FileID FID) const {
  auto It = Files.find(FID);
  assert(It != Files.end());
  return It->second.SpelledTokens;
}

const syntax::Token *
TokenBuffer::spelledTokenContaining(SourceLocation Loc) const {
  assert(Loc.isFileID());
  const auto *Tok = llvm::partition_point(
      spelledTokens(SourceMgr->getFileID(Loc)),
      [&](const syntax::Token &Tok) { return Tok.endLocation() <= Loc; });
  if (!Tok || Loc < Tok->location())
    return nullptr;
  return Tok;
}

std::string TokenBuffer::Mapping::str() const {
  return std::string(
      llvm::formatv("spelled tokens: [{0},{1}), expanded tokens: [{2},{3})",
                    BeginSpelled, EndSpelled, BeginExpanded, EndExpanded));
}

std::optional<llvm::ArrayRef<syntax::Token>>
TokenBuffer::spelledForExpanded(llvm::ArrayRef<syntax::Token> Expanded) const {
  // In cases of invalid code, AST nodes can have source ranges that include
  // the `eof` token. As there's no spelling for this token, exclude it from
  // the range.
  if (!Expanded.empty() && Expanded.back().kind() == tok::eof) {
    Expanded = Expanded.drop_back();
  }
  // Mapping an empty range is ambiguous in case of empty mappings at either end
  // of the range, bail out in that case.
  if (Expanded.empty())
    return std::nullopt;
  const syntax::Token *First = &Expanded.front();
  const syntax::Token *Last = &Expanded.back();
  auto [FirstSpelled, FirstMapping] = spelledForExpandedToken(First);
  auto [LastSpelled, LastMapping] = spelledForExpandedToken(Last);

  FileID FID = SourceMgr->getFileID(FirstSpelled->location());
  // FIXME: Handle multi-file changes by trying to map onto a common root.
  if (FID != SourceMgr->getFileID(LastSpelled->location()))
    return std::nullopt;

  const MarkedFile &File = Files.find(FID)->second;

  // If the range is within one macro argument, the result may be only part of a
  // Mapping. We must use the general (SourceManager-based) algorithm.
  // A token split from a token in a macro argument is mapped through the token
  // it was split from.
  if (FirstMapping && FirstMapping == LastMapping) {
    SourceLocation FirstLoc = getLocBeforeSplit(First->location(), *SourceMgr);
    SourceLocation LastLoc = getLocBeforeSplit(Last->location(), *SourceMgr);
    if (SourceMgr->isMacroArgExpansion(FirstLoc) &&
        SourceMgr->isMacroArgExpansion(LastLoc)) {
      // We use excluded Prev/Next token for bounds checking.
      SourceLocation Prev = (First == &ExpandedTokens.front())
                                ? SourceLocation()
                                : (First - 1)->location();
      SourceLocation Next = (Last == &ExpandedTokens.back())
                                ? SourceLocation()
                                : (Last + 1)->location();
      SourceRange Range = spelledForExpandedSlow(FirstLoc, LastLoc, Prev, Next,
                                                 FID, *SourceMgr);
      if (Range.isInvalid())
        return std::nullopt;
      return getTokensCovering(File.SpelledTokens, Range, *SourceMgr);
    }
  }

  // Otherwise, use the fast version based on Mappings.
  // Do not allow changes that doesn't cover full expansion.
  unsigned FirstExpanded = Expanded.begin() - ExpandedTokens.data();
  unsigned LastExpanded = Expanded.end() - ExpandedTokens.data();
  if (FirstMapping && FirstExpanded != FirstMapping->BeginExpanded)
    return std::nullopt;
  if (LastMapping && LastMapping->EndExpanded != LastExpanded)
    return std::nullopt;
  return llvm::ArrayRef(
      FirstMapping ? File.SpelledTokens.data() + FirstMapping->BeginSpelled
                   : FirstSpelled,
      LastMapping ? File.SpelledTokens.data() + LastMapping->EndSpelled
                  : LastSpelled + 1);
}

TokenBuffer::Expansion TokenBuffer::makeExpansion(const MarkedFile &F,
                                                  const Mapping &M) const {
  Expansion E;
  E.Spelled = llvm::ArrayRef(F.SpelledTokens.data() + M.BeginSpelled,
                             F.SpelledTokens.data() + M.EndSpelled);
  E.Expanded = llvm::ArrayRef(ExpandedTokens.data() + M.BeginExpanded,
                              ExpandedTokens.data() + M.EndExpanded);
  return E;
}

const TokenBuffer::MarkedFile &
TokenBuffer::fileForSpelled(llvm::ArrayRef<syntax::Token> Spelled) const {
  assert(!Spelled.empty());
  assert(Spelled.front().location().isFileID() && "not a spelled token");
  auto FileIt = Files.find(SourceMgr->getFileID(Spelled.front().location()));
  assert(FileIt != Files.end() && "file not tracked by token buffer");
  const auto &File = FileIt->second;
  assert(File.SpelledTokens.data() <= Spelled.data() &&
         Spelled.end() <=
             (File.SpelledTokens.data() + File.SpelledTokens.size()) &&
         "Tokens not in spelled range");
#ifndef NDEBUG
  auto T1 = Spelled.back().location();
  auto T2 = File.SpelledTokens.back().location();
  assert(T1 == T2 || sourceManager().isBeforeInTranslationUnit(T1, T2));
#endif
  return File;
}

std::optional<TokenBuffer::Expansion>
TokenBuffer::expansionStartingAt(const syntax::Token *Spelled) const {
  assert(Spelled);
  const auto &File = fileForSpelled(*Spelled);

  unsigned SpelledIndex = Spelled - File.SpelledTokens.data();
  auto M = llvm::partition_point(File.Mappings, [&](const Mapping &M) {
    return M.BeginSpelled < SpelledIndex;
  });
  if (M == File.Mappings.end() || M->BeginSpelled != SpelledIndex)
    return std::nullopt;
  return makeExpansion(File, *M);
}

std::vector<TokenBuffer::Expansion> TokenBuffer::expansionsOverlapping(
    llvm::ArrayRef<syntax::Token> Spelled) const {
  if (Spelled.empty())
    return {};
  const auto &File = fileForSpelled(Spelled);

  // Find the first overlapping range, and then copy until we stop overlapping.
  unsigned SpelledBeginIndex = Spelled.begin() - File.SpelledTokens.data();
  unsigned SpelledEndIndex = Spelled.end() - File.SpelledTokens.data();
  auto M = llvm::partition_point(File.Mappings, [&](const Mapping &M) {
    return M.EndSpelled <= SpelledBeginIndex;
  });
  std::vector<TokenBuffer::Expansion> Expansions;
  for (; M != File.Mappings.end() && M->BeginSpelled < SpelledEndIndex; ++M)
    Expansions.push_back(makeExpansion(File, *M));
  return Expansions;
}

llvm::ArrayRef<syntax::Token>
syntax::spelledTokensTouching(SourceLocation Loc,
                              llvm::ArrayRef<syntax::Token> Tokens) {
  assert(Loc.isFileID());

  auto *Right = llvm::partition_point(
      Tokens, [&](const syntax::Token &Tok) { return Tok.location() < Loc; });
  bool AcceptRight = Right != Tokens.end() && Right->location() <= Loc;
  bool AcceptLeft =
      Right != Tokens.begin() && (Right - 1)->endLocation() >= Loc;
  return llvm::ArrayRef(Right - (AcceptLeft ? 1 : 0),
                        Right + (AcceptRight ? 1 : 0));
}

llvm::ArrayRef<syntax::Token>
syntax::spelledTokensTouching(SourceLocation Loc,
                              const syntax::TokenBuffer &Tokens) {
  return spelledTokensTouching(
      Loc, Tokens.spelledTokens(Tokens.sourceManager().getFileID(Loc)));
}

const syntax::Token *
syntax::spelledIdentifierTouching(SourceLocation Loc,
                                  llvm::ArrayRef<syntax::Token> Tokens) {
  for (const syntax::Token &Tok : spelledTokensTouching(Loc, Tokens)) {
    if (Tok.kind() == tok::identifier)
      return &Tok;
  }
  return nullptr;
}

const syntax::Token *
syntax::spelledIdentifierTouching(SourceLocation Loc,
                                  const syntax::TokenBuffer &Tokens) {
  return spelledIdentifierTouching(
      Loc, Tokens.spelledTokens(Tokens.sourceManager().getFileID(Loc)));
}

std::vector<const syntax::Token *>
TokenBuffer::macroExpansions(FileID FID) const {
  auto FileIt = Files.find(FID);
  assert(FileIt != Files.end() && "file not tracked by token buffer");
  auto &File = FileIt->second;
  std::vector<const syntax::Token *> Expansions;
  auto &Spelled = File.SpelledTokens;
  for (auto Mapping : File.Mappings) {
    const syntax::Token *Token = &Spelled[Mapping.BeginSpelled];
    if (Token->kind() == tok::TokenKind::identifier)
      Expansions.push_back(Token);
  }
  return Expansions;
}

std::vector<syntax::Token> syntax::tokenize(const FileRange &FR,
                                            const SourceManager &SM,
                                            const LangOptions &LO) {
  std::vector<syntax::Token> Tokens;
  IdentifierTable Identifiers(LO);
  auto AddToken = [&](clang::Token T) {
    // Fill the proper token kind for keywords, etc.
    if (T.getKind() == tok::raw_identifier && !T.needsCleaning() &&
        !T.hasUCN()) { // FIXME: support needsCleaning and hasUCN cases.
      clang::IdentifierInfo &II = Identifiers.get(T.getRawIdentifier());
      T.setIdentifierInfo(&II);
      T.setKind(II.getTokenID());
    }
    Tokens.push_back(syntax::Token(T));
  };

  auto SrcBuffer = SM.getBufferData(FR.file());
  Lexer L(SM.getLocForStartOfFile(FR.file()), LO, SrcBuffer.data(),
          SrcBuffer.data() + FR.beginOffset(),
          // We can't make BufEnd point to FR.endOffset, as Lexer requires a
          // null terminated buffer.
          SrcBuffer.data() + SrcBuffer.size());

  clang::Token T;
  while (!L.LexFromRawLexer(T) && L.getCurrentBufferOffset() < FR.endOffset())
    AddToken(T);
  // LexFromRawLexer returns true when it parses the last token of the file, add
  // it iff it starts within the range we are interested in.
  if (SM.getFileOffset(T.getLocation()) < FR.endOffset())
    AddToken(T);
  return Tokens;
}

std::vector<syntax::Token> syntax::tokenize(FileID FID, const SourceManager &SM,
                                            const LangOptions &LO) {
  return tokenize(syntax::FileRange(FID, 0, SM.getFileIDSize(FID)), SM, LO);
}

/// Records information reqired to construct mappings for the token buffer that
/// we are collecting.
class TokenCollector::CollectPPExpansions : public PPCallbacks {
public:
  CollectPPExpansions(TokenCollector &C) : Collector(&C) {}

  /// Disabled instance will stop reporting anything to TokenCollector.
  /// This ensures that uses of the preprocessor after TokenCollector::consume()
  /// is called do not access the (possibly invalid) collector instance.
  void disable() { Collector = nullptr; }

  void MacroExpands(const clang::Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    if (!Collector)
      return;
    const auto &SM = Collector->PP.getSourceManager();
    // Only record top-level expansions that directly produce expanded tokens.
    // This excludes those where:
    //   - the macro use is inside a macro body,
    //   - the macro appears in an argument to another macro.
    // However macro expansion isn't really a tree, it's token rewrite rules,
    // so there are other cases, e.g.
    //   #define B(X) X
    //   #define A 1 + B
    //   A(2)
    // Both A and B produce expanded tokens, though the macro name 'B' comes
    // from an expansion. The best we can do is merge the mappings for both.

    // The *last* token of any top-level macro expansion must be in a file.
    // (In the example above, see the closing paren of the expansion of B).
    if (!Range.getEnd().isFileID())
      return;
    // If there's a current expansion that encloses this one, this one can't be
    // top-level.
    if (LastExpansionEnd.isValid() &&
        !SM.isBeforeInTranslationUnit(LastExpansionEnd, Range.getEnd()))
      return;

    // If the macro invocation (B) starts in a macro (A) but ends in a file,
    // we'll create a merged mapping for A + B by overwriting the endpoint for
    // A's startpoint.
    if (!Range.getBegin().isFileID()) {
      Range.setBegin(SM.getExpansionLoc(Range.getBegin()));
      assert(Collector->Expansions.count(Range.getBegin()) &&
             "Overlapping macros should have same expansion location");
    }

    Collector->Expansions[Range.getBegin()] = Range.getEnd();
    LastExpansionEnd = Range.getEnd();
  }
  // FIXME: handle directives like #pragma, #include, etc.
private:
  TokenCollector *Collector;
  /// Used to detect recursive macro expansions.
  SourceLocation LastExpansionEnd;
};

/// Fills in the TokenBuffer by tracing the run of a preprocessor. The
/// implementation tracks the tokens, macro expansions and directives coming
/// from the preprocessor and:
/// - for each token, figures out if it is a part of an expanded token stream,
///   spelled token stream or both. Stores the tokens appropriately.
/// - records mappings from the spelled to expanded token ranges, e.g. for macro
///   expansions.
/// FIXME: also properly record:
///          - #include directives,
///          - #pragma, #line and other PP directives,
///          - skipped pp regions,
///          - ...

TokenCollector::TokenCollector(Preprocessor &PP) : PP(PP) {
  // Collect the expanded token stream during preprocessing.
  PP.setTokenWatcher([this](const clang::Token &T) {
    if (T.is(tok::annot_module_name)) {
      auto &SM = this->PP.getSourceManager();
      StringRef Text = Lexer::getSourceText(
          CharSourceRange::getTokenRange(T.getAnnotationRange()), SM,
          this->PP.getLangOpts());
      Expanded.push_back(
          syntax::Token(T.getLocation(), Text.size(), tok::annot_module_name));
      return;
    }

    // These tokens do not have a one-to-one raw spelling.
    if (T.isAnnotation() || T.is(tok::eod))
      return;

    Expanded.push_back(syntax::Token(T));
    DEBUG_WITH_TYPE("collect-tokens", llvm::dbgs()
                                          << "Token: "
                                          << syntax::Token(T).dumpForTests(
                                                 this->PP.getSourceManager())
                                          << "\n"

    );
  });
  // The token watcher sees a token before the parser splits it, so record the
  // splits too.
  PP.setTokenSplitWatcher([this](SourceLocation TokLoc, unsigned Length,
                                 tok::TokenKind SplitKind,
                                 SourceLocation SplitLoc) {
    Splits.push_back({TokLoc, Length, SplitLoc, SplitKind});
  });
  // And locations of macro calls, to properly recover boundaries of those in
  // case of empty expansions.
  auto CB = std::make_unique<CollectPPExpansions>(*this);
  this->Collector = CB.get();
  PP.addPPCallbacks(std::move(CB));
}

/// Builds mappings and spelled tokens in the TokenBuffer based on the expanded
/// token stream.
class TokenCollector::Builder {
public:
  Builder(std::vector<syntax::Token> Expanded, std::vector<TokenSplit> Splits,
          PPExpansions CollectedExpansions, const SourceManager &SM,
          const LangOptions &LangOpts)
      : Result(SM), Splits(std::move(Splits)),
        CollectedExpansions(std::move(CollectedExpansions)), SM(SM),
        LangOpts(LangOpts) {
    Result.ExpandedTokens = std::move(Expanded);
  }

  TokenBuffer build() && {
    assert(!Result.ExpandedTokens.empty());

    // When the parser hits a hard limit (e.g. bracket depth or function scope
    // depth), it halts prematurely and leaves the expanded token stream
    // truncated with no final `eof` token. To keep the invariant, synthesize an
    // `eof` at the location of the last collected token.
    if (Result.ExpandedTokens.back().kind() != tok::eof) {
      SourceLocation Loc = Result.ExpandedTokens.back().location();
      Result.ExpandedTokens.emplace_back(Loc, 0, tok::eof);
    }

    // Replace the tokens that the parser split with the split tokens.
    applySplits();

    // Tokenize every file that contributed tokens to the expanded stream.
    buildSpelledTokens();

    // The expanded token stream consists of runs of tokens that came from
    // the same source (a macro expansion, part of a file etc).
    // Between these runs are the logical positions of spelled tokens that
    // didn't expand to anything.
    while (NextExpanded < Result.ExpandedTokens.size() - 1 /* eof */) {
      // Create empty mappings for spelled tokens that expanded to nothing here.
      // May advance NextSpelled, but NextExpanded is unchanged.
      discard();
      // Create mapping for a contiguous run of expanded tokens.
      // Advances NextExpanded past the run, and NextSpelled accordingly.
      unsigned OldPosition = NextExpanded;
      advance();
      if (NextExpanded == OldPosition)
        diagnoseAdvanceFailure();
    }
    // If any tokens remain in any of the files, they didn't expand to anything.
    // Create empty mappings up until the end of the file.
    for (const auto &File : Result.Files)
      discard(File.first);

#ifndef NDEBUG
    for (auto &pair : Result.Files) {
      auto &mappings = pair.second.Mappings;
      assert(llvm::is_sorted(mappings, [](const TokenBuffer::Mapping &M1,
                                          const TokenBuffer::Mapping &M2) {
        return M1.BeginSpelled < M2.BeginSpelled &&
               M1.EndSpelled < M2.EndSpelled &&
               M1.BeginExpanded < M2.BeginExpanded &&
               M1.EndExpanded < M2.EndExpanded;
      }));
    }
#endif

    return std::move(Result);
  }

private:
  // A token split from an expanded token. Offset is the offset of its first
  // character within the expanded token.
  struct Piece {
    syntax::Token Tok;
    unsigned Offset;
  };

  // Returns the pieces that Old becomes when the parser makes the split S, or
  // std::nullopt if S does not apply to Old.
  std::optional<llvm::SmallVector<Piece, 2>>
  splitPiece(const Piece &Old, const TokenSplit &S) const {
    if (S.Length > Old.Tok.length())
      return std::nullopt;
    // The parser includes trailing escaped newlines which we don't want.
    Piece Split{syntax::Token(S.SplitLoc,
                              lengthWithoutTrailingEscapedNewlines(
                                  S.SplitLoc, S.Length, SM, LangOpts),
                              S.SplitKind),
                Old.Offset};
    if (S.Length == Old.Tok.length()) {
      // The parser split the remainder of the original token, which happens if
      // it needs to prevent the subsequent token from being merged with the
      // remainder, e.g. the middle `>` in `A<B<C<I>>>`.
      return llvm::SmallVector<Piece, 2>{Split};
    }
    auto RestKind = getRemainderKind(Old.Tok.kind(), S.SplitKind);
    if (!RestKind)
      return std::nullopt;
    Piece Rest{syntax::Token(S.TokLoc.getLocWithOffset(S.Length),
                             Old.Tok.length() - S.Length, *RestKind),
               Old.Offset + S.Length};
    return llvm::SmallVector<Piece, 2>{Split, Rest};
  }

  // Applies Splits in order. Returns the pieces of each expanded token that
  // the parser split, by its index into ExpandedTokens.
  llvm::DenseMap<unsigned, llvm::SmallVector<Piece, 2>> computePieces() const {
    const auto &Expanded = Result.ExpandedTokens;
    // The index into Expanded of the token at the location of each split, or
    // nullopt if there is none.
    llvm::DenseMap<SourceLocation, std::optional<unsigned>> TokenIndexByLoc;
    for (const TokenSplit &S : Splits)
      TokenIndexByLoc.try_emplace(S.TokLoc);
    for (unsigned I = 0; I < Expanded.size(); ++I) {
      auto It = TokenIndexByLoc.find(Expanded[I].location());
      if (It != TokenIndexByLoc.end() && !It->second.has_value())
        It->second = I;
    }
    // The pieces of each expanded token that was split, by index into Expanded.
    llvm::DenseMap<unsigned, llvm::SmallVector<Piece, 2>> PiecesByToken;
    // Maps the location of each piece to its index into Expanded and index
    // among that token's pieces.
    llvm::DenseMap<SourceLocation, std::pair<unsigned, unsigned>> PieceByLoc;

    for (const TokenSplit &S : Splits) {
      unsigned TokenIndex, PieceIndex;
      bool StartFromWholeToken = false;
      if (auto It = PieceByLoc.find(S.TokLoc); It != PieceByLoc.end()) {
        std::tie(TokenIndex, PieceIndex) = It->second;
      } else if (auto It = TokenIndexByLoc.find(S.TokLoc);
                 It != TokenIndexByLoc.end() && It->second.has_value()) {
        TokenIndex = *It->second;
        PieceIndex = 0;
        StartFromWholeToken = true;
      } else {
        // The parser split a token that is not in the expanded stream.
        continue;
      }

      auto &TokenPieces = PiecesByToken[TokenIndex];
      if (StartFromWholeToken) {
        // The parser can split a token again after backtracking, so throw out
        // the previous split (if any) and start again.
        for (const Piece &Stale : TokenPieces)
          PieceByLoc.erase(Stale.Tok.location());
        TokenPieces.assign(1, {Expanded[TokenIndex], 0});
      }

      auto Replacement = splitPiece(TokenPieces[PieceIndex], S);
      if (!Replacement)
        continue;
      TokenPieces.erase(TokenPieces.begin() + PieceIndex);
      TokenPieces.insert(TokenPieces.begin() + PieceIndex, Replacement->begin(),
                         Replacement->end());
      PieceByLoc.erase(S.TokLoc);
      for (unsigned I = 0; I < TokenPieces.size(); ++I)
        PieceByLoc[TokenPieces[I].Tok.location()] = {TokenIndex, I};
    }
    return PiecesByToken;
  }

  // Records the spelled tokens that must be split to match the pieces of the
  // expanded token Orig, in SpelledSplits.
  void recordSpelledSplit(const syntax::Token &Orig,
                          llvm::ArrayRef<Piece> TokenPieces) {
    // Find the spelled token: the expanded token itself, or the token in a
    // macro argument that it was expanded from. The spelled token of a token
    // from a macro body is in the #define, which is not split.
    SourceLocation Spelled = Orig.location();
    while (Spelled.isMacroID() && SM.isMacroArgExpansion(Spelled))
      Spelled = SM.getImmediateSpellingLoc(Spelled);
    if (!Spelled.isFileID())
      return;
    llvm::SmallVector<syntax::Token, 2> SpelledPieces;
    for (const Piece &P : TokenPieces)
      SpelledPieces.emplace_back(Spelled.getLocWithOffset(P.Offset),
                                 P.Tok.length(), P.Tok.kind());
    // A token in a macro argument can be expanded more than once. Keep the
    // split of its first expansion.
    if (SpelledSplits.try_emplace(Spelled, std::move(SpelledPieces)).second)
      FilesWithSpelledSplits.insert(SM.getFileID(Spelled));
  }

  // Replaces each expanded token that the parser split with the tokens it was
  // split into, and records the spelled tokens that must be split to match.
  void applySplits() {
    if (Splits.empty())
      return;
    auto PiecesByToken = computePieces();
    if (PiecesByToken.empty())
      return;

    auto &Expanded = Result.ExpandedTokens;
    size_t Added = 0;
    for (const auto &[TokenIndex, TokenPieces] : PiecesByToken)
      Added += TokenPieces.size() -
               1; // A token split into N pieces adds N - 1 tokens.
    std::vector<syntax::Token> NewExpanded;
    NewExpanded.reserve(Expanded.size() + Added);
    for (unsigned TokenIndex = 0; TokenIndex < Expanded.size(); ++TokenIndex) {
      auto It = PiecesByToken.find(TokenIndex);
      if (It == PiecesByToken.end()) {
        NewExpanded.push_back(Expanded[TokenIndex]);
        continue;
      }
      for (const Piece &P : It->second)
        NewExpanded.push_back(P.Tok);
      recordSpelledSplit(Expanded[TokenIndex], It->second);
    }
    Expanded = std::move(NewExpanded);
  }

  // Returns the location of an expanded token in the file it was spelled in,
  // or an invalid location if it is part of a macro expansion.
  SourceLocation fileLocation(const syntax::Token &T) const {
    // A token split from a file token is spelled where its split started.
    SourceLocation Loc = getLocBeforeSplit(T.location(), SM);
    return Loc.isFileID() ? Loc : SourceLocation();
  }

  // Replaces the spelled tokens that were split by the parser with the tokens
  // they were split into.
  void splitSpelledTokens(FileID FID,
                          std::vector<syntax::Token> &Spelled) const {
    if (!FilesWithSpelledSplits.contains(FID))
      return;
    std::vector<syntax::Token> Out;
    Out.reserve(Spelled.size());
    for (const syntax::Token &T : Spelled) {
      auto It = SpelledSplits.find(T.location());
      if (It == SpelledSplits.end()) {
        Out.push_back(T);
        continue;
      }
      assert(It->second.back().endLocation() == T.endLocation() &&
             "split tokens do not cover the spelled token");
      llvm::append_range(Out, It->second);
    }
    Spelled = std::move(Out);
  }

  // Consume a sequence of spelled tokens that didn't expand to anything.
  // In the simplest case, skips spelled tokens until finding one that produced
  // the NextExpanded token, and creates an empty mapping for them.
  // If Drain is provided, skips remaining tokens from that file instead.
  void discard(std::optional<FileID> Drain = std::nullopt) {
    SourceLocation Target =
        Drain ? SM.getLocForEndOfFile(*Drain)
              : SM.getExpansionLoc(
                    Result.ExpandedTokens[NextExpanded].location());
    FileID File = SM.getFileID(Target);
    const auto &SpelledTokens = Result.Files[File].SpelledTokens;
    auto &NextSpelled = this->NextSpelled[File];

    TokenBuffer::Mapping Mapping;
    Mapping.BeginSpelled = NextSpelled;
    // When dropping trailing tokens from a file, the empty mapping should
    // be positioned within the file's expanded-token range (at the end).
    Mapping.BeginExpanded = Mapping.EndExpanded =
        Drain ? Result.Files[*Drain].EndExpanded : NextExpanded;
    // We may want to split into several adjacent empty mappings.
    // FlushMapping() emits the current mapping and starts a new one.
    auto FlushMapping = [&, this] {
      Mapping.EndSpelled = NextSpelled;
      if (Mapping.BeginSpelled != Mapping.EndSpelled)
        Result.Files[File].Mappings.push_back(Mapping);
      Mapping.BeginSpelled = NextSpelled;
    };

    while (NextSpelled < SpelledTokens.size() &&
           SpelledTokens[NextSpelled].location() < Target) {
      // If we know mapping bounds at [NextSpelled, KnownEnd] (macro expansion)
      // then we want to partition our (empty) mapping.
      //   [Start, NextSpelled) [NextSpelled, KnownEnd] (KnownEnd, Target)
      SourceLocation KnownEnd =
          CollectedExpansions.lookup(SpelledTokens[NextSpelled].location());
      if (KnownEnd.isValid()) {
        FlushMapping(); // Emits [Start, NextSpelled)
        while (NextSpelled < SpelledTokens.size() &&
               SpelledTokens[NextSpelled].location() <= KnownEnd)
          ++NextSpelled;
        FlushMapping(); // Emits [NextSpelled, KnownEnd]
        // Now the loop continues and will emit (KnownEnd, Target).
      } else {
        ++NextSpelled;
      }
    }
    FlushMapping();
  }

  // Consumes the NextExpanded token and others that are part of the same run.
  // Increases NextExpanded and NextSpelled by at least one, and adds a mapping
  // (unless this is a run of file tokens, which we represent with no mapping).
  void advance() {
    const syntax::Token &Tok = Result.ExpandedTokens[NextExpanded];
    SourceLocation Expansion = SM.getExpansionLoc(Tok.location());
    FileID File = SM.getFileID(Expansion);
    const auto &SpelledTokens = Result.Files[File].SpelledTokens;
    auto &NextSpelled = this->NextSpelled[File];

    if (fileLocation(Tok).isValid()) {
      // A run of file tokens continues while the expanded/spelled tokens match.
      while (NextSpelled < SpelledTokens.size() &&
             NextExpanded < Result.ExpandedTokens.size() &&
             SpelledTokens[NextSpelled].location() ==
                 fileLocation(Result.ExpandedTokens[NextExpanded])) {
        ++NextSpelled;
        ++NextExpanded;
      }
      // We need no mapping for file tokens copied to the expanded stream.
    } else {
      // We found a new macro expansion. We should have its spelling bounds.
      auto End = CollectedExpansions.lookup(Expansion);
      assert(End.isValid() && "Macro expansion wasn't captured?");

      // Mapping starts here...
      TokenBuffer::Mapping Mapping;
      Mapping.BeginExpanded = NextExpanded;
      Mapping.BeginSpelled = NextSpelled;
      // ... consumes spelled tokens within bounds we captured ...
      while (NextSpelled < SpelledTokens.size() &&
             SpelledTokens[NextSpelled].location() <= End)
        ++NextSpelled;
      // ... consumes expanded tokens rooted at the same expansion ...
      while (NextExpanded < Result.ExpandedTokens.size() &&
             SM.getExpansionLoc(
                 Result.ExpandedTokens[NextExpanded].location()) == Expansion)
        ++NextExpanded;
      // ... and ends here.
      Mapping.EndExpanded = NextExpanded;
      Mapping.EndSpelled = NextSpelled;
      Result.Files[File].Mappings.push_back(Mapping);
    }
  }

  // advance() is supposed to consume at least one token - if not, we crash.
  void diagnoseAdvanceFailure() {
#ifndef NDEBUG
    // Show the failed-to-map token in context.
    for (unsigned I = (NextExpanded < 10) ? 0 : NextExpanded - 10;
         I < NextExpanded + 5 && I < Result.ExpandedTokens.size(); ++I) {
      const char *L =
          (I == NextExpanded) ? "!! " : (I < NextExpanded) ? "ok " : "   ";
      llvm::errs() << L << Result.ExpandedTokens[I].dumpForTests(SM) << "\n";
    }
#endif
    llvm_unreachable("Couldn't map expanded token to spelled tokens!");
  }

  /// Initializes TokenBuffer::Files and fills spelled tokens and expanded
  /// ranges for each of the files.
  void buildSpelledTokens() {
    for (unsigned I = 0; I < Result.ExpandedTokens.size(); ++I) {
      const auto &Tok = Result.ExpandedTokens[I];
      auto FID = SM.getFileID(SM.getExpansionLoc(Tok.location()));
      auto It = Result.Files.try_emplace(FID);
      TokenBuffer::MarkedFile &File = It.first->second;

      // The eof token should not be considered part of the main-file's range.
      File.EndExpanded = Tok.kind() == tok::eof ? I : I + 1;

      if (!It.second)
        continue; // we have seen this file before.
      // This is the first time we see this file.
      File.BeginExpanded = I;
      File.SpelledTokens = tokenize(FID, SM, LangOpts);
      splitSpelledTokens(FID, File.SpelledTokens);
    }
  }

  TokenBuffer Result;
  unsigned NextExpanded = 0;                    // cursor in ExpandedTokens
  llvm::DenseMap<FileID, unsigned> NextSpelled; // cursor in SpelledTokens
  std::vector<TokenSplit> Splits;
  // The tokens to replace each split spelled token with, by its location.
  llvm::DenseMap<SourceLocation, llvm::SmallVector<syntax::Token, 2>>
      SpelledSplits;
  // The files that contain a key of SpelledSplits.
  llvm::DenseSet<FileID> FilesWithSpelledSplits;
  PPExpansions CollectedExpansions;
  const SourceManager &SM;
  const LangOptions &LangOpts;
};

TokenBuffer TokenCollector::consume() && {
  PP.setTokenWatcher(nullptr);
  PP.setTokenSplitWatcher(nullptr);
  Collector->disable();
  return Builder(std::move(Expanded), std::move(Splits), std::move(Expansions),
                 PP.getSourceManager(), PP.getLangOpts())
      .build();
}

std::string syntax::Token::str() const {
  return std::string(llvm::formatv("Token({0}, length = {1})",
                                   tok::getTokenName(kind()), length()));
}

std::string syntax::Token::dumpForTests(const SourceManager &SM) const {
  return std::string(llvm::formatv("Token(`{0}`, {1}, length = {2})", text(SM),
                                   tok::getTokenName(kind()), length()));
}

std::string TokenBuffer::dumpForTests() const {
  auto PrintToken = [this](const syntax::Token &T) -> std::string {
    if (T.kind() == tok::eof)
      return "<eof>";
    return std::string(T.text(*SourceMgr));
  };

  auto DumpTokens = [this, &PrintToken](llvm::raw_ostream &OS,
                                        llvm::ArrayRef<syntax::Token> Tokens) {
    if (Tokens.empty()) {
      OS << "<empty>";
      return;
    }
    OS << Tokens[0].text(*SourceMgr);
    for (unsigned I = 1; I < Tokens.size(); ++I) {
      if (Tokens[I].kind() == tok::eof)
        continue;
      OS << " " << PrintToken(Tokens[I]);
    }
  };

  std::string Dump;
  llvm::raw_string_ostream OS(Dump);

  OS << "expanded tokens:\n"
     << "  ";
  // (!) we do not show '<eof>'.
  DumpTokens(OS, llvm::ArrayRef(ExpandedTokens).drop_back());
  OS << "\n";

  std::vector<FileID> Keys;
  for (const auto &F : Files)
    Keys.push_back(F.first);
  llvm::sort(Keys);

  for (FileID ID : Keys) {
    const MarkedFile &File = Files.find(ID)->second;
    auto Entry = SourceMgr->getFileEntryRefForID(ID);
    if (!Entry)
      continue; // Skip builtin files.
    std::string Path = llvm::sys::path::convert_to_slash(Entry->getName());
    OS << llvm::formatv("file '{0}'\n", Path) << "  spelled tokens:\n"
       << "    ";
    DumpTokens(OS, File.SpelledTokens);
    OS << "\n";

    if (File.Mappings.empty()) {
      OS << "  no mappings.\n";
      continue;
    }
    OS << "  mappings:\n";
    for (auto &M : File.Mappings) {
      OS << llvm::formatv(
          "    ['{0}'_{1}, '{2}'_{3}) => ['{4}'_{5}, '{6}'_{7})\n",
          PrintToken(File.SpelledTokens[M.BeginSpelled]), M.BeginSpelled,
          M.EndSpelled == File.SpelledTokens.size()
              ? "<eof>"
              : PrintToken(File.SpelledTokens[M.EndSpelled]),
          M.EndSpelled, PrintToken(ExpandedTokens[M.BeginExpanded]),
          M.BeginExpanded, PrintToken(ExpandedTokens[M.EndExpanded]),
          M.EndExpanded);
    }
  }
  return Dump;
}
