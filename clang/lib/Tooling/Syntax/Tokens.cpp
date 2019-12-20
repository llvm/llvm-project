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
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace clang::syntax;

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
  assert((F == L || F.endOffset() <= L.beginOffset()) && "wrong order of tokens");
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

llvm::ArrayRef<syntax::Token> TokenBuffer::expandedTokens(SourceRange R) const {
  if (R.isInvalid())
    return {};
  const Token *Begin =
      llvm::partition_point(expandedTokens(), [&](const syntax::Token &T) {
        return SourceMgr->isBeforeInTranslationUnit(T.location(), R.getBegin());
      });
  const Token *End =
      llvm::partition_point(expandedTokens(), [&](const syntax::Token &T) {
        return !SourceMgr->isBeforeInTranslationUnit(R.getEnd(), T.location());
      });
  if (Begin > End)
    return {};
  return {Begin, End};
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
    return {&File.SpelledTokens[ExpandedIndex - File.BeginExpanded], nullptr};
  }
  --It; // 'It' now points to last mapping that started before our token.

  // Check if the token is part of the mapping.
  if (ExpandedIndex < It->EndExpanded)
    return {&File.SpelledTokens[It->BeginSpelled], /*Mapping*/ &*It};

  // Not part of the mapping, use the index from previous mapping to compute the
  // corresponding spelled token.
  return {
      &File.SpelledTokens[It->EndSpelled + (ExpandedIndex - It->EndExpanded)],
      /*Mapping*/ nullptr};
}

llvm::ArrayRef<syntax::Token> TokenBuffer::spelledTokens(FileID FID) const {
  auto It = Files.find(FID);
  assert(It != Files.end());
  return It->second.SpelledTokens;
}

std::string TokenBuffer::Mapping::str() const {
  return llvm::formatv("spelled tokens: [{0},{1}), expanded tokens: [{2},{3})",
                       BeginSpelled, EndSpelled, BeginExpanded, EndExpanded);
}

llvm::Optional<llvm::ArrayRef<syntax::Token>>
TokenBuffer::spelledForExpanded(llvm::ArrayRef<syntax::Token> Expanded) const {
  // Mapping an empty range is ambiguous in case of empty mappings at either end
  // of the range, bail out in that case.
  if (Expanded.empty())
    return llvm::None;

  // FIXME: also allow changes uniquely mapping to macro arguments.

  const syntax::Token *BeginSpelled;
  const Mapping *BeginMapping;
  std::tie(BeginSpelled, BeginMapping) =
      spelledForExpandedToken(&Expanded.front());

  const syntax::Token *LastSpelled;
  const Mapping *LastMapping;
  std::tie(LastSpelled, LastMapping) =
      spelledForExpandedToken(&Expanded.back());

  FileID FID = SourceMgr->getFileID(BeginSpelled->location());
  // FIXME: Handle multi-file changes by trying to map onto a common root.
  if (FID != SourceMgr->getFileID(LastSpelled->location()))
    return llvm::None;

  const MarkedFile &File = Files.find(FID)->second;

  // Do not allow changes that cross macro expansion boundaries.
  unsigned BeginExpanded = Expanded.begin() - ExpandedTokens.data();
  unsigned EndExpanded = Expanded.end() - ExpandedTokens.data();
  if (BeginMapping && BeginMapping->BeginExpanded < BeginExpanded)
    return llvm::None;
  if (LastMapping && EndExpanded < LastMapping->EndExpanded)
    return llvm::None;
  // All is good, return the result.
  return llvm::makeArrayRef(
      BeginMapping ? File.SpelledTokens.data() + BeginMapping->BeginSpelled
                   : BeginSpelled,
      LastMapping ? File.SpelledTokens.data() + LastMapping->EndSpelled
                  : LastSpelled + 1);
}

llvm::Optional<TokenBuffer::Expansion>
TokenBuffer::expansionStartingAt(const syntax::Token *Spelled) const {
  assert(Spelled);
  assert(Spelled->location().isFileID() && "not a spelled token");
  auto FileIt = Files.find(SourceMgr->getFileID(Spelled->location()));
  assert(FileIt != Files.end() && "file not tracked by token buffer");

  auto &File = FileIt->second;
  assert(File.SpelledTokens.data() <= Spelled &&
         Spelled < (File.SpelledTokens.data() + File.SpelledTokens.size()));

  unsigned SpelledIndex = Spelled - File.SpelledTokens.data();
  auto M = llvm::partition_point(File.Mappings, [&](const Mapping &M) {
    return M.BeginSpelled < SpelledIndex;
  });
  if (M == File.Mappings.end() || M->BeginSpelled != SpelledIndex)
    return llvm::None;

  Expansion E;
  E.Spelled = llvm::makeArrayRef(File.SpelledTokens.data() + M->BeginSpelled,
                                 File.SpelledTokens.data() + M->EndSpelled);
  E.Expanded = llvm::makeArrayRef(ExpandedTokens.data() + M->BeginExpanded,
                                  ExpandedTokens.data() + M->EndExpanded);
  return E;
}

llvm::ArrayRef<syntax::Token>
syntax::spelledTokensTouching(SourceLocation Loc,
                              const syntax::TokenBuffer &Tokens) {
  assert(Loc.isFileID());
  llvm::ArrayRef<syntax::Token> All =
      Tokens.spelledTokens(Tokens.sourceManager().getFileID(Loc));
  auto *Right = llvm::partition_point(
      All, [&](const syntax::Token &Tok) { return Tok.location() < Loc; });
  bool AcceptRight = Right != All.end() && Right->location() <= Loc;
  bool AcceptLeft = Right != All.begin() && (Right - 1)->endLocation() >= Loc;
  return llvm::makeArrayRef(Right - (AcceptLeft ? 1 : 0),
                            Right + (AcceptRight ? 1 : 0));
}

const syntax::Token *
syntax::spelledIdentifierTouching(SourceLocation Loc,
                                  const syntax::TokenBuffer &Tokens) {
  for (const syntax::Token &Tok : spelledTokensTouching(Loc, Tokens)) {
    if (Tok.kind() == tok::identifier)
      return &Tok;
  }
  return nullptr;
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

std::vector<syntax::Token> syntax::tokenize(FileID FID, const SourceManager &SM,
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

  Lexer L(FID, SM.getBuffer(FID), SM, LO);

  clang::Token T;
  while (!L.LexFromRawLexer(T))
    AddToken(T);
  // 'eof' is only the last token if the input is null-terminated. Never store
  // it, for consistency.
  if (T.getKind() != tok::eof)
    AddToken(T);
  return Tokens;
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
    // Only record top-level expansions, not those where:
    //   - the macro use is inside a macro body,
    //   - the macro appears in an argument to another macro.
    if (!MacroNameTok.getLocation().isFileID() ||
        (LastExpansionEnd.isValid() &&
         Collector->PP.getSourceManager().isBeforeInTranslationUnit(
             Range.getBegin(), LastExpansionEnd)))
      return;
    Collector->Expansions[Range.getBegin().getRawEncoding()] = Range.getEnd();
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
    if (T.isAnnotation())
      return;
    DEBUG_WITH_TYPE("collect-tokens", llvm::dbgs()
                                          << "Token: "
                                          << syntax::Token(T).dumpForTests(
                                                 this->PP.getSourceManager())
                                          << "\n"

    );
    Expanded.push_back(syntax::Token(T));
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
  Builder(std::vector<syntax::Token> Expanded, PPExpansions CollectedExpansions,
          const SourceManager &SM, const LangOptions &LangOpts)
      : Result(SM), CollectedExpansions(std::move(CollectedExpansions)), SM(SM),
        LangOpts(LangOpts) {
    Result.ExpandedTokens = std::move(Expanded);
  }

  TokenBuffer build() && {
    buildSpelledTokens();

    // Walk over expanded tokens and spelled tokens in parallel, building the
    // mappings between those using source locations.
    // To correctly recover empty macro expansions, we also take locations
    // reported to PPCallbacks::MacroExpands into account as we do not have any
    // expanded tokens with source locations to guide us.

    // The 'eof' token is special, it is not part of spelled token stream. We
    // handle it separately at the end.
    assert(!Result.ExpandedTokens.empty());
    assert(Result.ExpandedTokens.back().kind() == tok::eof);
    for (unsigned I = 0; I < Result.ExpandedTokens.size() - 1; ++I) {
      // (!) I might be updated by the following call.
      processExpandedToken(I);
    }

    // 'eof' not handled in the loop, do it here.
    assert(SM.getMainFileID() ==
           SM.getFileID(Result.ExpandedTokens.back().location()));
    fillGapUntil(Result.Files[SM.getMainFileID()],
                 Result.ExpandedTokens.back().location(),
                 Result.ExpandedTokens.size() - 1);
    Result.Files[SM.getMainFileID()].EndExpanded = Result.ExpandedTokens.size();

    // Some files might have unaccounted spelled tokens at the end, add an empty
    // mapping for those as they did not have expanded counterparts.
    fillGapsAtEndOfFiles();

    return std::move(Result);
  }

private:
  /// Process the next token in an expanded stream and move corresponding
  /// spelled tokens, record any mapping if needed.
  /// (!) \p I will be updated if this had to skip tokens, e.g. for macros.
  void processExpandedToken(unsigned &I) {
    auto L = Result.ExpandedTokens[I].location();
    if (L.isMacroID()) {
      processMacroExpansion(SM.getExpansionRange(L), I);
      return;
    }
    if (L.isFileID()) {
      auto FID = SM.getFileID(L);
      TokenBuffer::MarkedFile &File = Result.Files[FID];

      fillGapUntil(File, L, I);

      // Skip the token.
      assert(File.SpelledTokens[NextSpelled[FID]].location() == L &&
             "no corresponding token in the spelled stream");
      ++NextSpelled[FID];
      return;
    }
  }

  /// Skipped expanded and spelled tokens of a macro expansion that covers \p
  /// SpelledRange. Add a corresponding mapping.
  /// (!) \p I will be the index of the last token in an expansion after this
  /// function returns.
  void processMacroExpansion(CharSourceRange SpelledRange, unsigned &I) {
    auto FID = SM.getFileID(SpelledRange.getBegin());
    assert(FID == SM.getFileID(SpelledRange.getEnd()));
    TokenBuffer::MarkedFile &File = Result.Files[FID];

    fillGapUntil(File, SpelledRange.getBegin(), I);

    // Skip all expanded tokens from the same macro expansion.
    unsigned BeginExpanded = I;
    for (; I + 1 < Result.ExpandedTokens.size(); ++I) {
      auto NextL = Result.ExpandedTokens[I + 1].location();
      if (!NextL.isMacroID() ||
          SM.getExpansionLoc(NextL) != SpelledRange.getBegin())
        break;
    }
    unsigned EndExpanded = I + 1;
    consumeMapping(File, SM.getFileOffset(SpelledRange.getEnd()), BeginExpanded,
                   EndExpanded, NextSpelled[FID]);
  }

  /// Initializes TokenBuffer::Files and fills spelled tokens and expanded
  /// ranges for each of the files.
  void buildSpelledTokens() {
    for (unsigned I = 0; I < Result.ExpandedTokens.size(); ++I) {
      auto FID =
          SM.getFileID(SM.getExpansionLoc(Result.ExpandedTokens[I].location()));
      auto It = Result.Files.try_emplace(FID);
      TokenBuffer::MarkedFile &File = It.first->second;

      File.EndExpanded = I + 1;
      if (!It.second)
        continue; // we have seen this file before.

      // This is the first time we see this file.
      File.BeginExpanded = I;
      File.SpelledTokens = tokenize(FID, SM, LangOpts);
    }
  }

  void consumeEmptyMapping(TokenBuffer::MarkedFile &File, unsigned EndOffset,
                           unsigned ExpandedIndex, unsigned &SpelledIndex) {
    consumeMapping(File, EndOffset, ExpandedIndex, ExpandedIndex, SpelledIndex);
  }

  /// Consumes spelled tokens that form a macro expansion and adds a entry to
  /// the resulting token buffer.
  /// (!) SpelledIndex is updated in-place.
  void consumeMapping(TokenBuffer::MarkedFile &File, unsigned EndOffset,
                      unsigned BeginExpanded, unsigned EndExpanded,
                      unsigned &SpelledIndex) {
    // We need to record this mapping before continuing.
    unsigned MappingBegin = SpelledIndex;
    ++SpelledIndex;

    bool HitMapping =
        tryConsumeSpelledUntil(File, EndOffset + 1, SpelledIndex).hasValue();
    (void)HitMapping;
    assert(!HitMapping && "recursive macro expansion?");

    TokenBuffer::Mapping M;
    M.BeginExpanded = BeginExpanded;
    M.EndExpanded = EndExpanded;
    M.BeginSpelled = MappingBegin;
    M.EndSpelled = SpelledIndex;

    File.Mappings.push_back(M);
  }

  /// Consumes spelled tokens until location \p L is reached and adds a mapping
  /// covering the consumed tokens. The mapping will point to an empty expanded
  /// range at position \p ExpandedIndex.
  void fillGapUntil(TokenBuffer::MarkedFile &File, SourceLocation L,
                    unsigned ExpandedIndex) {
    assert(L.isFileID());
    FileID FID;
    unsigned Offset;
    std::tie(FID, Offset) = SM.getDecomposedLoc(L);

    unsigned &SpelledIndex = NextSpelled[FID];
    unsigned MappingBegin = SpelledIndex;
    while (true) {
      auto EndLoc = tryConsumeSpelledUntil(File, Offset, SpelledIndex);
      if (SpelledIndex != MappingBegin) {
        TokenBuffer::Mapping M;
        M.BeginSpelled = MappingBegin;
        M.EndSpelled = SpelledIndex;
        M.BeginExpanded = M.EndExpanded = ExpandedIndex;
        File.Mappings.push_back(M);
      }
      if (!EndLoc)
        break;
      consumeEmptyMapping(File, SM.getFileOffset(*EndLoc), ExpandedIndex,
                          SpelledIndex);

      MappingBegin = SpelledIndex;
    }
  };

  /// Consumes spelled tokens until it reaches Offset or a mapping boundary,
  /// i.e. a name of a macro expansion or the start '#' token of a PP directive.
  /// (!) NextSpelled is updated in place.
  ///
  /// returns None if \p Offset was reached, otherwise returns the end location
  /// of a mapping that starts at \p NextSpelled.
  llvm::Optional<SourceLocation>
  tryConsumeSpelledUntil(TokenBuffer::MarkedFile &File, unsigned Offset,
                         unsigned &NextSpelled) {
    for (; NextSpelled < File.SpelledTokens.size(); ++NextSpelled) {
      auto L = File.SpelledTokens[NextSpelled].location();
      if (Offset <= SM.getFileOffset(L))
        return llvm::None; // reached the offset we are looking for.
      auto Mapping = CollectedExpansions.find(L.getRawEncoding());
      if (Mapping != CollectedExpansions.end())
        return Mapping->second; // found a mapping before the offset.
    }
    return llvm::None; // no more tokens, we "reached" the offset.
  }

  /// Adds empty mappings for unconsumed spelled tokens at the end of each file.
  void fillGapsAtEndOfFiles() {
    for (auto &F : Result.Files) {
      if (F.second.SpelledTokens.empty())
        continue;
      fillGapUntil(F.second, F.second.SpelledTokens.back().endLocation(),
                   F.second.EndExpanded);
    }
  }

  TokenBuffer Result;
  /// For each file, a position of the next spelled token we will consume.
  llvm::DenseMap<FileID, unsigned> NextSpelled;
  PPExpansions CollectedExpansions;
  const SourceManager &SM;
  const LangOptions &LangOpts;
};

TokenBuffer TokenCollector::consume() && {
  PP.setTokenWatcher(nullptr);
  Collector->disable();
  return Builder(std::move(Expanded), std::move(Expansions),
                 PP.getSourceManager(), PP.getLangOpts())
      .build();
}

std::string syntax::Token::str() const {
  return llvm::formatv("Token({0}, length = {1})", tok::getTokenName(kind()),
                       length());
}

std::string syntax::Token::dumpForTests(const SourceManager &SM) const {
  return llvm::formatv("{0}   {1}", tok::getTokenName(kind()), text(SM));
}

std::string TokenBuffer::dumpForTests() const {
  auto PrintToken = [this](const syntax::Token &T) -> std::string {
    if (T.kind() == tok::eof)
      return "<eof>";
    return T.text(*SourceMgr);
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
  DumpTokens(OS, llvm::makeArrayRef(ExpandedTokens).drop_back());
  OS << "\n";

  std::vector<FileID> Keys;
  for (auto F : Files)
    Keys.push_back(F.first);
  llvm::sort(Keys);

  for (FileID ID : Keys) {
    const MarkedFile &File = Files.find(ID)->second;
    auto *Entry = SourceMgr->getFileEntryForID(ID);
    if (!Entry)
      continue; // Skip builtin files.
    OS << llvm::formatv("file '{0}'\n", Entry->getName())
       << "  spelled tokens:\n"
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
  return OS.str();
}
