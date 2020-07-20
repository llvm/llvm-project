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
      assert(Collector->Expansions.count(Range.getBegin().getRawEncoding()) &&
             "Overlapping macros should have same expansion location");
    }

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
    assert(!Result.ExpandedTokens.empty());
    assert(Result.ExpandedTokens.back().kind() == tok::eof);

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

    return std::move(Result);
  }

private:
  // Consume a sequence of spelled tokens that didn't expand to anything.
  // In the simplest case, skips spelled tokens until finding one that produced
  // the NextExpanded token, and creates an empty mapping for them.
  // If Drain is provided, skips remaining tokens from that file instead.
  void discard(llvm::Optional<FileID> Drain = llvm::None) {
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
      SourceLocation KnownEnd = CollectedExpansions.lookup(
          SpelledTokens[NextSpelled].location().getRawEncoding());
      if (KnownEnd.isValid()) {
        FlushMapping(); // Emits [Start, NextSpelled)
        while (NextSpelled < SpelledTokens.size() &&
               SpelledTokens[NextSpelled].location() <= KnownEnd)
          ++NextSpelled;
        FlushMapping(); // Emits [NextSpelled, KnownEnd]
        // Now the loop contitues and will emit (KnownEnd, Target).
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

    if (Tok.location().isFileID()) {
      // A run of file tokens continues while the expanded/spelled tokens match.
      while (NextSpelled < SpelledTokens.size() &&
             NextExpanded < Result.ExpandedTokens.size() &&
             SpelledTokens[NextSpelled].location() ==
                 Result.ExpandedTokens[NextExpanded].location()) {
        ++NextSpelled;
        ++NextExpanded;
      }
      // We need no mapping for file tokens copied to the expanded stream.
    } else {
      // We found a new macro expansion. We should have its spelling bounds.
      auto End = CollectedExpansions.lookup(Expansion.getRawEncoding());
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
    }
  }

  TokenBuffer Result;
  unsigned NextExpanded = 0;                    // cursor in ExpandedTokens
  llvm::DenseMap<FileID, unsigned> NextSpelled; // cursor in SpelledTokens
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
