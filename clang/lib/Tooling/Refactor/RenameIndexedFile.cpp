//===--- RenameIndexedFile.cpp - ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/RenameIndexedFile.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Refactor/RefactoringOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"

using namespace clang;

namespace clang {
namespace tooling {
namespace rename {

IndexedFileOccurrenceProducer::IndexedFileOccurrenceProducer(
    ArrayRef<IndexedSymbol> Symbols, IndexedFileOccurrenceConsumer &Consumer,
    IndexedFileRenamerLock &Lock, const RefactoringOptionSet *Options)
    : Symbols(Symbols), Consumer(Consumer), Lock(Lock), Options(Options) {
  IsMultiPiece = false;
  for (const auto &Symbol : Symbols) {
    if (Symbol.Name.size() > 1) {
      IsMultiPiece = true;
      break;
    }
  }
  if (IsMultiPiece) {
    for (const auto &Symbol : Symbols) {
      (void)Symbol;
      assert(Symbol.Name.size() > 1 &&
             "Mixed multi-piece and single piece symbols "
             "are unsupported");
    }
  }
}

namespace {

enum class MatchKind {
  SourceMatch,
  SourcePropSetterMatch,
  MacroExpansion,
  None
};

} // end anonymous namespace

static bool isSetterNameEqualToPropName(StringRef SetterName,
                                        StringRef PropertyName) {
  assert(SetterName.startswith("set") && "invalid setter name");
  SetterName = SetterName.drop_front(3);
  return SetterName[0] == toUppercase(PropertyName[0]) &&
         SetterName.drop_front() == PropertyName.drop_front();
}

static MatchKind checkOccurrence(const IndexedOccurrence &Occurrence,
                                 const IndexedSymbol &Symbol,
                                 const SourceManager &SM,
                                 const LangOptions &LangOpts,
                                 SourceRange &SymbolRange,
                                 bool AllowObjCSetterProp = false) {
  if (!Occurrence.Line || !Occurrence.Column)
    return MatchKind::None; // Ignore any invalid indexed locations.

  // Ensure that the first string in the name is present at the given
  // location.
  SourceLocation BeginLoc = SM.translateLineCol(
      SM.getMainFileID(), Occurrence.Line, Occurrence.Column);
  if (BeginLoc.isInvalid())
    return MatchKind::None;
  StringRef SymbolNameStart = Symbol.Name[0];
  // Extract the token at the location.
  auto DecomposedLoc = SM.getDecomposedLoc(BeginLoc);
  const llvm::MemoryBuffer *File = SM.getBuffer(DecomposedLoc.first);
  Lexer RawLex(
      BeginLoc, LangOpts, File->getBufferStart() + DecomposedLoc.second,
      File->getBufferStart() + DecomposedLoc.second, File->getBufferEnd());
  Token Tok;
  RawLex.LexFromRawLexer(Tok);
  if (Tok.isNot(tok::raw_identifier) || Tok.getLocation() != BeginLoc) {
    if (SymbolNameStart.empty() && Tok.is(tok::colon) &&
        Tok.getLocation() == BeginLoc) {
      // Must be the location of an empty Objective-C selector piece.
      SymbolRange = SourceRange(BeginLoc, BeginLoc);
      return MatchKind::SourceMatch;
    }
    // FIXME: Handle empty selector piece in a macro?
    return MatchKind::None;
  }
  SymbolRange = SourceRange(BeginLoc, Tok.getEndLoc());
  if (Tok.getRawIdentifier() == SymbolNameStart)
    return MatchKind::SourceMatch;
  // Match 'prop' when looking for 'setProp'.
  // FIXME: Verify that the previous token is a '.' to be sure.
  if (AllowObjCSetterProp &&
      Occurrence.Kind == IndexedOccurrence::IndexedObjCMessageSend &&
      SymbolNameStart.startswith("set") &&
      isSetterNameEqualToPropName(SymbolNameStart, Tok.getRawIdentifier()))
    return MatchKind::SourcePropSetterMatch;
  return MatchKind::MacroExpansion;
}

static void
findObjCMultiPieceSelectorOccurrences(CompilerInstance &CI,
                                      ArrayRef<IndexedSymbol> Symbols,
                                      IndexedFileOccurrenceConsumer &Consumer);

namespace {

struct TextualMatchOccurrence {
  SourceLocation Location;
  unsigned SymbolIndex;
};

/// Finds '@selector' expressions by looking at tokens one-by-one.
class SelectorParser {
  enum ParseState {
    None,
    At,
    Selector,
    ExpectingSelectorPiece,
    ExpectingColon,
    ExpectingRParenOrColon,
    ExpectingRParen,
    Success
  };
  ParseState State = None;
  const OldSymbolName &Name;

  ParseState stateForToken(const Token &RawTok);

public:
  unsigned SymbolIndex;
  llvm::SmallVector<SourceLocation, 8> SelectorLocations;

  SelectorParser(const OldSymbolName &Name, unsigned SymbolIndex)
      : Name(Name), SymbolIndex(SymbolIndex) {}

  /// Returns true if the parses has found a '@selector' expression.
  bool handleToken(const Token &RawTok);
};

class InclusionLexer final : public Lexer {
public:
  InclusionLexer(SourceLocation FileLoc, const LangOptions &LangOpts,
                 const char *BufStart, const char *BufEnd)
      : Lexer(FileLoc, LangOpts, BufStart, BufStart, BufEnd) {}

  void IndirectLex(Token &Result) override { LexFromRawLexer(Result); }
};

/// Finds matching textual occurrences in string literals.
class StringLiteralTextualParser {
  const OldSymbolName &Name;

public:
  unsigned SymbolIndex;

  StringLiteralTextualParser(const OldSymbolName &Name, unsigned SymbolIndex)
      : Name(Name), SymbolIndex(SymbolIndex) {
    assert(Name.size() == 1 && "can't search for multi-piece names in strings");
  }

  /// Returns the name's location if the parses has found a matching textual
  /// name in a string literal.
  SourceLocation handleToken(const Token &RawTok, Preprocessor &PP);
};

} // end anonymous namespace

SelectorParser::ParseState SelectorParser::stateForToken(const Token &RawTok) {
  assert(RawTok.isNot(tok::comment) && "unexpected comment token");
  switch (State) {
  case None:
    break;
  case At:
    if (RawTok.is(tok::raw_identifier) &&
        RawTok.getRawIdentifier() == "selector")
      return Selector;
    break;
  case Selector:
    if (RawTok.isNot(tok::l_paren))
      break;
    SelectorLocations.clear();
    return ExpectingSelectorPiece;
  case ExpectingSelectorPiece: {
    assert(SelectorLocations.size() < Name.size() &&
           "Expecting invalid selector piece");
    StringRef NamePiece = Name[SelectorLocations.size()];
    if ((RawTok.isNot(tok::raw_identifier) ||
         RawTok.getRawIdentifier() != NamePiece) &&
        !(NamePiece.empty() && RawTok.is(tok::colon))) {
      break;
    }
    SelectorLocations.push_back(RawTok.getLocation());
    if (SelectorLocations.size() == Name.size()) {
      // We found the selector that we were looking for, now check for ')'.
      return NamePiece.empty() ? ExpectingRParen : ExpectingRParenOrColon;
    }
    return NamePiece.empty() ? ExpectingSelectorPiece : ExpectingColon;
  }
  case ExpectingColon:
    if (RawTok.is(tok::colon))
      return ExpectingSelectorPiece;
    break;
  case ExpectingRParenOrColon:
    if (RawTok.is(tok::colon))
      return ExpectingRParen;
    LLVM_FALLTHROUGH;
  case ExpectingRParen:
    if (RawTok.is(tok::r_paren)) {
      // We found the selector that we were looking for.
      return Success;
    }
    break;
  case Success:
    llvm_unreachable("should not get here");
  }
  // Look for the start of the selector expression.
  return RawTok.is(tok::at) ? At : None;
}

bool SelectorParser::handleToken(const Token &RawTok) {
  if (RawTok.is(tok::coloncolon)) {
    // Split the '::' into two ':'.
    Token T(RawTok);
    T.setKind(tok::colon);
    T.setLength(1);
    handleToken(T);
    T.setLocation(T.getLocation().getLocWithOffset(1));
    return handleToken(T);
  }
  State = stateForToken(RawTok);
  if (State != Success)
    return false;
  State = None;
  return true;
}

SourceLocation StringLiteralTextualParser::handleToken(const Token &RawTok,
                                                       Preprocessor &PP) {
  if (!tok::isStringLiteral(RawTok.getKind()))
    return SourceLocation();
  StringLiteralParser Literal(RawTok, PP);
  if (Literal.hadError)
    return SourceLocation();
  return Literal.GetString() == Name[0]
             ? RawTok.getLocation().getLocWithOffset(
                   Literal.getOffsetOfStringByte(RawTok, 0))
             : SourceLocation();
}

static void collectTextualMatchesInComment(
    ArrayRef<IndexedSymbol> Symbols, SourceLocation CommentLoc,
    StringRef Comment, llvm::SmallVectorImpl<TextualMatchOccurrence> &Result) {
  for (const auto &Symbol : llvm::enumerate(Symbols)) {
    const OldSymbolName &Name = Symbol.value().Name;
    if (Name.containsEmptyPiece()) // Ignore Objective-C selectors with empty
                                   // pieces.
      continue;
    size_t Offset = 0;
    while (true) {
      Offset = Comment.find(Name[0], /*From=*/Offset);
      if (Offset == StringRef::npos)
        break;
      Result.push_back(
          {CommentLoc.getLocWithOffset(Offset), (unsigned)Symbol.index()});
      Offset += Name[0].size();
    }
  }
}

/// Lex the comment to figure out if textual matches in a comment are standalone
/// tokens.
static void findTextualMatchesInComment(
    const SourceManager &SM, const LangOptions &LangOpts,
    ArrayRef<IndexedSymbol> Symbols,
    ArrayRef<TextualMatchOccurrence> TextualMatches, SourceRange CommentRange,
    llvm::function_ref<void(OldSymbolOccurrence::OccurrenceKind,
                            ArrayRef<SourceLocation> Locations,
                            unsigned SymbolIndex)>
        MatchHandler) {
  std::string Source =
      Lexer::getSourceText(CharSourceRange::getCharRange(CommentRange), SM,
                           LangOpts)
          .str();
  OldSymbolOccurrence::OccurrenceKind Kind =
      RawComment(SM, CommentRange, LangOpts.CommentOpts, /*Merged=*/false)
              .isDocumentation()
          ? OldSymbolOccurrence::MatchingDocComment
          : OldSymbolOccurrence::MatchingComment;
  // Replace some special characters  with ' ' to avoid comments and literals.
  std::replace_if(
      Source.begin(), Source.end(),
      [](char c) -> bool { return c == '/' || c == '"' || c == '\''; }, ' ');
  Lexer RawLex(CommentRange.getBegin(), LangOpts, Source.c_str(),
               Source.c_str(), Source.c_str() + Source.size());
  Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(tok::eof)) {
    auto It = std::find_if(TextualMatches.begin(), TextualMatches.end(),
                           [&](const TextualMatchOccurrence &Match) {
                             return Match.Location == RawTok.getLocation();
                           });
    if (It != TextualMatches.end()) {
      StringRef TokenName =
          Lexer::getSourceText(CharSourceRange::getCharRange(
                                   RawTok.getLocation(), RawTok.getEndLoc()),
                               SM, LangOpts);
      // Only report matches that are identical to the symbol. When dealing with
      // multi-piece selectors we only look for the first selector piece as we
      // assume that textual matches correspond to a match of the first selector
      // piece.
      if (TokenName == Symbols[It->SymbolIndex].Name[0])
        MatchHandler(Kind, It->Location, It->SymbolIndex);
    }
    RawLex.LexFromRawLexer(RawTok);
  }
}

static void findMatchingTextualOccurrences(
    Preprocessor &PP, const SourceManager &SM, const LangOptions &LangOpts,
    ArrayRef<IndexedSymbol> Symbols,
    llvm::function_ref<void(OldSymbolOccurrence::OccurrenceKind,
                            ArrayRef<SourceLocation> Locations,
                            unsigned SymbolIndex)>
        MatchHandler) {
  const llvm::MemoryBuffer *FromFile = SM.getBuffer(SM.getMainFileID());
  Lexer RawLex(SM.getMainFileID(), FromFile, SM, LangOpts);
  RawLex.SetCommentRetentionState(true);

  llvm::SmallVector<TextualMatchOccurrence, 4> CommentMatches;
  llvm::SmallVector<SelectorParser, 2> SelectorParsers;
  for (const auto &Symbol : llvm::enumerate(Symbols)) {
    if (Symbol.value().IsObjCSelector)
      SelectorParsers.push_back(
          SelectorParser(Symbol.value().Name, Symbol.index()));
  }
  llvm::SmallVector<StringLiteralTextualParser, 1> StringParsers;
  for (const auto &Symbol : llvm::enumerate(Symbols)) {
    if (Symbol.value().SearchForStringLiteralOccurrences)
      StringParsers.push_back(
          StringLiteralTextualParser(Symbol.value().Name, Symbol.index()));
  }

  Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  bool ScanNonCommentTokens =
      !SelectorParsers.empty() || !StringParsers.empty();
  while (RawTok.isNot(tok::eof)) {
    if (RawTok.is(tok::comment)) {
      SourceRange Range(RawTok.getLocation(), RawTok.getEndLoc());
      StringRef Comment = Lexer::getSourceText(
          CharSourceRange::getCharRange(Range), SM, LangOpts);
      collectTextualMatchesInComment(Symbols, Range.getBegin(), Comment,
                                     CommentMatches);
      if (!CommentMatches.empty()) {
        findTextualMatchesInComment(SM, LangOpts, Symbols, CommentMatches,
                                    Range, MatchHandler);
        CommentMatches.clear();
      }
    } else if (ScanNonCommentTokens) {
      for (auto &Parser : SelectorParsers) {
        if (Parser.handleToken(RawTok))
          MatchHandler(OldSymbolOccurrence::MatchingSelector,
                       Parser.SelectorLocations, Parser.SymbolIndex);
      }
      for (auto &Parser : StringParsers) {
        SourceLocation Loc = Parser.handleToken(RawTok, PP);
        if (Loc.isValid())
          MatchHandler(OldSymbolOccurrence::MatchingStringLiteral, Loc,
                       Parser.SymbolIndex);
      }
    }
    RawLex.LexFromRawLexer(RawTok);
  }
}

static void findInclusionDirectiveOccurrence(
    const IndexedOccurrence &Occurrence, const IndexedSymbol &Symbol,
    unsigned SymbolIndex, SourceManager &SM, const LangOptions &LangOpts,
    IndexedFileOccurrenceConsumer &Consumer) {
  if (!Occurrence.Line || !Occurrence.Column)
    return; // Ignore any invalid indexed locations.

  SourceLocation Loc = SM.translateLineCol(SM.getMainFileID(), Occurrence.Line,
                                           Occurrence.Column);
  if (Loc.isInvalid())
    return;
  unsigned Offset = SM.getDecomposedLoc(Loc).second;
  const llvm::MemoryBuffer *File = SM.getBuffer(SM.getMainFileID());

  InclusionLexer RawLex(Loc, LangOpts, File->getBufferStart() + Offset,
                        File->getBufferEnd());
  Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  if (RawTok.isNot(tok::hash))
    return;
  // include/import
  RawLex.LexFromRawLexer(RawTok);
  if (RawTok.isNot(tok::raw_identifier))
    return;
  // string literal/angled literal.
  RawLex.setParsingPreprocessorDirective(true);
  RawLex.LexIncludeFilename(RawTok);
  if (RawTok.isNot(tok::string_literal) &&
      RawTok.isNot(tok::angle_string_literal))
    return;
  StringRef Filename = llvm::sys::path::filename(
      StringRef(RawTok.getLiteralData(), RawTok.getLength())
          .drop_front()
          .drop_back());
  size_t NameOffset = Filename.rfind_lower(Symbol.Name[0]);
  if (NameOffset == StringRef::npos)
    return;
  OldSymbolOccurrence Result(
      OldSymbolOccurrence::MatchingFilename,
      /*IsMacroExpansion=*/false, SymbolIndex,
      RawTok.getLocation().getLocWithOffset(
          NameOffset + (Filename.data() - RawTok.getLiteralData())));
  Consumer.handleOccurrence(Result, SM, LangOpts);
}

void IndexedFileOccurrenceProducer::ExecuteAction() {
  Lock.unlock(); // The execution should now be thread-safe.
  Preprocessor &PP = getCompilerInstance().getPreprocessor();
  PP.EnterMainSourceFile();

  SourceManager &SM = getCompilerInstance().getSourceManager();
  const LangOptions &LangOpts = getCompilerInstance().getLangOpts();
  if (IsMultiPiece) {
    findObjCMultiPieceSelectorOccurrences(getCompilerInstance(), Symbols,
                                          Consumer);
  } else {
    for (const auto &Symbol : llvm::enumerate(Symbols)) {
      for (const IndexedOccurrence &Occurrence :
           Symbol.value().IndexedOccurrences) {
        if (Occurrence.Kind == IndexedOccurrence::InclusionDirective) {
          findInclusionDirectiveOccurrence(Occurrence, Symbol.value(),
                                           Symbol.index(), SM, LangOpts,
                                           Consumer);
          continue;
        }
        SourceRange SymbolRange;
        MatchKind Match = checkOccurrence(Occurrence, Symbol.value(), SM,
                                          LangOpts, SymbolRange,
                                          /*AllowObjCSetterProp=*/true);
        if (Match == MatchKind::None)
          continue;
        llvm::SmallVector<SourceLocation, 2> Locs;
        Locs.push_back(SymbolRange.getBegin());
        bool IsImpProp = Match == MatchKind::SourcePropSetterMatch;
        if (IsImpProp)
          Locs.push_back(SymbolRange.getEnd());
        OldSymbolOccurrence Result(
            IsImpProp ? OldSymbolOccurrence::MatchingImplicitProperty
                      : OldSymbolOccurrence::MatchingSymbol,
            /*IsMacroExpansion=*/Match == MatchKind::MacroExpansion,
            Symbol.index(), Locs);
        Consumer.handleOccurrence(Result, SM, LangOpts);
      }
    }
  }

  if (Options && Options->get(option::AvoidTextualMatches()))
    return;
  findMatchingTextualOccurrences(
      PP, SM, LangOpts, Symbols,
      [&](OldSymbolOccurrence::OccurrenceKind Kind,
          ArrayRef<SourceLocation> Locations, unsigned SymbolIndex) {
        OldSymbolOccurrence Result(Kind, /*IsMacroExpansion=*/false,
                                   SymbolIndex, Locations);
        Consumer.handleOccurrence(Result, SM, LangOpts);
      });
}

namespace {

/// Maps from source locations to the indexed occurrences.
typedef llvm::DenseMap<unsigned, std::pair<IndexedOccurrence, unsigned>>
    SourceLocationsToIndexedOccurrences;

enum class ObjCSymbolSelectorKind { MessageSend, MethodDecl };

} // end anonymous namespace

static bool isMatchingSelectorName(const Token &Tok, const Token &Next,
                                   StringRef NamePiece) {
  if (NamePiece.empty())
    return Tok.is(tok::colon);
  return Tok.is(tok::raw_identifier) && Next.is(tok::colon) &&
         Tok.getRawIdentifier() == NamePiece;
}

static bool
findObjCSymbolSelectorPieces(ArrayRef<Token> Tokens, const OldSymbolName &Name,
                             SmallVectorImpl<SourceLocation> &Pieces,
                             ObjCSymbolSelectorKind Kind) {
  assert(!Tokens.empty() && "no tokens");
  assert(Name[0].empty() || Tokens[0].getRawIdentifier() == Name[0]);
  assert(Name.size() > 1);
  assert(Pieces.empty());

  Pieces.push_back(Tokens[0].getLocation());

  // We have to track square brackets, parens and braces as we want to skip the
  // tokens inside them. This ensures that we don't use identical selector
  // pieces in inner message sends, blocks, lambdas and @selector expressions.
  unsigned SquareCount = 0;
  unsigned ParenCount = 0;
  unsigned BraceCount = 0;

  // Start looking for the next selector piece.
  unsigned Last = Tokens.size() - 1;
  // Skip the ':' or any other token after the first selector piece token.
  for (unsigned Index = Name[0].empty() ? 1 : 2; Index < Last; ++Index) {
    const auto &Tok = Tokens[Index];

    bool NoScoping = SquareCount == 0 && BraceCount == 0 && ParenCount == 0;
    if (NoScoping &&
        isMatchingSelectorName(Tok, Tokens[Index + 1], Name[Pieces.size()])) {
      if (!Name[Pieces.size()].empty()) {
        // Skip the ':' after the name. This ensures that it won't match a
        // follow-up selector piece with an empty name.
        ++Index;
      }
      Pieces.push_back(Tok.getLocation());
      // All the selector pieces have been found.
      if (Pieces.size() == Name.size())
        return true;
    } else if (Tok.is(tok::r_square)) {
      // Stop scanning at the end of the message send.
      // Also account for spurious ']' in blocks or lambdas.
      if (Kind == ObjCSymbolSelectorKind::MessageSend && !SquareCount &&
          !BraceCount)
        break;
      if (SquareCount)
        --SquareCount;
    } else if (Tok.is(tok::l_square))
      ++SquareCount;
    else if (Tok.is(tok::l_paren))
      ++ParenCount;
    else if (Tok.is(tok::r_paren)) {
      if (!ParenCount)
        break;
      --ParenCount;
    } else if (Tok.is(tok::l_brace)) {
      // Stop scanning at the start of the of the method's body.
      // Also account for any spurious blocks inside argument parameter types
      // or parameter attributes.
      if (Kind == ObjCSymbolSelectorKind::MethodDecl && !BraceCount &&
          !ParenCount)
        break;
      ++BraceCount;
    } else if (Tok.is(tok::r_brace)) {
      if (!BraceCount)
        break;
      --BraceCount;
    }
    // Stop scanning at the end of the method's declaration.
    if (Kind == ObjCSymbolSelectorKind::MethodDecl && NoScoping &&
        (Tok.is(tok::semi) || Tok.is(tok::minus) || Tok.is(tok::plus)))
      break;
  }
  return false;
}

// Scan the file and find multi-piece selector occurrences in a token stream.
static void
findObjCMultiPieceSelectorOccurrences(CompilerInstance &CI,
                                      ArrayRef<IndexedSymbol> Symbols,
                                      IndexedFileOccurrenceConsumer &Consumer) {
  for (const auto &Symbol : Symbols) {
    (void)Symbol;
    assert(Symbol.Name.size() > 1 && "Not a multi-piece symbol!");
  }

  SourceManager &SM = CI.getSourceManager();
  const LangOptions &LangOpts = CI.getLangOpts();
  // Create a mapping from source locations to the indexed occurrences.
  SourceLocationsToIndexedOccurrences MappedIndexedOccurrences;
  for (const auto &Symbol : llvm::enumerate(Symbols)) {
    for (const IndexedOccurrence &Occurrence :
         Symbol.value().IndexedOccurrences) {
      // Selectors and names in #includes shouldn't really mix.
      if (Occurrence.Kind == IndexedOccurrence::InclusionDirective)
        continue;
      SourceRange SymbolRange;
      MatchKind Match = checkOccurrence(Occurrence, Symbol.value(), SM,
                                        LangOpts, SymbolRange);
      if (Match == MatchKind::None)
        continue;
      SourceLocation Loc = SymbolRange.getBegin();
      if (Match == MatchKind::MacroExpansion) {
        OldSymbolOccurrence Result(OldSymbolOccurrence::MatchingSymbol,
                                   /*IsMacroExpansion=*/true, Symbol.index(),
                                   Loc);
        Consumer.handleOccurrence(Result, SM, LangOpts);
        continue;
      }
      MappedIndexedOccurrences.try_emplace(Loc.getRawEncoding(), Occurrence,
                                           Symbol.index());
    }
  }

  // Lex the file and look for tokens.
  // Start lexing the specified input file.
  const llvm::MemoryBuffer *FromFile = SM.getBuffer(SM.getMainFileID());
  Lexer RawLex(SM.getMainFileID(), FromFile, SM, LangOpts);

  std::vector<Token> Tokens;
  bool SaveTokens = false;
  Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(tok::eof)) {
    // Start saving tokens only when we've got a match
    if (!SaveTokens) {
      if (MappedIndexedOccurrences.find(
              RawTok.getLocation().getRawEncoding()) !=
          MappedIndexedOccurrences.end())
        SaveTokens = true;
    }
    if (SaveTokens)
      Tokens.push_back(RawTok);
    RawLex.LexFromRawLexer(RawTok);
  }

  for (const auto &I : llvm::enumerate(Tokens)) {
    const auto &Tok = I.value();
    auto It = MappedIndexedOccurrences.find(Tok.getLocation().getRawEncoding());
    if (It == MappedIndexedOccurrences.end())
      continue;
    unsigned SymbolIndex = It->second.second;
    if (Tok.getKind() != tok::raw_identifier &&
        !(Symbols[SymbolIndex].Name[0].empty() && Tok.is(tok::colon)))
      continue;
    const IndexedOccurrence &Occurrence = It->second.first;

    // Scan the source for the remaining selector pieces.
    SmallVector<SourceLocation, 4> SelectorPieces;
    ObjCSymbolSelectorKind Kind =
        Occurrence.Kind == IndexedOccurrence::IndexedObjCMessageSend
            ? ObjCSymbolSelectorKind::MessageSend
            : ObjCSymbolSelectorKind::MethodDecl;
    if (findObjCSymbolSelectorPieces(
            llvm::makeArrayRef(Tokens).drop_front(I.index()),
            Symbols[SymbolIndex].Name, SelectorPieces, Kind)) {
      OldSymbolOccurrence Result(OldSymbolOccurrence::MatchingSymbol,
                                 /*IsMacroExpansion=*/false, SymbolIndex,
                                 std::move(SelectorPieces));
      Consumer.handleOccurrence(Result, SM, LangOpts);
    }
  }
}

} // end namespace rename
} // end namespace tooling
} // end namespace clang
