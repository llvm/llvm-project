//===--- UnwrappedLineParser.cpp - Format C++ code ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the UnwrappedLineParser,
/// which turns a stream of tokens into UnwrappedLines.
///
//===----------------------------------------------------------------------===//

#include "UnwrappedLineParser.h"
#include "FormatToken.h"
#include "TokenAnnotator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <utility>

#define DEBUG_TYPE "format-parser"

namespace clang {
namespace format {

class FormatTokenSource {
public:
  virtual ~FormatTokenSource() {}

  // Returns the next token in the token stream.
  virtual FormatToken *getNextToken() = 0;

  // Returns the token preceding the token returned by the last call to
  // getNextToken() in the token stream, or nullptr if no such token exists.
  virtual FormatToken *getPreviousToken() = 0;

  // Returns the token that would be returned by the next call to
  // getNextToken().
  virtual FormatToken *peekNextToken() = 0;

  // Returns the token that would be returned after the next N calls to
  // getNextToken(). N needs to be greater than zero, and small enough that
  // there are still tokens. Check for tok::eof with N-1 before calling it with
  // N.
  virtual FormatToken *peekNextToken(int N) = 0;

  // Returns whether we are at the end of the file.
  // This can be different from whether getNextToken() returned an eof token
  // when the FormatTokenSource is a view on a part of the token stream.
  virtual bool isEOF() = 0;

  // Gets the current position in the token stream, to be used by setPosition().
  virtual unsigned getPosition() = 0;

  // Resets the token stream to the state it was in when getPosition() returned
  // Position, and return the token at that position in the stream.
  virtual FormatToken *setPosition(unsigned Position) = 0;
};

namespace {

class ScopedDeclarationState {
public:
  ScopedDeclarationState(UnwrappedLine &Line, llvm::BitVector &Stack,
                         bool MustBeDeclaration)
      : Line(Line), Stack(Stack) {
    Line.MustBeDeclaration = MustBeDeclaration;
    Stack.push_back(MustBeDeclaration);
  }
  ~ScopedDeclarationState() {
    Stack.pop_back();
    if (!Stack.empty())
      Line.MustBeDeclaration = Stack.back();
    else
      Line.MustBeDeclaration = true;
  }

private:
  UnwrappedLine &Line;
  llvm::BitVector &Stack;
};

static bool isLineComment(const FormatToken &FormatTok) {
  return FormatTok.is(tok::comment) && !FormatTok.TokenText.startswith("/*");
}

// Checks if \p FormatTok is a line comment that continues the line comment
// \p Previous. The original column of \p MinColumnToken is used to determine
// whether \p FormatTok is indented enough to the right to continue \p Previous.
static bool continuesLineComment(const FormatToken &FormatTok,
                                 const FormatToken *Previous,
                                 const FormatToken *MinColumnToken) {
  if (!Previous || !MinColumnToken)
    return false;
  unsigned MinContinueColumn =
      MinColumnToken->OriginalColumn + (isLineComment(*MinColumnToken) ? 0 : 1);
  return isLineComment(FormatTok) && FormatTok.NewlinesBefore == 1 &&
         isLineComment(*Previous) &&
         FormatTok.OriginalColumn >= MinContinueColumn;
}

class ScopedMacroState : public FormatTokenSource {
public:
  ScopedMacroState(UnwrappedLine &Line, FormatTokenSource *&TokenSource,
                   FormatToken *&ResetToken)
      : Line(Line), TokenSource(TokenSource), ResetToken(ResetToken),
        PreviousLineLevel(Line.Level), PreviousTokenSource(TokenSource),
        Token(nullptr), PreviousToken(nullptr) {
    FakeEOF.Tok.startToken();
    FakeEOF.Tok.setKind(tok::eof);
    TokenSource = this;
    Line.Level = 0;
    Line.InPPDirective = true;
  }

  ~ScopedMacroState() override {
    TokenSource = PreviousTokenSource;
    ResetToken = Token;
    Line.InPPDirective = false;
    Line.Level = PreviousLineLevel;
  }

  FormatToken *getNextToken() override {
    // The \c UnwrappedLineParser guards against this by never calling
    // \c getNextToken() after it has encountered the first eof token.
    assert(!eof());
    PreviousToken = Token;
    Token = PreviousTokenSource->getNextToken();
    if (eof())
      return &FakeEOF;
    return Token;
  }

  FormatToken *getPreviousToken() override {
    return PreviousTokenSource->getPreviousToken();
  }

  FormatToken *peekNextToken() override {
    if (eof())
      return &FakeEOF;
    return PreviousTokenSource->peekNextToken();
  }

  FormatToken *peekNextToken(int N) override {
    assert(N > 0);
    if (eof())
      return &FakeEOF;
    return PreviousTokenSource->peekNextToken(N);
  }

  bool isEOF() override { return PreviousTokenSource->isEOF(); }

  unsigned getPosition() override { return PreviousTokenSource->getPosition(); }

  FormatToken *setPosition(unsigned Position) override {
    PreviousToken = nullptr;
    Token = PreviousTokenSource->setPosition(Position);
    return Token;
  }

private:
  bool eof() {
    return Token && Token->HasUnescapedNewline &&
           !continuesLineComment(*Token, PreviousToken,
                                 /*MinColumnToken=*/PreviousToken);
  }

  FormatToken FakeEOF;
  UnwrappedLine &Line;
  FormatTokenSource *&TokenSource;
  FormatToken *&ResetToken;
  unsigned PreviousLineLevel;
  FormatTokenSource *PreviousTokenSource;

  FormatToken *Token;
  FormatToken *PreviousToken;
};

} // end anonymous namespace

class ScopedLineState {
public:
  ScopedLineState(UnwrappedLineParser &Parser,
                  bool SwitchToPreprocessorLines = false)
      : Parser(Parser), OriginalLines(Parser.CurrentLines) {
    if (SwitchToPreprocessorLines)
      Parser.CurrentLines = &Parser.PreprocessorDirectives;
    else if (!Parser.Line->Tokens.empty())
      Parser.CurrentLines = &Parser.Line->Tokens.back().Children;
    PreBlockLine = std::move(Parser.Line);
    Parser.Line = std::make_unique<UnwrappedLine>();
    Parser.Line->Level = PreBlockLine->Level;
    Parser.Line->InPPDirective = PreBlockLine->InPPDirective;
  }

  ~ScopedLineState() {
    if (!Parser.Line->Tokens.empty())
      Parser.addUnwrappedLine();
    assert(Parser.Line->Tokens.empty());
    Parser.Line = std::move(PreBlockLine);
    if (Parser.CurrentLines == &Parser.PreprocessorDirectives)
      Parser.MustBreakBeforeNextToken = true;
    Parser.CurrentLines = OriginalLines;
  }

private:
  UnwrappedLineParser &Parser;

  std::unique_ptr<UnwrappedLine> PreBlockLine;
  SmallVectorImpl<UnwrappedLine> *OriginalLines;
};

class CompoundStatementIndenter {
public:
  CompoundStatementIndenter(UnwrappedLineParser *Parser,
                            const FormatStyle &Style, unsigned &LineLevel)
      : CompoundStatementIndenter(Parser, LineLevel,
                                  Style.BraceWrapping.AfterControlStatement,
                                  Style.BraceWrapping.IndentBraces) {}
  CompoundStatementIndenter(UnwrappedLineParser *Parser, unsigned &LineLevel,
                            bool WrapBrace, bool IndentBrace)
      : LineLevel(LineLevel), OldLineLevel(LineLevel) {
    if (WrapBrace)
      Parser->addUnwrappedLine();
    if (IndentBrace)
      ++LineLevel;
  }
  ~CompoundStatementIndenter() { LineLevel = OldLineLevel; }

private:
  unsigned &LineLevel;
  unsigned OldLineLevel;
};

namespace {

class IndexedTokenSource : public FormatTokenSource {
public:
  IndexedTokenSource(ArrayRef<FormatToken *> Tokens)
      : Tokens(Tokens), Position(-1) {}

  FormatToken *getNextToken() override {
    if (Position >= 0 && Tokens[Position]->is(tok::eof)) {
      LLVM_DEBUG({
        llvm::dbgs() << "Next ";
        dbgToken(Position);
      });
      return Tokens[Position];
    }
    ++Position;
    LLVM_DEBUG({
      llvm::dbgs() << "Next ";
      dbgToken(Position);
    });
    return Tokens[Position];
  }

  FormatToken *getPreviousToken() override {
    return Position > 0 ? Tokens[Position - 1] : nullptr;
  }

  FormatToken *peekNextToken() override {
    int Next = Position + 1;
    LLVM_DEBUG({
      llvm::dbgs() << "Peeking ";
      dbgToken(Next);
    });
    return Tokens[Next];
  }

  FormatToken *peekNextToken(int N) override {
    assert(N > 0);
    int Next = Position + N;
    LLVM_DEBUG({
      llvm::dbgs() << "Peeking (+" << (N - 1) << ") ";
      dbgToken(Next);
    });
    return Tokens[Next];
  }

  bool isEOF() override { return Tokens[Position]->is(tok::eof); }

  unsigned getPosition() override {
    LLVM_DEBUG(llvm::dbgs() << "Getting Position: " << Position << "\n");
    assert(Position >= 0);
    return Position;
  }

  FormatToken *setPosition(unsigned P) override {
    LLVM_DEBUG(llvm::dbgs() << "Setting Position: " << P << "\n");
    Position = P;
    return Tokens[Position];
  }

  void reset() { Position = -1; }

private:
  void dbgToken(int Position, llvm::StringRef Indent = "") {
    FormatToken *Tok = Tokens[Position];
    llvm::dbgs() << Indent << "[" << Position
                 << "] Token: " << Tok->Tok.getName() << " / " << Tok->TokenText
                 << ", Macro: " << !!Tok->MacroCtx << "\n";
  }

  ArrayRef<FormatToken *> Tokens;
  int Position;
};

} // end anonymous namespace

UnwrappedLineParser::UnwrappedLineParser(const FormatStyle &Style,
                                         const AdditionalKeywords &Keywords,
                                         unsigned FirstStartColumn,
                                         ArrayRef<FormatToken *> Tokens,
                                         UnwrappedLineConsumer &Callback)
    : Line(new UnwrappedLine), MustBreakBeforeNextToken(false),
      CurrentLines(&Lines), Style(Style), Keywords(Keywords),
      CommentPragmasRegex(Style.CommentPragmas), Tokens(nullptr),
      Callback(Callback), AllTokens(Tokens), PPBranchLevel(-1),
      IncludeGuard(Style.IndentPPDirectives == FormatStyle::PPDIS_None
                       ? IG_Rejected
                       : IG_Inited),
      IncludeGuardToken(nullptr), FirstStartColumn(FirstStartColumn) {}

void UnwrappedLineParser::reset() {
  PPBranchLevel = -1;
  IncludeGuard = Style.IndentPPDirectives == FormatStyle::PPDIS_None
                     ? IG_Rejected
                     : IG_Inited;
  IncludeGuardToken = nullptr;
  Line.reset(new UnwrappedLine);
  CommentsBeforeNextToken.clear();
  FormatTok = nullptr;
  MustBreakBeforeNextToken = false;
  PreprocessorDirectives.clear();
  CurrentLines = &Lines;
  DeclarationScopeStack.clear();
  NestedTooDeep.clear();
  PPStack.clear();
  Line->FirstStartColumn = FirstStartColumn;
}

void UnwrappedLineParser::parse() {
  IndexedTokenSource TokenSource(AllTokens);
  Line->FirstStartColumn = FirstStartColumn;
  do {
    LLVM_DEBUG(llvm::dbgs() << "----\n");
    reset();
    Tokens = &TokenSource;
    TokenSource.reset();

    readToken();
    parseFile();

    // If we found an include guard then all preprocessor directives (other than
    // the guard) are over-indented by one.
    if (IncludeGuard == IG_Found)
      for (auto &Line : Lines)
        if (Line.InPPDirective && Line.Level > 0)
          --Line.Level;

    // Create line with eof token.
    pushToken(FormatTok);
    addUnwrappedLine();

    for (const UnwrappedLine &Line : Lines)
      Callback.consumeUnwrappedLine(Line);

    Callback.finishRun();
    Lines.clear();
    while (!PPLevelBranchIndex.empty() &&
           PPLevelBranchIndex.back() + 1 >= PPLevelBranchCount.back()) {
      PPLevelBranchIndex.resize(PPLevelBranchIndex.size() - 1);
      PPLevelBranchCount.resize(PPLevelBranchCount.size() - 1);
    }
    if (!PPLevelBranchIndex.empty()) {
      ++PPLevelBranchIndex.back();
      assert(PPLevelBranchIndex.size() == PPLevelBranchCount.size());
      assert(PPLevelBranchIndex.back() <= PPLevelBranchCount.back());
    }
  } while (!PPLevelBranchIndex.empty());
}

void UnwrappedLineParser::parseFile() {
  // The top-level context in a file always has declarations, except for pre-
  // processor directives and JavaScript files.
  bool MustBeDeclaration = !Line->InPPDirective && !Style.isJavaScript();
  ScopedDeclarationState DeclarationState(*Line, DeclarationScopeStack,
                                          MustBeDeclaration);
  if (Style.Language == FormatStyle::LK_TextProto)
    parseBracedList();
  else
    parseLevel(/*OpeningBrace=*/nullptr, /*CanContainBracedList=*/true);
  // Make sure to format the remaining tokens.
  //
  // LK_TextProto is special since its top-level is parsed as the body of a
  // braced list, which does not necessarily have natural line separators such
  // as a semicolon. Comments after the last entry that have been determined to
  // not belong to that line, as in:
  //   key: value
  //   // endfile comment
  // do not have a chance to be put on a line of their own until this point.
  // Here we add this newline before end-of-file comments.
  if (Style.Language == FormatStyle::LK_TextProto &&
      !CommentsBeforeNextToken.empty())
    addUnwrappedLine();
  flushComments(true);
  addUnwrappedLine();
}

void UnwrappedLineParser::parseCSharpGenericTypeConstraint() {
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::l_brace:
      return;
    default:
      if (FormatTok->is(Keywords.kw_where)) {
        addUnwrappedLine();
        nextToken();
        parseCSharpGenericTypeConstraint();
        break;
      }
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseCSharpAttribute() {
  int UnpairedSquareBrackets = 1;
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::r_square:
      nextToken();
      --UnpairedSquareBrackets;
      if (UnpairedSquareBrackets == 0) {
        addUnwrappedLine();
        return;
      }
      break;
    case tok::l_square:
      ++UnpairedSquareBrackets;
      nextToken();
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

bool UnwrappedLineParser::precededByCommentOrPPDirective() const {
  if (!Lines.empty() && Lines.back().InPPDirective)
    return true;

  const FormatToken *Previous = Tokens->getPreviousToken();
  return Previous && Previous->is(tok::comment) &&
         (Previous->IsMultiline || Previous->NewlinesBefore > 0);
}

/// \brief Parses a level, that is ???.
/// \param OpeningBrace Opening brace (\p nullptr if absent) of that level
/// \param CanContainBracedList If the content can contain (at any level) a
/// braced list.
/// \param NextLBracesType The type for left brace found in this level.
/// \returns true if a simple block of if/else/for/while, or false otherwise.
/// (A simple block has a single statement.)
bool UnwrappedLineParser::parseLevel(const FormatToken *OpeningBrace,
                                     bool CanContainBracedList,
                                     IfStmtKind *IfKind,
                                     TokenType NextLBracesType) {
  auto NextLevelLBracesType = NextLBracesType == TT_CompoundRequirementLBrace
                                  ? TT_BracedListLBrace
                                  : TT_Unknown;
  const bool IsPrecededByCommentOrPPDirective =
      !Style.RemoveBracesLLVM || precededByCommentOrPPDirective();
  bool HasLabel = false;
  unsigned StatementCount = 0;
  bool SwitchLabelEncountered = false;
  do {
    if (FormatTok->getType() == TT_AttributeMacro) {
      nextToken();
      continue;
    }
    tok::TokenKind kind = FormatTok->Tok.getKind();
    if (FormatTok->getType() == TT_MacroBlockBegin)
      kind = tok::l_brace;
    else if (FormatTok->getType() == TT_MacroBlockEnd)
      kind = tok::r_brace;

    auto ParseDefault = [this, OpeningBrace, IfKind, NextLevelLBracesType,
                         &HasLabel, &StatementCount] {
      parseStructuralElement(IfKind, !OpeningBrace, NextLevelLBracesType,
                             HasLabel ? nullptr : &HasLabel);
      ++StatementCount;
      assert(StatementCount > 0 && "StatementCount overflow!");
    };

    switch (kind) {
    case tok::comment:
      nextToken();
      addUnwrappedLine();
      break;
    case tok::l_brace:
      if (NextLBracesType != TT_Unknown)
        FormatTok->setFinalizedType(NextLBracesType);
      else if (FormatTok->Previous &&
               FormatTok->Previous->ClosesRequiresClause) {
        // We need the 'default' case here to correctly parse a function
        // l_brace.
        ParseDefault();
        continue;
      }
      if (CanContainBracedList && !FormatTok->is(TT_MacroBlockBegin) &&
          tryToParseBracedList())
        continue;
      parseBlock(/*MustBeDeclaration=*/false, /*AddLevels=*/1u,
                 /*MunchSemi=*/true, /*KeepBraces=*/true,
                 /*UnindentWhitesmithsBraces=*/false, CanContainBracedList,
                 NextLBracesType);
      ++StatementCount;
      assert(StatementCount > 0 && "StatementCount overflow!");
      addUnwrappedLine();
      break;
    case tok::r_brace:
      if (OpeningBrace) {
        if (!Style.RemoveBracesLLVM ||
            !OpeningBrace->isOneOf(TT_ControlStatementLBrace, TT_ElseLBrace))
          return false;
        if (FormatTok->isNot(tok::r_brace) || StatementCount != 1 || HasLabel ||
            IsPrecededByCommentOrPPDirective ||
            precededByCommentOrPPDirective())
          return false;
        const FormatToken *Next = Tokens->peekNextToken();
        return Next->isNot(tok::comment) || Next->NewlinesBefore > 0;
      }
      nextToken();
      addUnwrappedLine();
      break;
    case tok::kw_default: {
      unsigned StoredPosition = Tokens->getPosition();
      FormatToken *Next;
      do {
        Next = Tokens->getNextToken();
        assert(Next);
      } while (Next->is(tok::comment));
      FormatTok = Tokens->setPosition(StoredPosition);
      if (Next->isNot(tok::colon)) {
        // default not followed by ':' is not a case label; treat it like
        // an identifier.
        parseStructuralElement();
        break;
      }
      // Else, if it is 'default:', fall through to the case handling.
      LLVM_FALLTHROUGH;
    }
    case tok::kw_case:
      if (Style.isJavaScript() && Line->MustBeDeclaration) {
        // A 'case: string' style field declaration.
        parseStructuralElement();
        break;
      }
      if (!SwitchLabelEncountered &&
          (Style.IndentCaseLabels || (Line->InPPDirective && Line->Level == 1)))
        ++Line->Level;
      SwitchLabelEncountered = true;
      parseStructuralElement();
      break;
    case tok::l_square:
      if (Style.isCSharp()) {
        nextToken();
        parseCSharpAttribute();
        break;
      }
      if (handleCppAttributes())
        break;
      LLVM_FALLTHROUGH;
    default:
      ParseDefault();
      break;
    }
  } while (!eof());
  return false;
}

void UnwrappedLineParser::calculateBraceTypes(bool ExpectClassBody) {
  // We'll parse forward through the tokens until we hit
  // a closing brace or eof - note that getNextToken() will
  // parse macros, so this will magically work inside macro
  // definitions, too.
  unsigned StoredPosition = Tokens->getPosition();
  FormatToken *Tok = FormatTok;
  const FormatToken *PrevTok = Tok->Previous;
  // Keep a stack of positions of lbrace tokens. We will
  // update information about whether an lbrace starts a
  // braced init list or a different block during the loop.
  SmallVector<FormatToken *, 8> LBraceStack;
  assert(Tok->is(tok::l_brace));
  do {
    // Get next non-comment token.
    FormatToken *NextTok;
    do {
      NextTok = Tokens->getNextToken();
    } while (NextTok->is(tok::comment));

    switch (Tok->Tok.getKind()) {
    case tok::l_brace:
      if (Style.isJavaScript() && PrevTok) {
        if (PrevTok->isOneOf(tok::colon, tok::less))
          // A ':' indicates this code is in a type, or a braced list
          // following a label in an object literal ({a: {b: 1}}).
          // A '<' could be an object used in a comparison, but that is nonsense
          // code (can never return true), so more likely it is a generic type
          // argument (`X<{a: string; b: number}>`).
          // The code below could be confused by semicolons between the
          // individual members in a type member list, which would normally
          // trigger BK_Block. In both cases, this must be parsed as an inline
          // braced init.
          Tok->setBlockKind(BK_BracedInit);
        else if (PrevTok->is(tok::r_paren))
          // `) { }` can only occur in function or method declarations in JS.
          Tok->setBlockKind(BK_Block);
      } else {
        Tok->setBlockKind(BK_Unknown);
      }
      LBraceStack.push_back(Tok);
      break;
    case tok::r_brace:
      if (LBraceStack.empty())
        break;
      if (LBraceStack.back()->is(BK_Unknown)) {
        bool ProbablyBracedList = false;
        if (Style.Language == FormatStyle::LK_Proto) {
          ProbablyBracedList = NextTok->isOneOf(tok::comma, tok::r_square);
        } else {
          // Skip NextTok over preprocessor lines, otherwise we may not
          // properly diagnose the block as a braced intializer
          // if the comma separator appears after the pp directive.
          while (NextTok->is(tok::hash)) {
            ScopedMacroState MacroState(*Line, Tokens, NextTok);
            do {
              NextTok = Tokens->getNextToken();
            } while (NextTok->isNot(tok::eof));
          }

          // Using OriginalColumn to distinguish between ObjC methods and
          // binary operators is a bit hacky.
          bool NextIsObjCMethod = NextTok->isOneOf(tok::plus, tok::minus) &&
                                  NextTok->OriginalColumn == 0;

          // Try to detect a braced list. Note that regardless how we mark inner
          // braces here, we will overwrite the BlockKind later if we parse a
          // braced list (where all blocks inside are by default braced lists),
          // or when we explicitly detect blocks (for example while parsing
          // lambdas).

          // If we already marked the opening brace as braced list, the closing
          // must also be part of it.
          ProbablyBracedList = LBraceStack.back()->is(TT_BracedListLBrace);

          ProbablyBracedList = ProbablyBracedList ||
                               (Style.isJavaScript() &&
                                NextTok->isOneOf(Keywords.kw_of, Keywords.kw_in,
                                                 Keywords.kw_as));
          ProbablyBracedList = ProbablyBracedList ||
                               (Style.isCpp() && NextTok->is(tok::l_paren));

          // If there is a comma, semicolon or right paren after the closing
          // brace, we assume this is a braced initializer list.
          // FIXME: Some of these do not apply to JS, e.g. "} {" can never be a
          // braced list in JS.
          ProbablyBracedList =
              ProbablyBracedList ||
              NextTok->isOneOf(tok::comma, tok::period, tok::colon,
                               tok::r_paren, tok::r_square, tok::l_brace,
                               tok::ellipsis);

          ProbablyBracedList =
              ProbablyBracedList ||
              (NextTok->is(tok::identifier) &&
               !PrevTok->isOneOf(tok::semi, tok::r_brace, tok::l_brace));

          ProbablyBracedList = ProbablyBracedList ||
                               (NextTok->is(tok::semi) &&
                                (!ExpectClassBody || LBraceStack.size() != 1));

          ProbablyBracedList =
              ProbablyBracedList ||
              (NextTok->isBinaryOperator() && !NextIsObjCMethod);

          if (!Style.isCSharp() && NextTok->is(tok::l_square)) {
            // We can have an array subscript after a braced init
            // list, but C++11 attributes are expected after blocks.
            NextTok = Tokens->getNextToken();
            ProbablyBracedList = NextTok->isNot(tok::l_square);
          }
        }
        if (ProbablyBracedList) {
          Tok->setBlockKind(BK_BracedInit);
          LBraceStack.back()->setBlockKind(BK_BracedInit);
        } else {
          Tok->setBlockKind(BK_Block);
          LBraceStack.back()->setBlockKind(BK_Block);
        }
      }
      LBraceStack.pop_back();
      break;
    case tok::identifier:
      if (!Tok->is(TT_StatementMacro))
        break;
      LLVM_FALLTHROUGH;
    case tok::at:
    case tok::semi:
    case tok::kw_if:
    case tok::kw_while:
    case tok::kw_for:
    case tok::kw_switch:
    case tok::kw_try:
    case tok::kw___try:
      if (!LBraceStack.empty() && LBraceStack.back()->is(BK_Unknown))
        LBraceStack.back()->setBlockKind(BK_Block);
      break;
    default:
      break;
    }
    PrevTok = Tok;
    Tok = NextTok;
  } while (Tok->isNot(tok::eof) && !LBraceStack.empty());

  // Assume other blocks for all unclosed opening braces.
  for (FormatToken *LBrace : LBraceStack)
    if (LBrace->is(BK_Unknown))
      LBrace->setBlockKind(BK_Block);

  FormatTok = Tokens->setPosition(StoredPosition);
}

template <class T>
static inline void hash_combine(std::size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

size_t UnwrappedLineParser::computePPHash() const {
  size_t h = 0;
  for (const auto &i : PPStack) {
    hash_combine(h, size_t(i.Kind));
    hash_combine(h, i.Line);
  }
  return h;
}

// Checks whether \p ParsedLine might fit on a single line. We must clone the
// tokens of \p ParsedLine before running the token annotator on it so that we
// can restore them afterward.
bool UnwrappedLineParser::mightFitOnOneLine(UnwrappedLine &ParsedLine) const {
  const auto ColumnLimit = Style.ColumnLimit;
  if (ColumnLimit == 0)
    return true;

  auto &Tokens = ParsedLine.Tokens;
  assert(!Tokens.empty());
  const auto *LastToken = Tokens.back().Tok;
  assert(LastToken);

  SmallVector<UnwrappedLineNode> SavedTokens(Tokens.size());

  int Index = 0;
  for (const auto &Token : Tokens) {
    assert(Token.Tok);
    auto &SavedToken = SavedTokens[Index++];
    SavedToken.Tok = new FormatToken;
    SavedToken.Tok->copyFrom(*Token.Tok);
    SavedToken.Children = std::move(Token.Children);
  }

  AnnotatedLine Line(ParsedLine);
  assert(Line.Last == LastToken);

  TokenAnnotator Annotator(Style, Keywords);
  Annotator.annotate(Line);
  Annotator.calculateFormattingInformation(Line);

  const int Length = LastToken->TotalLength;

  Index = 0;
  for (auto &Token : Tokens) {
    const auto &SavedToken = SavedTokens[Index++];
    Token.Tok->copyFrom(*SavedToken.Tok);
    Token.Children = std::move(SavedToken.Children);
    delete SavedToken.Tok;
  }

  return Line.Level * Style.IndentWidth + Length <= ColumnLimit;
}

UnwrappedLineParser::IfStmtKind UnwrappedLineParser::parseBlock(
    bool MustBeDeclaration, unsigned AddLevels, bool MunchSemi, bool KeepBraces,
    bool UnindentWhitesmithsBraces, bool CanContainBracedList,
    TokenType NextLBracesType) {
  assert(FormatTok->isOneOf(tok::l_brace, TT_MacroBlockBegin) &&
         "'{' or macro block token expected");
  FormatToken *Tok = FormatTok;
  const bool MacroBlock = FormatTok->is(TT_MacroBlockBegin);
  FormatTok->setBlockKind(BK_Block);

  // For Whitesmiths mode, jump to the next level prior to skipping over the
  // braces.
  if (AddLevels > 0 && Style.BreakBeforeBraces == FormatStyle::BS_Whitesmiths)
    ++Line->Level;

  size_t PPStartHash = computePPHash();

  unsigned InitialLevel = Line->Level;
  nextToken(/*LevelDifference=*/AddLevels);

  if (MacroBlock && FormatTok->is(tok::l_paren))
    parseParens();

  size_t NbPreprocessorDirectives =
      CurrentLines == &Lines ? PreprocessorDirectives.size() : 0;
  addUnwrappedLine();
  size_t OpeningLineIndex =
      CurrentLines->empty()
          ? (UnwrappedLine::kInvalidIndex)
          : (CurrentLines->size() - 1 - NbPreprocessorDirectives);

  // Whitesmiths is weird here. The brace needs to be indented for the namespace
  // block, but the block itself may not be indented depending on the style
  // settings. This allows the format to back up one level in those cases.
  if (UnindentWhitesmithsBraces)
    --Line->Level;

  ScopedDeclarationState DeclarationState(*Line, DeclarationScopeStack,
                                          MustBeDeclaration);
  if (AddLevels > 0u && Style.BreakBeforeBraces != FormatStyle::BS_Whitesmiths)
    Line->Level += AddLevels;

  IfStmtKind IfKind = IfStmtKind::NotIf;
  const bool SimpleBlock =
      parseLevel(Tok, CanContainBracedList, &IfKind, NextLBracesType);

  if (eof())
    return IfKind;

  if (MacroBlock ? !FormatTok->is(TT_MacroBlockEnd)
                 : !FormatTok->is(tok::r_brace)) {
    Line->Level = InitialLevel;
    FormatTok->setBlockKind(BK_Block);
    return IfKind;
  }

  if (SimpleBlock && !KeepBraces &&
      Tok->isOneOf(TT_ControlStatementLBrace, TT_ElseLBrace)) {
    assert(FormatTok->is(tok::r_brace));
    const FormatToken *Previous = Tokens->getPreviousToken();
    assert(Previous);
    if (Previous->isNot(tok::r_brace) || Previous->Optional) {
      assert(!CurrentLines->empty());
      if (mightFitOnOneLine(CurrentLines->back())) {
        Tok->MatchingParen = FormatTok;
        FormatTok->MatchingParen = Tok;
      }
    }
  }

  size_t PPEndHash = computePPHash();

  // Munch the closing brace.
  nextToken(/*LevelDifference=*/-AddLevels);

  if (MacroBlock && FormatTok->is(tok::l_paren))
    parseParens();

  if (FormatTok->is(tok::kw_noexcept)) {
    // A noexcept in a requires expression.
    nextToken();
  }

  if (FormatTok->is(tok::arrow)) {
    // Following the } or noexcept we can find a trailing return type arrow
    // as part of an implicit conversion constraint.
    nextToken();
    parseStructuralElement();
  }

  if (MunchSemi && FormatTok->is(tok::semi))
    nextToken();

  Line->Level = InitialLevel;

  if (PPStartHash == PPEndHash) {
    Line->MatchingOpeningBlockLineIndex = OpeningLineIndex;
    if (OpeningLineIndex != UnwrappedLine::kInvalidIndex) {
      // Update the opening line to add the forward reference as well
      (*CurrentLines)[OpeningLineIndex].MatchingClosingBlockLineIndex =
          CurrentLines->size() - 1;
    }
  }

  return IfKind;
}

static bool isGoogScope(const UnwrappedLine &Line) {
  // FIXME: Closure-library specific stuff should not be hard-coded but be
  // configurable.
  if (Line.Tokens.size() < 4)
    return false;
  auto I = Line.Tokens.begin();
  if (I->Tok->TokenText != "goog")
    return false;
  ++I;
  if (I->Tok->isNot(tok::period))
    return false;
  ++I;
  if (I->Tok->TokenText != "scope")
    return false;
  ++I;
  return I->Tok->is(tok::l_paren);
}

static bool isIIFE(const UnwrappedLine &Line,
                   const AdditionalKeywords &Keywords) {
  // Look for the start of an immediately invoked anonymous function.
  // https://en.wikipedia.org/wiki/Immediately-invoked_function_expression
  // This is commonly done in JavaScript to create a new, anonymous scope.
  // Example: (function() { ... })()
  if (Line.Tokens.size() < 3)
    return false;
  auto I = Line.Tokens.begin();
  if (I->Tok->isNot(tok::l_paren))
    return false;
  ++I;
  if (I->Tok->isNot(Keywords.kw_function))
    return false;
  ++I;
  return I->Tok->is(tok::l_paren);
}

static bool ShouldBreakBeforeBrace(const FormatStyle &Style,
                                   const FormatToken &InitialToken) {
  tok::TokenKind Kind = InitialToken.Tok.getKind();
  if (InitialToken.is(TT_NamespaceMacro))
    Kind = tok::kw_namespace;

  switch (Kind) {
  case tok::kw_namespace:
    return Style.BraceWrapping.AfterNamespace;
  case tok::kw_class:
    return Style.BraceWrapping.AfterClass;
  case tok::kw_union:
    return Style.BraceWrapping.AfterUnion;
  case tok::kw_struct:
    return Style.BraceWrapping.AfterStruct;
  case tok::kw_enum:
    return Style.BraceWrapping.AfterEnum;
  default:
    return false;
  }
}

void UnwrappedLineParser::parseChildBlock(
    bool CanContainBracedList, clang::format::TokenType NextLBracesType) {
  assert(FormatTok->is(tok::l_brace));
  FormatTok->setBlockKind(BK_Block);
  const FormatToken *OpeningBrace = FormatTok;
  nextToken();
  {
    bool SkipIndent = (Style.isJavaScript() &&
                       (isGoogScope(*Line) || isIIFE(*Line, Keywords)));
    ScopedLineState LineState(*this);
    ScopedDeclarationState DeclarationState(*Line, DeclarationScopeStack,
                                            /*MustBeDeclaration=*/false);
    Line->Level += SkipIndent ? 0 : 1;
    parseLevel(OpeningBrace, CanContainBracedList, /*IfKind=*/nullptr,
               NextLBracesType);
    flushComments(isOnNewLine(*FormatTok));
    Line->Level -= SkipIndent ? 0 : 1;
  }
  nextToken();
}

void UnwrappedLineParser::parsePPDirective() {
  assert(FormatTok->is(tok::hash) && "'#' expected");
  ScopedMacroState MacroState(*Line, Tokens, FormatTok);

  nextToken();

  if (!FormatTok->Tok.getIdentifierInfo()) {
    parsePPUnknown();
    return;
  }

  switch (FormatTok->Tok.getIdentifierInfo()->getPPKeywordID()) {
  case tok::pp_define:
    parsePPDefine();
    return;
  case tok::pp_if:
    parsePPIf(/*IfDef=*/false);
    break;
  case tok::pp_ifdef:
  case tok::pp_ifndef:
    parsePPIf(/*IfDef=*/true);
    break;
  case tok::pp_else:
    parsePPElse();
    break;
  case tok::pp_elifdef:
  case tok::pp_elifndef:
  case tok::pp_elif:
    parsePPElIf();
    break;
  case tok::pp_endif:
    parsePPEndIf();
    break;
  default:
    parsePPUnknown();
    break;
  }
}

void UnwrappedLineParser::conditionalCompilationCondition(bool Unreachable) {
  size_t Line = CurrentLines->size();
  if (CurrentLines == &PreprocessorDirectives)
    Line += Lines.size();

  if (Unreachable ||
      (!PPStack.empty() && PPStack.back().Kind == PP_Unreachable))
    PPStack.push_back({PP_Unreachable, Line});
  else
    PPStack.push_back({PP_Conditional, Line});
}

void UnwrappedLineParser::conditionalCompilationStart(bool Unreachable) {
  ++PPBranchLevel;
  assert(PPBranchLevel >= 0 && PPBranchLevel <= (int)PPLevelBranchIndex.size());
  if (PPBranchLevel == (int)PPLevelBranchIndex.size()) {
    PPLevelBranchIndex.push_back(0);
    PPLevelBranchCount.push_back(0);
  }
  PPChainBranchIndex.push(0);
  bool Skip = PPLevelBranchIndex[PPBranchLevel] > 0;
  conditionalCompilationCondition(Unreachable || Skip);
}

void UnwrappedLineParser::conditionalCompilationAlternative() {
  if (!PPStack.empty())
    PPStack.pop_back();
  assert(PPBranchLevel < (int)PPLevelBranchIndex.size());
  if (!PPChainBranchIndex.empty())
    ++PPChainBranchIndex.top();
  conditionalCompilationCondition(
      PPBranchLevel >= 0 && !PPChainBranchIndex.empty() &&
      PPLevelBranchIndex[PPBranchLevel] != PPChainBranchIndex.top());
}

void UnwrappedLineParser::conditionalCompilationEnd() {
  assert(PPBranchLevel < (int)PPLevelBranchIndex.size());
  if (PPBranchLevel >= 0 && !PPChainBranchIndex.empty()) {
    if (PPChainBranchIndex.top() + 1 > PPLevelBranchCount[PPBranchLevel])
      PPLevelBranchCount[PPBranchLevel] = PPChainBranchIndex.top() + 1;
  }
  // Guard against #endif's without #if.
  if (PPBranchLevel > -1)
    --PPBranchLevel;
  if (!PPChainBranchIndex.empty())
    PPChainBranchIndex.pop();
  if (!PPStack.empty())
    PPStack.pop_back();
}

void UnwrappedLineParser::parsePPIf(bool IfDef) {
  bool IfNDef = FormatTok->is(tok::pp_ifndef);
  nextToken();
  bool Unreachable = false;
  if (!IfDef && (FormatTok->is(tok::kw_false) || FormatTok->TokenText == "0"))
    Unreachable = true;
  if (IfDef && !IfNDef && FormatTok->TokenText == "SWIG")
    Unreachable = true;
  conditionalCompilationStart(Unreachable);
  FormatToken *IfCondition = FormatTok;
  // If there's a #ifndef on the first line, and the only lines before it are
  // comments, it could be an include guard.
  bool MaybeIncludeGuard = IfNDef;
  if (IncludeGuard == IG_Inited && MaybeIncludeGuard)
    for (auto &Line : Lines) {
      if (!Line.Tokens.front().Tok->is(tok::comment)) {
        MaybeIncludeGuard = false;
        IncludeGuard = IG_Rejected;
        break;
      }
    }
  --PPBranchLevel;
  parsePPUnknown();
  ++PPBranchLevel;
  if (IncludeGuard == IG_Inited && MaybeIncludeGuard) {
    IncludeGuard = IG_IfNdefed;
    IncludeGuardToken = IfCondition;
  }
}

void UnwrappedLineParser::parsePPElse() {
  // If a potential include guard has an #else, it's not an include guard.
  if (IncludeGuard == IG_Defined && PPBranchLevel == 0)
    IncludeGuard = IG_Rejected;
  conditionalCompilationAlternative();
  if (PPBranchLevel > -1)
    --PPBranchLevel;
  parsePPUnknown();
  ++PPBranchLevel;
}

void UnwrappedLineParser::parsePPElIf() { parsePPElse(); }

void UnwrappedLineParser::parsePPEndIf() {
  conditionalCompilationEnd();
  parsePPUnknown();
  // If the #endif of a potential include guard is the last thing in the file,
  // then we found an include guard.
  if (IncludeGuard == IG_Defined && PPBranchLevel == -1 && Tokens->isEOF() &&
      Style.IndentPPDirectives != FormatStyle::PPDIS_None)
    IncludeGuard = IG_Found;
}

void UnwrappedLineParser::parsePPDefine() {
  nextToken();

  if (!FormatTok->Tok.getIdentifierInfo()) {
    IncludeGuard = IG_Rejected;
    IncludeGuardToken = nullptr;
    parsePPUnknown();
    return;
  }

  if (IncludeGuard == IG_IfNdefed &&
      IncludeGuardToken->TokenText == FormatTok->TokenText) {
    IncludeGuard = IG_Defined;
    IncludeGuardToken = nullptr;
    for (auto &Line : Lines) {
      if (!Line.Tokens.front().Tok->isOneOf(tok::comment, tok::hash)) {
        IncludeGuard = IG_Rejected;
        break;
      }
    }
  }

  // In the context of a define, even keywords should be treated as normal
  // identifiers. Setting the kind to identifier is not enough, because we need
  // to treat additional keywords like __except as well, which are already
  // identifiers. Setting the identifier info to null interferes with include
  // guard processing above, and changes preprocessing nesting.
  FormatTok->Tok.setKind(tok::identifier);
  FormatTok->Tok.setIdentifierInfo(Keywords.kw_internal_ident_after_define);
  nextToken();
  if (FormatTok->Tok.getKind() == tok::l_paren &&
      !FormatTok->hasWhitespaceBefore())
    parseParens();
  if (Style.IndentPPDirectives != FormatStyle::PPDIS_None)
    Line->Level += PPBranchLevel + 1;
  addUnwrappedLine();
  ++Line->Level;

  // Errors during a preprocessor directive can only affect the layout of the
  // preprocessor directive, and thus we ignore them. An alternative approach
  // would be to use the same approach we use on the file level (no
  // re-indentation if there was a structural error) within the macro
  // definition.
  parseFile();
}

void UnwrappedLineParser::parsePPUnknown() {
  do {
    nextToken();
  } while (!eof());
  if (Style.IndentPPDirectives != FormatStyle::PPDIS_None)
    Line->Level += PPBranchLevel + 1;
  addUnwrappedLine();
}

// Here we exclude certain tokens that are not usually the first token in an
// unwrapped line. This is used in attempt to distinguish macro calls without
// trailing semicolons from other constructs split to several lines.
static bool tokenCanStartNewLine(const FormatToken &Tok) {
  // Semicolon can be a null-statement, l_square can be a start of a macro or
  // a C++11 attribute, but this doesn't seem to be common.
  return Tok.isNot(tok::semi) && Tok.isNot(tok::l_brace) &&
         Tok.isNot(TT_AttributeSquare) &&
         // Tokens that can only be used as binary operators and a part of
         // overloaded operator names.
         Tok.isNot(tok::period) && Tok.isNot(tok::periodstar) &&
         Tok.isNot(tok::arrow) && Tok.isNot(tok::arrowstar) &&
         Tok.isNot(tok::less) && Tok.isNot(tok::greater) &&
         Tok.isNot(tok::slash) && Tok.isNot(tok::percent) &&
         Tok.isNot(tok::lessless) && Tok.isNot(tok::greatergreater) &&
         Tok.isNot(tok::equal) && Tok.isNot(tok::plusequal) &&
         Tok.isNot(tok::minusequal) && Tok.isNot(tok::starequal) &&
         Tok.isNot(tok::slashequal) && Tok.isNot(tok::percentequal) &&
         Tok.isNot(tok::ampequal) && Tok.isNot(tok::pipeequal) &&
         Tok.isNot(tok::caretequal) && Tok.isNot(tok::greatergreaterequal) &&
         Tok.isNot(tok::lesslessequal) &&
         // Colon is used in labels, base class lists, initializer lists,
         // range-based for loops, ternary operator, but should never be the
         // first token in an unwrapped line.
         Tok.isNot(tok::colon) &&
         // 'noexcept' is a trailing annotation.
         Tok.isNot(tok::kw_noexcept);
}

static bool mustBeJSIdent(const AdditionalKeywords &Keywords,
                          const FormatToken *FormatTok) {
  // FIXME: This returns true for C/C++ keywords like 'struct'.
  return FormatTok->is(tok::identifier) &&
         (FormatTok->Tok.getIdentifierInfo() == nullptr ||
          !FormatTok->isOneOf(
              Keywords.kw_in, Keywords.kw_of, Keywords.kw_as, Keywords.kw_async,
              Keywords.kw_await, Keywords.kw_yield, Keywords.kw_finally,
              Keywords.kw_function, Keywords.kw_import, Keywords.kw_is,
              Keywords.kw_let, Keywords.kw_var, tok::kw_const,
              Keywords.kw_abstract, Keywords.kw_extends, Keywords.kw_implements,
              Keywords.kw_instanceof, Keywords.kw_interface,
              Keywords.kw_override, Keywords.kw_throws, Keywords.kw_from));
}

static bool mustBeJSIdentOrValue(const AdditionalKeywords &Keywords,
                                 const FormatToken *FormatTok) {
  return FormatTok->Tok.isLiteral() ||
         FormatTok->isOneOf(tok::kw_true, tok::kw_false) ||
         mustBeJSIdent(Keywords, FormatTok);
}

// isJSDeclOrStmt returns true if |FormatTok| starts a declaration or statement
// when encountered after a value (see mustBeJSIdentOrValue).
static bool isJSDeclOrStmt(const AdditionalKeywords &Keywords,
                           const FormatToken *FormatTok) {
  return FormatTok->isOneOf(
      tok::kw_return, Keywords.kw_yield,
      // conditionals
      tok::kw_if, tok::kw_else,
      // loops
      tok::kw_for, tok::kw_while, tok::kw_do, tok::kw_continue, tok::kw_break,
      // switch/case
      tok::kw_switch, tok::kw_case,
      // exceptions
      tok::kw_throw, tok::kw_try, tok::kw_catch, Keywords.kw_finally,
      // declaration
      tok::kw_const, tok::kw_class, Keywords.kw_var, Keywords.kw_let,
      Keywords.kw_async, Keywords.kw_function,
      // import/export
      Keywords.kw_import, tok::kw_export);
}

// Checks whether a token is a type in K&R C (aka C78).
static bool isC78Type(const FormatToken &Tok) {
  return Tok.isOneOf(tok::kw_char, tok::kw_short, tok::kw_int, tok::kw_long,
                     tok::kw_unsigned, tok::kw_float, tok::kw_double,
                     tok::identifier);
}

// This function checks whether a token starts the first parameter declaration
// in a K&R C (aka C78) function definition, e.g.:
//   int f(a, b)
//   short a, b;
//   {
//      return a + b;
//   }
static bool isC78ParameterDecl(const FormatToken *Tok, const FormatToken *Next,
                               const FormatToken *FuncName) {
  assert(Tok);
  assert(Next);
  assert(FuncName);

  if (FuncName->isNot(tok::identifier))
    return false;

  const FormatToken *Prev = FuncName->Previous;
  if (!Prev || (Prev->isNot(tok::star) && !isC78Type(*Prev)))
    return false;

  if (!isC78Type(*Tok) &&
      !Tok->isOneOf(tok::kw_register, tok::kw_struct, tok::kw_union))
    return false;

  if (Next->isNot(tok::star) && !Next->Tok.getIdentifierInfo())
    return false;

  Tok = Tok->Previous;
  if (!Tok || Tok->isNot(tok::r_paren))
    return false;

  Tok = Tok->Previous;
  if (!Tok || Tok->isNot(tok::identifier))
    return false;

  return Tok->Previous && Tok->Previous->isOneOf(tok::l_paren, tok::comma);
}

void UnwrappedLineParser::parseModuleImport() {
  nextToken();
  while (!eof()) {
    if (FormatTok->is(tok::colon)) {
      FormatTok->setFinalizedType(TT_ModulePartitionColon);
    }
    // Handle import <foo/bar.h> as we would an include statement.
    else if (FormatTok->is(tok::less)) {
      nextToken();
      while (!FormatTok->isOneOf(tok::semi, tok::greater, tok::eof)) {
        // Mark tokens up to the trailing line comments as implicit string
        // literals.
        if (FormatTok->isNot(tok::comment) &&
            !FormatTok->TokenText.startswith("//"))
          FormatTok->setFinalizedType(TT_ImplicitStringLiteral);
        nextToken();
      }
    }
    if (FormatTok->is(tok::semi)) {
      nextToken();
      break;
    }
    nextToken();
  }

  addUnwrappedLine();
}

// readTokenWithJavaScriptASI reads the next token and terminates the current
// line if JavaScript Automatic Semicolon Insertion must
// happen between the current token and the next token.
//
// This method is conservative - it cannot cover all edge cases of JavaScript,
// but only aims to correctly handle certain well known cases. It *must not*
// return true in speculative cases.
void UnwrappedLineParser::readTokenWithJavaScriptASI() {
  FormatToken *Previous = FormatTok;
  readToken();
  FormatToken *Next = FormatTok;

  bool IsOnSameLine =
      CommentsBeforeNextToken.empty()
          ? Next->NewlinesBefore == 0
          : CommentsBeforeNextToken.front()->NewlinesBefore == 0;
  if (IsOnSameLine)
    return;

  bool PreviousMustBeValue = mustBeJSIdentOrValue(Keywords, Previous);
  bool PreviousStartsTemplateExpr =
      Previous->is(TT_TemplateString) && Previous->TokenText.endswith("${");
  if (PreviousMustBeValue || Previous->is(tok::r_paren)) {
    // If the line contains an '@' sign, the previous token might be an
    // annotation, which can precede another identifier/value.
    bool HasAt = llvm::any_of(Line->Tokens, [](UnwrappedLineNode &LineNode) {
      return LineNode.Tok->is(tok::at);
    });
    if (HasAt)
      return;
  }
  if (Next->is(tok::exclaim) && PreviousMustBeValue)
    return addUnwrappedLine();
  bool NextMustBeValue = mustBeJSIdentOrValue(Keywords, Next);
  bool NextEndsTemplateExpr =
      Next->is(TT_TemplateString) && Next->TokenText.startswith("}");
  if (NextMustBeValue && !NextEndsTemplateExpr && !PreviousStartsTemplateExpr &&
      (PreviousMustBeValue ||
       Previous->isOneOf(tok::r_square, tok::r_paren, tok::plusplus,
                         tok::minusminus)))
    return addUnwrappedLine();
  if ((PreviousMustBeValue || Previous->is(tok::r_paren)) &&
      isJSDeclOrStmt(Keywords, Next))
    return addUnwrappedLine();
}

void UnwrappedLineParser::parseStructuralElement(IfStmtKind *IfKind,
                                                 bool IsTopLevel,
                                                 TokenType NextLBracesType,
                                                 bool *HasLabel) {
  if (Style.Language == FormatStyle::LK_TableGen &&
      FormatTok->is(tok::pp_include)) {
    nextToken();
    if (FormatTok->is(tok::string_literal))
      nextToken();
    addUnwrappedLine();
    return;
  }
  switch (FormatTok->Tok.getKind()) {
  case tok::kw_asm:
    nextToken();
    if (FormatTok->is(tok::l_brace)) {
      FormatTok->setFinalizedType(TT_InlineASMBrace);
      nextToken();
      while (FormatTok && FormatTok->isNot(tok::eof)) {
        if (FormatTok->is(tok::r_brace)) {
          FormatTok->setFinalizedType(TT_InlineASMBrace);
          nextToken();
          addUnwrappedLine();
          break;
        }
        FormatTok->Finalized = true;
        nextToken();
      }
    }
    break;
  case tok::kw_namespace:
    parseNamespace();
    return;
  case tok::kw_public:
  case tok::kw_protected:
  case tok::kw_private:
    if (Style.Language == FormatStyle::LK_Java || Style.isJavaScript() ||
        Style.isCSharp())
      nextToken();
    else
      parseAccessSpecifier();
    return;
  case tok::kw_if:
    if (Style.isJavaScript() && Line->MustBeDeclaration)
      // field/method declaration.
      break;
    parseIfThenElse(IfKind);
    return;
  case tok::kw_for:
  case tok::kw_while:
    if (Style.isJavaScript() && Line->MustBeDeclaration)
      // field/method declaration.
      break;
    parseForOrWhileLoop();
    return;
  case tok::kw_do:
    if (Style.isJavaScript() && Line->MustBeDeclaration)
      // field/method declaration.
      break;
    parseDoWhile();
    return;
  case tok::kw_switch:
    if (Style.isJavaScript() && Line->MustBeDeclaration)
      // 'switch: string' field declaration.
      break;
    parseSwitch();
    return;
  case tok::kw_default:
    if (Style.isJavaScript() && Line->MustBeDeclaration)
      // 'default: string' field declaration.
      break;
    nextToken();
    if (FormatTok->is(tok::colon)) {
      parseLabel();
      return;
    }
    // e.g. "default void f() {}" in a Java interface.
    break;
  case tok::kw_case:
    if (Style.isJavaScript() && Line->MustBeDeclaration) {
      // 'case: string' field declaration.
      nextToken();
      break;
    }
    parseCaseLabel();
    return;
  case tok::kw_try:
  case tok::kw___try:
    if (Style.isJavaScript() && Line->MustBeDeclaration)
      // field/method declaration.
      break;
    parseTryCatch();
    return;
  case tok::kw_extern:
    nextToken();
    if (FormatTok->is(tok::string_literal)) {
      nextToken();
      if (FormatTok->is(tok::l_brace)) {
        if (Style.BraceWrapping.AfterExternBlock)
          addUnwrappedLine();
        // Either we indent or for backwards compatibility we follow the
        // AfterExternBlock style.
        unsigned AddLevels =
            (Style.IndentExternBlock == FormatStyle::IEBS_Indent) ||
                    (Style.BraceWrapping.AfterExternBlock &&
                     Style.IndentExternBlock ==
                         FormatStyle::IEBS_AfterExternBlock)
                ? 1u
                : 0u;
        parseBlock(/*MustBeDeclaration=*/true, AddLevels);
        addUnwrappedLine();
        return;
      }
    }
    break;
  case tok::kw_export:
    if (Style.isJavaScript()) {
      parseJavaScriptEs6ImportExport();
      return;
    }
    if (!Style.isCpp())
      break;
    // Handle C++ "(inline|export) namespace".
    LLVM_FALLTHROUGH;
  case tok::kw_inline:
    nextToken();
    if (FormatTok->is(tok::kw_namespace)) {
      parseNamespace();
      return;
    }
    break;
  case tok::identifier:
    if (FormatTok->is(TT_ForEachMacro)) {
      parseForOrWhileLoop();
      return;
    }
    if (FormatTok->is(TT_MacroBlockBegin)) {
      parseBlock(/*MustBeDeclaration=*/false, /*AddLevels=*/1u,
                 /*MunchSemi=*/false);
      return;
    }
    if (FormatTok->is(Keywords.kw_import)) {
      if (Style.isJavaScript()) {
        parseJavaScriptEs6ImportExport();
        return;
      }
      if (Style.Language == FormatStyle::LK_Proto) {
        nextToken();
        if (FormatTok->is(tok::kw_public))
          nextToken();
        if (!FormatTok->is(tok::string_literal))
          return;
        nextToken();
        if (FormatTok->is(tok::semi))
          nextToken();
        addUnwrappedLine();
        return;
      }
      if (Style.isCpp()) {
        parseModuleImport();
        return;
      }
    }
    if (Style.isCpp() &&
        FormatTok->isOneOf(Keywords.kw_signals, Keywords.kw_qsignals,
                           Keywords.kw_slots, Keywords.kw_qslots)) {
      nextToken();
      if (FormatTok->is(tok::colon)) {
        nextToken();
        addUnwrappedLine();
        return;
      }
    }
    if (Style.isCpp() && FormatTok->is(TT_StatementMacro)) {
      parseStatementMacro();
      return;
    }
    if (Style.isCpp() && FormatTok->is(TT_NamespaceMacro)) {
      parseNamespace();
      return;
    }
    // In all other cases, parse the declaration.
    break;
  default:
    break;
  }
  do {
    const FormatToken *Previous = FormatTok->Previous;
    switch (FormatTok->Tok.getKind()) {
    case tok::at:
      nextToken();
      if (FormatTok->is(tok::l_brace)) {
        nextToken();
        parseBracedList();
        break;
      } else if (Style.Language == FormatStyle::LK_Java &&
                 FormatTok->is(Keywords.kw_interface)) {
        nextToken();
        break;
      }
      switch (FormatTok->Tok.getObjCKeywordID()) {
      case tok::objc_public:
      case tok::objc_protected:
      case tok::objc_package:
      case tok::objc_private:
        return parseAccessSpecifier();
      case tok::objc_interface:
      case tok::objc_implementation:
        return parseObjCInterfaceOrImplementation();
      case tok::objc_protocol:
        if (parseObjCProtocol())
          return;
        break;
      case tok::objc_end:
        return; // Handled by the caller.
      case tok::objc_optional:
      case tok::objc_required:
        nextToken();
        addUnwrappedLine();
        return;
      case tok::objc_autoreleasepool:
        nextToken();
        if (FormatTok->is(tok::l_brace)) {
          if (Style.BraceWrapping.AfterControlStatement ==
              FormatStyle::BWACS_Always)
            addUnwrappedLine();
          parseBlock();
        }
        addUnwrappedLine();
        return;
      case tok::objc_synchronized:
        nextToken();
        if (FormatTok->is(tok::l_paren))
          // Skip synchronization object
          parseParens();
        if (FormatTok->is(tok::l_brace)) {
          if (Style.BraceWrapping.AfterControlStatement ==
              FormatStyle::BWACS_Always)
            addUnwrappedLine();
          parseBlock();
        }
        addUnwrappedLine();
        return;
      case tok::objc_try:
        // This branch isn't strictly necessary (the kw_try case below would
        // do this too after the tok::at is parsed above).  But be explicit.
        parseTryCatch();
        return;
      default:
        break;
      }
      break;
    case tok::kw_concept:
      parseConcept();
      return;
    case tok::kw_requires: {
      if (Style.isCpp()) {
        bool ParsedClause = parseRequires();
        if (ParsedClause)
          return;
      } else {
        nextToken();
      }
      break;
    }
    case tok::kw_enum:
      // Ignore if this is part of "template <enum ...".
      if (Previous && Previous->is(tok::less)) {
        nextToken();
        break;
      }

      // parseEnum falls through and does not yet add an unwrapped line as an
      // enum definition can start a structural element.
      if (!parseEnum())
        break;
      // This only applies for C++.
      if (!Style.isCpp()) {
        addUnwrappedLine();
        return;
      }
      break;
    case tok::kw_typedef:
      nextToken();
      if (FormatTok->isOneOf(Keywords.kw_NS_ENUM, Keywords.kw_NS_OPTIONS,
                             Keywords.kw_CF_ENUM, Keywords.kw_CF_OPTIONS,
                             Keywords.kw_CF_CLOSED_ENUM,
                             Keywords.kw_NS_CLOSED_ENUM))
        parseEnum();
      break;
    case tok::kw_struct:
    case tok::kw_union:
    case tok::kw_class:
      if (parseStructLike())
        return;
      break;
    case tok::period:
      nextToken();
      // In Java, classes have an implicit static member "class".
      if (Style.Language == FormatStyle::LK_Java && FormatTok &&
          FormatTok->is(tok::kw_class))
        nextToken();
      if (Style.isJavaScript() && FormatTok &&
          FormatTok->Tok.getIdentifierInfo())
        // JavaScript only has pseudo keywords, all keywords are allowed to
        // appear in "IdentifierName" positions. See http://es5.github.io/#x7.6
        nextToken();
      break;
    case tok::semi:
      nextToken();
      addUnwrappedLine();
      return;
    case tok::r_brace:
      addUnwrappedLine();
      return;
    case tok::l_paren: {
      parseParens();
      // Break the unwrapped line if a K&R C function definition has a parameter
      // declaration.
      if (!IsTopLevel || !Style.isCpp() || !Previous || FormatTok->is(tok::eof))
        break;
      if (isC78ParameterDecl(FormatTok, Tokens->peekNextToken(), Previous)) {
        addUnwrappedLine();
        return;
      }
      break;
    }
    case tok::kw_operator:
      nextToken();
      if (FormatTok->isBinaryOperator())
        nextToken();
      break;
    case tok::caret:
      nextToken();
      if (FormatTok->Tok.isAnyIdentifier() ||
          FormatTok->isSimpleTypeSpecifier())
        nextToken();
      if (FormatTok->is(tok::l_paren))
        parseParens();
      if (FormatTok->is(tok::l_brace))
        parseChildBlock();
      break;
    case tok::l_brace:
      if (NextLBracesType != TT_Unknown)
        FormatTok->setFinalizedType(NextLBracesType);
      if (!tryToParsePropertyAccessor() && !tryToParseBracedList()) {
        // A block outside of parentheses must be the last part of a
        // structural element.
        // FIXME: Figure out cases where this is not true, and add projections
        // for them (the one we know is missing are lambdas).
        if (Style.Language == FormatStyle::LK_Java &&
            Line->Tokens.front().Tok->is(Keywords.kw_synchronized)) {
          // If necessary, we could set the type to something different than
          // TT_FunctionLBrace.
          if (Style.BraceWrapping.AfterControlStatement ==
              FormatStyle::BWACS_Always)
            addUnwrappedLine();
        } else if (Style.BraceWrapping.AfterFunction) {
          addUnwrappedLine();
        }
        if (!Line->InPPDirective)
          FormatTok->setFinalizedType(TT_FunctionLBrace);
        parseBlock();
        addUnwrappedLine();
        return;
      }
      // Otherwise this was a braced init list, and the structural
      // element continues.
      break;
    case tok::kw_try:
      if (Style.isJavaScript() && Line->MustBeDeclaration) {
        // field/method declaration.
        nextToken();
        break;
      }
      // We arrive here when parsing function-try blocks.
      if (Style.BraceWrapping.AfterFunction)
        addUnwrappedLine();
      parseTryCatch();
      return;
    case tok::identifier: {
      if (Style.isCSharp() && FormatTok->is(Keywords.kw_where) &&
          Line->MustBeDeclaration) {
        addUnwrappedLine();
        parseCSharpGenericTypeConstraint();
        break;
      }
      if (FormatTok->is(TT_MacroBlockEnd)) {
        addUnwrappedLine();
        return;
      }

      // Function declarations (as opposed to function expressions) are parsed
      // on their own unwrapped line by continuing this loop. Function
      // expressions (functions that are not on their own line) must not create
      // a new unwrapped line, so they are special cased below.
      size_t TokenCount = Line->Tokens.size();
      if (Style.isJavaScript() && FormatTok->is(Keywords.kw_function) &&
          (TokenCount > 1 || (TokenCount == 1 && !Line->Tokens.front().Tok->is(
                                                     Keywords.kw_async)))) {
        tryToParseJSFunction();
        break;
      }
      if ((Style.isJavaScript() || Style.Language == FormatStyle::LK_Java) &&
          FormatTok->is(Keywords.kw_interface)) {
        if (Style.isJavaScript()) {
          // In JavaScript/TypeScript, "interface" can be used as a standalone
          // identifier, e.g. in `var interface = 1;`. If "interface" is
          // followed by another identifier, it is very like to be an actual
          // interface declaration.
          unsigned StoredPosition = Tokens->getPosition();
          FormatToken *Next = Tokens->getNextToken();
          FormatTok = Tokens->setPosition(StoredPosition);
          if (!mustBeJSIdent(Keywords, Next)) {
            nextToken();
            break;
          }
        }
        parseRecord();
        addUnwrappedLine();
        return;
      }

      if (FormatTok->is(Keywords.kw_interface)) {
        if (parseStructLike())
          return;
        break;
      }

      if (Style.isCpp() && FormatTok->is(TT_StatementMacro)) {
        parseStatementMacro();
        return;
      }

      // See if the following token should start a new unwrapped line.
      StringRef Text = FormatTok->TokenText;

      FormatToken *PreviousToken = FormatTok;
      nextToken();

      // JS doesn't have macros, and within classes colons indicate fields, not
      // labels.
      if (Style.isJavaScript())
        break;

      TokenCount = Line->Tokens.size();
      if (TokenCount == 1 ||
          (TokenCount == 2 && Line->Tokens.front().Tok->is(tok::comment))) {
        if (FormatTok->is(tok::colon) && !Line->MustBeDeclaration) {
          Line->Tokens.begin()->Tok->MustBreakBefore = true;
          parseLabel(!Style.IndentGotoLabels);
          if (HasLabel)
            *HasLabel = true;
          return;
        }
        // Recognize function-like macro usages without trailing semicolon as
        // well as free-standing macros like Q_OBJECT.
        bool FunctionLike = FormatTok->is(tok::l_paren);
        if (FunctionLike)
          parseParens();

        bool FollowedByNewline =
            CommentsBeforeNextToken.empty()
                ? FormatTok->NewlinesBefore > 0
                : CommentsBeforeNextToken.front()->NewlinesBefore > 0;

        if (FollowedByNewline && (Text.size() >= 5 || FunctionLike) &&
            tokenCanStartNewLine(*FormatTok) && Text == Text.upper()) {
          PreviousToken->setFinalizedType(TT_FunctionLikeOrFreestandingMacro);
          addUnwrappedLine();
          return;
        }
      }
      break;
    }
    case tok::equal:
      if ((Style.isJavaScript() || Style.isCSharp()) &&
          FormatTok->is(TT_FatArrow)) {
        tryToParseChildBlock();
        break;
      }

      nextToken();
      if (FormatTok->is(tok::l_brace)) {
        // Block kind should probably be set to BK_BracedInit for any language.
        // C# needs this change to ensure that array initialisers and object
        // initialisers are indented the same way.
        if (Style.isCSharp())
          FormatTok->setBlockKind(BK_BracedInit);
        nextToken();
        parseBracedList();
      } else if (Style.Language == FormatStyle::LK_Proto &&
                 FormatTok->is(tok::less)) {
        nextToken();
        parseBracedList(/*ContinueOnSemicolons=*/false, /*IsEnum=*/false,
                        /*ClosingBraceKind=*/tok::greater);
      }
      break;
    case tok::l_square:
      parseSquare();
      break;
    case tok::kw_new:
      parseNew();
      break;
    case tok::kw_case:
      if (Style.isJavaScript() && Line->MustBeDeclaration) {
        // 'case: string' field declaration.
        nextToken();
        break;
      }
      parseCaseLabel();
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

bool UnwrappedLineParser::tryToParsePropertyAccessor() {
  assert(FormatTok->is(tok::l_brace));
  if (!Style.isCSharp())
    return false;
  // See if it's a property accessor.
  if (FormatTok->Previous->isNot(tok::identifier))
    return false;

  // See if we are inside a property accessor.
  //
  // Record the current tokenPosition so that we can advance and
  // reset the current token. `Next` is not set yet so we need
  // another way to advance along the token stream.
  unsigned int StoredPosition = Tokens->getPosition();
  FormatToken *Tok = Tokens->getNextToken();

  // A trivial property accessor is of the form:
  // { [ACCESS_SPECIFIER] [get]; [ACCESS_SPECIFIER] [set|init] }
  // Track these as they do not require line breaks to be introduced.
  bool HasSpecialAccessor = false;
  bool IsTrivialPropertyAccessor = true;
  while (!eof()) {
    if (Tok->isOneOf(tok::semi, tok::kw_public, tok::kw_private,
                     tok::kw_protected, Keywords.kw_internal, Keywords.kw_get,
                     Keywords.kw_init, Keywords.kw_set)) {
      if (Tok->isOneOf(Keywords.kw_get, Keywords.kw_init, Keywords.kw_set))
        HasSpecialAccessor = true;
      Tok = Tokens->getNextToken();
      continue;
    }
    if (Tok->isNot(tok::r_brace))
      IsTrivialPropertyAccessor = false;
    break;
  }

  if (!HasSpecialAccessor) {
    Tokens->setPosition(StoredPosition);
    return false;
  }

  // Try to parse the property accessor:
  // https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/classes-and-structs/properties
  Tokens->setPosition(StoredPosition);
  if (!IsTrivialPropertyAccessor && Style.BraceWrapping.AfterFunction)
    addUnwrappedLine();
  nextToken();
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::r_brace:
      nextToken();
      if (FormatTok->is(tok::equal)) {
        while (!eof() && FormatTok->isNot(tok::semi))
          nextToken();
        nextToken();
      }
      addUnwrappedLine();
      return true;
    case tok::l_brace:
      ++Line->Level;
      parseBlock(/*MustBeDeclaration=*/true);
      addUnwrappedLine();
      --Line->Level;
      break;
    case tok::equal:
      if (FormatTok->is(TT_FatArrow)) {
        ++Line->Level;
        do {
          nextToken();
        } while (!eof() && FormatTok->isNot(tok::semi));
        nextToken();
        addUnwrappedLine();
        --Line->Level;
        break;
      }
      nextToken();
      break;
    default:
      if (FormatTok->isOneOf(Keywords.kw_get, Keywords.kw_init,
                             Keywords.kw_set) &&
          !IsTrivialPropertyAccessor) {
        // Non-trivial get/set needs to be on its own line.
        addUnwrappedLine();
      }
      nextToken();
    }
  } while (!eof());

  // Unreachable for well-formed code (paired '{' and '}').
  return true;
}

bool UnwrappedLineParser::tryToParseLambda() {
  assert(FormatTok->is(tok::l_square));
  if (!Style.isCpp()) {
    nextToken();
    return false;
  }
  FormatToken &LSquare = *FormatTok;
  if (!tryToParseLambdaIntroducer())
    return false;

  bool SeenArrow = false;
  bool InTemplateParameterList = false;

  while (FormatTok->isNot(tok::l_brace)) {
    if (FormatTok->isSimpleTypeSpecifier()) {
      nextToken();
      continue;
    }
    switch (FormatTok->Tok.getKind()) {
    case tok::l_brace:
      break;
    case tok::l_paren:
      parseParens();
      break;
    case tok::l_square:
      parseSquare();
      break;
    case tok::kw_class:
    case tok::kw_template:
    case tok::kw_typename:
      assert(FormatTok->Previous);
      if (FormatTok->Previous->is(tok::less))
        InTemplateParameterList = true;
      nextToken();
      break;
    case tok::amp:
    case tok::star:
    case tok::kw_const:
    case tok::comma:
    case tok::less:
    case tok::greater:
    case tok::identifier:
    case tok::numeric_constant:
    case tok::coloncolon:
    case tok::kw_mutable:
    case tok::kw_noexcept:
      nextToken();
      break;
    // Specialization of a template with an integer parameter can contain
    // arithmetic, logical, comparison and ternary operators.
    //
    // FIXME: This also accepts sequences of operators that are not in the scope
    // of a template argument list.
    //
    // In a C++ lambda a template type can only occur after an arrow. We use
    // this as an heuristic to distinguish between Objective-C expressions
    // followed by an `a->b` expression, such as:
    // ([obj func:arg] + a->b)
    // Otherwise the code below would parse as a lambda.
    //
    // FIXME: This heuristic is incorrect for C++20 generic lambdas with
    // explicit template lists: []<bool b = true && false>(U &&u){}
    case tok::plus:
    case tok::minus:
    case tok::exclaim:
    case tok::tilde:
    case tok::slash:
    case tok::percent:
    case tok::lessless:
    case tok::pipe:
    case tok::pipepipe:
    case tok::ampamp:
    case tok::caret:
    case tok::equalequal:
    case tok::exclaimequal:
    case tok::greaterequal:
    case tok::lessequal:
    case tok::question:
    case tok::colon:
    case tok::ellipsis:
    case tok::kw_true:
    case tok::kw_false:
      if (SeenArrow || InTemplateParameterList) {
        nextToken();
        break;
      }
      return true;
    case tok::arrow:
      // This might or might not actually be a lambda arrow (this could be an
      // ObjC method invocation followed by a dereferencing arrow). We might
      // reset this back to TT_Unknown in TokenAnnotator.
      FormatTok->setFinalizedType(TT_LambdaArrow);
      SeenArrow = true;
      nextToken();
      break;
    default:
      return true;
    }
  }
  FormatTok->setFinalizedType(TT_LambdaLBrace);
  LSquare.setFinalizedType(TT_LambdaLSquare);
  parseChildBlock();
  return true;
}

bool UnwrappedLineParser::tryToParseLambdaIntroducer() {
  const FormatToken *Previous = FormatTok->Previous;
  const FormatToken *LeftSquare = FormatTok;
  nextToken();
  if (Previous &&
      (Previous->isOneOf(tok::identifier, tok::kw_operator, tok::kw_new,
                         tok::kw_delete, tok::l_square) ||
       LeftSquare->isCppStructuredBinding(Style) || Previous->closesScope() ||
       Previous->isSimpleTypeSpecifier())) {
    return false;
  }
  if (FormatTok->is(tok::l_square))
    return false;
  if (FormatTok->is(tok::r_square)) {
    const FormatToken *Next = Tokens->peekNextToken();
    if (Next->is(tok::greater))
      return false;
  }
  parseSquare(/*LambdaIntroducer=*/true);
  return true;
}

void UnwrappedLineParser::tryToParseJSFunction() {
  assert(FormatTok->is(Keywords.kw_function) ||
         FormatTok->startsSequence(Keywords.kw_async, Keywords.kw_function));
  if (FormatTok->is(Keywords.kw_async))
    nextToken();
  // Consume "function".
  nextToken();

  // Consume * (generator function). Treat it like C++'s overloaded operators.
  if (FormatTok->is(tok::star)) {
    FormatTok->setFinalizedType(TT_OverloadedOperator);
    nextToken();
  }

  // Consume function name.
  if (FormatTok->is(tok::identifier))
    nextToken();

  if (FormatTok->isNot(tok::l_paren))
    return;

  // Parse formal parameter list.
  parseParens();

  if (FormatTok->is(tok::colon)) {
    // Parse a type definition.
    nextToken();

    // Eat the type declaration. For braced inline object types, balance braces,
    // otherwise just parse until finding an l_brace for the function body.
    if (FormatTok->is(tok::l_brace))
      tryToParseBracedList();
    else
      while (!FormatTok->isOneOf(tok::l_brace, tok::semi) && !eof())
        nextToken();
  }

  if (FormatTok->is(tok::semi))
    return;

  parseChildBlock();
}

bool UnwrappedLineParser::tryToParseBracedList() {
  if (FormatTok->is(BK_Unknown))
    calculateBraceTypes();
  assert(FormatTok->isNot(BK_Unknown));
  if (FormatTok->is(BK_Block))
    return false;
  nextToken();
  parseBracedList();
  return true;
}

bool UnwrappedLineParser::tryToParseChildBlock() {
  assert(Style.isJavaScript() || Style.isCSharp());
  assert(FormatTok->is(TT_FatArrow));
  // Fat arrows (=>) have tok::TokenKind tok::equal but TokenType TT_FatArrow.
  // They always start an expression or a child block if followed by a curly
  // brace.
  nextToken();
  if (FormatTok->isNot(tok::l_brace))
    return false;
  parseChildBlock();
  return true;
}

bool UnwrappedLineParser::parseBracedList(bool ContinueOnSemicolons,
                                          bool IsEnum,
                                          tok::TokenKind ClosingBraceKind) {
  bool HasError = false;

  // FIXME: Once we have an expression parser in the UnwrappedLineParser,
  // replace this by using parseAssignmentExpression() inside.
  do {
    if (Style.isCSharp() && FormatTok->is(TT_FatArrow) &&
        tryToParseChildBlock())
      continue;
    if (Style.isJavaScript()) {
      if (FormatTok->is(Keywords.kw_function) ||
          FormatTok->startsSequence(Keywords.kw_async, Keywords.kw_function)) {
        tryToParseJSFunction();
        continue;
      }
      if (FormatTok->is(tok::l_brace)) {
        // Could be a method inside of a braced list `{a() { return 1; }}`.
        if (tryToParseBracedList())
          continue;
        parseChildBlock();
      }
    }
    if (FormatTok->Tok.getKind() == ClosingBraceKind) {
      if (IsEnum && !Style.AllowShortEnumsOnASingleLine)
        addUnwrappedLine();
      nextToken();
      return !HasError;
    }
    switch (FormatTok->Tok.getKind()) {
    case tok::l_square:
      if (Style.isCSharp())
        parseSquare();
      else
        tryToParseLambda();
      break;
    case tok::l_paren:
      parseParens();
      // JavaScript can just have free standing methods and getters/setters in
      // object literals. Detect them by a "{" following ")".
      if (Style.isJavaScript()) {
        if (FormatTok->is(tok::l_brace))
          parseChildBlock();
        break;
      }
      break;
    case tok::l_brace:
      // Assume there are no blocks inside a braced init list apart
      // from the ones we explicitly parse out (like lambdas).
      FormatTok->setBlockKind(BK_BracedInit);
      nextToken();
      parseBracedList();
      break;
    case tok::less:
      if (Style.Language == FormatStyle::LK_Proto ||
          ClosingBraceKind == tok::greater) {
        nextToken();
        parseBracedList(/*ContinueOnSemicolons=*/false, /*IsEnum=*/false,
                        /*ClosingBraceKind=*/tok::greater);
      } else {
        nextToken();
      }
      break;
    case tok::semi:
      // JavaScript (or more precisely TypeScript) can have semicolons in braced
      // lists (in so-called TypeMemberLists). Thus, the semicolon cannot be
      // used for error recovery if we have otherwise determined that this is
      // a braced list.
      if (Style.isJavaScript()) {
        nextToken();
        break;
      }
      HasError = true;
      if (!ContinueOnSemicolons)
        return !HasError;
      nextToken();
      break;
    case tok::comma:
      nextToken();
      if (IsEnum && !Style.AllowShortEnumsOnASingleLine)
        addUnwrappedLine();
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
  return false;
}

/// \brief Parses a pair of parentheses (and everything between them).
/// \param AmpAmpTokenType If different than TT_Unknown sets this type for all
/// double ampersands. This only counts for the current parens scope.
void UnwrappedLineParser::parseParens(TokenType AmpAmpTokenType) {
  assert(FormatTok->is(tok::l_paren) && "'(' expected.");
  nextToken();
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::l_paren:
      parseParens();
      if (Style.Language == FormatStyle::LK_Java && FormatTok->is(tok::l_brace))
        parseChildBlock();
      break;
    case tok::r_paren:
      nextToken();
      return;
    case tok::r_brace:
      // A "}" inside parenthesis is an error if there wasn't a matching "{".
      return;
    case tok::l_square:
      tryToParseLambda();
      break;
    case tok::l_brace:
      if (!tryToParseBracedList())
        parseChildBlock();
      break;
    case tok::at:
      nextToken();
      if (FormatTok->is(tok::l_brace)) {
        nextToken();
        parseBracedList();
      }
      break;
    case tok::equal:
      if (Style.isCSharp() && FormatTok->is(TT_FatArrow))
        tryToParseChildBlock();
      else
        nextToken();
      break;
    case tok::kw_class:
      if (Style.isJavaScript())
        parseRecord(/*ParseAsExpr=*/true);
      else
        nextToken();
      break;
    case tok::identifier:
      if (Style.isJavaScript() &&
          (FormatTok->is(Keywords.kw_function) ||
           FormatTok->startsSequence(Keywords.kw_async, Keywords.kw_function)))
        tryToParseJSFunction();
      else
        nextToken();
      break;
    case tok::kw_requires: {
      auto RequiresToken = FormatTok;
      nextToken();
      parseRequiresExpression(RequiresToken);
      break;
    }
    case tok::ampamp:
      if (AmpAmpTokenType != TT_Unknown)
        FormatTok->setFinalizedType(AmpAmpTokenType);
      LLVM_FALLTHROUGH;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseSquare(bool LambdaIntroducer) {
  if (!LambdaIntroducer) {
    assert(FormatTok->is(tok::l_square) && "'[' expected.");
    if (tryToParseLambda())
      return;
  }
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::l_paren:
      parseParens();
      break;
    case tok::r_square:
      nextToken();
      return;
    case tok::r_brace:
      // A "}" inside parenthesis is an error if there wasn't a matching "{".
      return;
    case tok::l_square:
      parseSquare();
      break;
    case tok::l_brace: {
      if (!tryToParseBracedList())
        parseChildBlock();
      break;
    }
    case tok::at:
      nextToken();
      if (FormatTok->is(tok::l_brace)) {
        nextToken();
        parseBracedList();
      }
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::keepAncestorBraces() {
  if (!Style.RemoveBracesLLVM)
    return;

  const int MaxNestingLevels = 2;
  const int Size = NestedTooDeep.size();
  if (Size >= MaxNestingLevels)
    NestedTooDeep[Size - MaxNestingLevels] = true;
  NestedTooDeep.push_back(false);
}

static FormatToken *getLastNonComment(const UnwrappedLine &Line) {
  for (const auto &Token : llvm::reverse(Line.Tokens))
    if (Token.Tok->isNot(tok::comment))
      return Token.Tok;

  return nullptr;
}

void UnwrappedLineParser::parseUnbracedBody(bool CheckEOF) {
  FormatToken *Tok = nullptr;

  if (Style.InsertBraces && !Line->InPPDirective && !Line->Tokens.empty() &&
      PreprocessorDirectives.empty()) {
    Tok = getLastNonComment(*Line);
    assert(Tok);
    if (Tok->BraceCount < 0) {
      assert(Tok->BraceCount == -1);
      Tok = nullptr;
    } else {
      Tok->BraceCount = -1;
    }
  }

  addUnwrappedLine();
  ++Line->Level;
  parseStructuralElement();

  if (Tok) {
    assert(!Line->InPPDirective);
    Tok = nullptr;
    for (const auto &L : llvm::reverse(*CurrentLines)) {
      if (!L.InPPDirective && getLastNonComment(L)) {
        Tok = L.Tokens.back().Tok;
        break;
      }
    }
    assert(Tok);
    ++Tok->BraceCount;
  }

  if (CheckEOF && FormatTok->is(tok::eof))
    addUnwrappedLine();

  --Line->Level;
}

static void markOptionalBraces(FormatToken *LeftBrace) {
  if (!LeftBrace)
    return;

  assert(LeftBrace->is(tok::l_brace));

  FormatToken *RightBrace = LeftBrace->MatchingParen;
  if (!RightBrace) {
    assert(!LeftBrace->Optional);
    return;
  }

  assert(RightBrace->is(tok::r_brace));
  assert(RightBrace->MatchingParen == LeftBrace);
  assert(LeftBrace->Optional == RightBrace->Optional);

  LeftBrace->Optional = true;
  RightBrace->Optional = true;
}

void UnwrappedLineParser::handleAttributes() {
  // Handle AttributeMacro, e.g. `if (x) UNLIKELY`.
  if (FormatTok->is(TT_AttributeMacro))
    nextToken();
  handleCppAttributes();
}

bool UnwrappedLineParser::handleCppAttributes() {
  // Handle [[likely]] / [[unlikely]] attributes.
  if (FormatTok->is(tok::l_square) && tryToParseSimpleAttribute()) {
    parseSquare();
    return true;
  }
  return false;
}

FormatToken *UnwrappedLineParser::parseIfThenElse(IfStmtKind *IfKind,
                                                  bool KeepBraces) {
  assert(FormatTok->is(tok::kw_if) && "'if' expected");
  nextToken();
  if (FormatTok->is(tok::exclaim))
    nextToken();

  bool KeepIfBraces = true;
  if (FormatTok->is(tok::kw_consteval)) {
    nextToken();
  } else {
    if (Style.RemoveBracesLLVM)
      KeepIfBraces = KeepBraces;
    if (FormatTok->isOneOf(tok::kw_constexpr, tok::identifier))
      nextToken();
    if (FormatTok->is(tok::l_paren))
      parseParens();
  }
  handleAttributes();

  bool NeedsUnwrappedLine = false;
  keepAncestorBraces();

  FormatToken *IfLeftBrace = nullptr;
  IfStmtKind IfBlockKind = IfStmtKind::NotIf;

  if (FormatTok->is(tok::l_brace)) {
    FormatTok->setFinalizedType(TT_ControlStatementLBrace);
    IfLeftBrace = FormatTok;
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    IfBlockKind = parseBlock(/*MustBeDeclaration=*/false, /*AddLevels=*/1u,
                             /*MunchSemi=*/true, KeepIfBraces);
    if (Style.BraceWrapping.BeforeElse)
      addUnwrappedLine();
    else
      NeedsUnwrappedLine = true;
  } else {
    parseUnbracedBody();
  }

  if (Style.RemoveBracesLLVM) {
    assert(!NestedTooDeep.empty());
    KeepIfBraces = KeepIfBraces ||
                   (IfLeftBrace && !IfLeftBrace->MatchingParen) ||
                   NestedTooDeep.back() || IfBlockKind == IfStmtKind::IfOnly ||
                   IfBlockKind == IfStmtKind::IfElseIf;
  }

  bool KeepElseBraces = KeepIfBraces;
  FormatToken *ElseLeftBrace = nullptr;
  IfStmtKind Kind = IfStmtKind::IfOnly;

  if (FormatTok->is(tok::kw_else)) {
    if (Style.RemoveBracesLLVM) {
      NestedTooDeep.back() = false;
      Kind = IfStmtKind::IfElse;
    }
    nextToken();
    handleAttributes();
    if (FormatTok->is(tok::l_brace)) {
      FormatTok->setFinalizedType(TT_ElseLBrace);
      ElseLeftBrace = FormatTok;
      CompoundStatementIndenter Indenter(this, Style, Line->Level);
      if (parseBlock(/*MustBeDeclaration=*/false, /*AddLevels=*/1u,
                     /*MunchSemi=*/true, KeepElseBraces) == IfStmtKind::IfOnly)
        Kind = IfStmtKind::IfElseIf;
      addUnwrappedLine();
    } else if (FormatTok->is(tok::kw_if)) {
      const FormatToken *Previous = Tokens->getPreviousToken();
      assert(Previous);
      const bool IsPrecededByComment = Previous->is(tok::comment);
      if (IsPrecededByComment) {
        addUnwrappedLine();
        ++Line->Level;
      }
      bool TooDeep = true;
      if (Style.RemoveBracesLLVM) {
        Kind = IfStmtKind::IfElseIf;
        TooDeep = NestedTooDeep.pop_back_val();
      }
      ElseLeftBrace = parseIfThenElse(/*IfKind=*/nullptr, KeepIfBraces);
      if (Style.RemoveBracesLLVM)
        NestedTooDeep.push_back(TooDeep);
      if (IsPrecededByComment)
        --Line->Level;
    } else {
      parseUnbracedBody(/*CheckEOF=*/true);
    }
  } else {
    if (Style.RemoveBracesLLVM)
      KeepIfBraces = KeepIfBraces || IfBlockKind == IfStmtKind::IfElse;
    if (NeedsUnwrappedLine)
      addUnwrappedLine();
  }

  if (!Style.RemoveBracesLLVM)
    return nullptr;

  assert(!NestedTooDeep.empty());
  KeepElseBraces = KeepElseBraces ||
                   (ElseLeftBrace && !ElseLeftBrace->MatchingParen) ||
                   NestedTooDeep.back();

  NestedTooDeep.pop_back();

  if (!KeepIfBraces && !KeepElseBraces) {
    markOptionalBraces(IfLeftBrace);
    markOptionalBraces(ElseLeftBrace);
  } else if (IfLeftBrace) {
    FormatToken *IfRightBrace = IfLeftBrace->MatchingParen;
    if (IfRightBrace) {
      assert(IfRightBrace->MatchingParen == IfLeftBrace);
      assert(!IfLeftBrace->Optional);
      assert(!IfRightBrace->Optional);
      IfLeftBrace->MatchingParen = nullptr;
      IfRightBrace->MatchingParen = nullptr;
    }
  }

  if (IfKind)
    *IfKind = Kind;

  return IfLeftBrace;
}

void UnwrappedLineParser::parseTryCatch() {
  assert(FormatTok->isOneOf(tok::kw_try, tok::kw___try) && "'try' expected");
  nextToken();
  bool NeedsUnwrappedLine = false;
  if (FormatTok->is(tok::colon)) {
    // We are in a function try block, what comes is an initializer list.
    nextToken();

    // In case identifiers were removed by clang-tidy, what might follow is
    // multiple commas in sequence - before the first identifier.
    while (FormatTok->is(tok::comma))
      nextToken();

    while (FormatTok->is(tok::identifier)) {
      nextToken();
      if (FormatTok->is(tok::l_paren))
        parseParens();
      if (FormatTok->Previous && FormatTok->Previous->is(tok::identifier) &&
          FormatTok->is(tok::l_brace)) {
        do {
          nextToken();
        } while (!FormatTok->is(tok::r_brace));
        nextToken();
      }

      // In case identifiers were removed by clang-tidy, what might follow is
      // multiple commas in sequence - after the first identifier.
      while (FormatTok->is(tok::comma))
        nextToken();
    }
  }
  // Parse try with resource.
  if (Style.Language == FormatStyle::LK_Java && FormatTok->is(tok::l_paren))
    parseParens();

  keepAncestorBraces();

  if (FormatTok->is(tok::l_brace)) {
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock();
    if (Style.BraceWrapping.BeforeCatch)
      addUnwrappedLine();
    else
      NeedsUnwrappedLine = true;
  } else if (!FormatTok->is(tok::kw_catch)) {
    // The C++ standard requires a compound-statement after a try.
    // If there's none, we try to assume there's a structuralElement
    // and try to continue.
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }
  while (true) {
    if (FormatTok->is(tok::at))
      nextToken();
    if (!(FormatTok->isOneOf(tok::kw_catch, Keywords.kw___except,
                             tok::kw___finally) ||
          ((Style.Language == FormatStyle::LK_Java || Style.isJavaScript()) &&
           FormatTok->is(Keywords.kw_finally)) ||
          (FormatTok->isObjCAtKeyword(tok::objc_catch) ||
           FormatTok->isObjCAtKeyword(tok::objc_finally))))
      break;
    nextToken();
    while (FormatTok->isNot(tok::l_brace)) {
      if (FormatTok->is(tok::l_paren)) {
        parseParens();
        continue;
      }
      if (FormatTok->isOneOf(tok::semi, tok::r_brace, tok::eof)) {
        if (Style.RemoveBracesLLVM)
          NestedTooDeep.pop_back();
        return;
      }
      nextToken();
    }
    NeedsUnwrappedLine = false;
    Line->MustBeDeclaration = false;
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock();
    if (Style.BraceWrapping.BeforeCatch)
      addUnwrappedLine();
    else
      NeedsUnwrappedLine = true;
  }

  if (Style.RemoveBracesLLVM)
    NestedTooDeep.pop_back();

  if (NeedsUnwrappedLine)
    addUnwrappedLine();
}

void UnwrappedLineParser::parseNamespace() {
  assert(FormatTok->isOneOf(tok::kw_namespace, TT_NamespaceMacro) &&
         "'namespace' expected");

  const FormatToken &InitialToken = *FormatTok;
  nextToken();
  if (InitialToken.is(TT_NamespaceMacro)) {
    parseParens();
  } else {
    while (FormatTok->isOneOf(tok::identifier, tok::coloncolon, tok::kw_inline,
                              tok::l_square, tok::period, tok::l_paren) ||
           (Style.isCSharp() && FormatTok->is(tok::kw_union)))
      if (FormatTok->is(tok::l_square))
        parseSquare();
      else if (FormatTok->is(tok::l_paren))
        parseParens();
      else
        nextToken();
  }
  if (FormatTok->is(tok::l_brace)) {
    if (ShouldBreakBeforeBrace(Style, InitialToken))
      addUnwrappedLine();

    unsigned AddLevels =
        Style.NamespaceIndentation == FormatStyle::NI_All ||
                (Style.NamespaceIndentation == FormatStyle::NI_Inner &&
                 DeclarationScopeStack.size() > 1)
            ? 1u
            : 0u;
    bool ManageWhitesmithsBraces =
        AddLevels == 0u &&
        Style.BreakBeforeBraces == FormatStyle::BS_Whitesmiths;

    // If we're in Whitesmiths mode, indent the brace if we're not indenting
    // the whole block.
    if (ManageWhitesmithsBraces)
      ++Line->Level;

    parseBlock(/*MustBeDeclaration=*/true, AddLevels, /*MunchSemi=*/true,
               /*KeepBraces=*/true, ManageWhitesmithsBraces);

    // Munch the semicolon after a namespace. This is more common than one would
    // think. Putting the semicolon into its own line is very ugly.
    if (FormatTok->is(tok::semi))
      nextToken();

    addUnwrappedLine(AddLevels > 0 ? LineLevel::Remove : LineLevel::Keep);

    if (ManageWhitesmithsBraces)
      --Line->Level;
  }
  // FIXME: Add error handling.
}

void UnwrappedLineParser::parseNew() {
  assert(FormatTok->is(tok::kw_new) && "'new' expected");
  nextToken();

  if (Style.isCSharp()) {
    do {
      if (FormatTok->is(tok::l_brace))
        parseBracedList();

      if (FormatTok->isOneOf(tok::semi, tok::comma))
        return;

      nextToken();
    } while (!eof());
  }

  if (Style.Language != FormatStyle::LK_Java)
    return;

  // In Java, we can parse everything up to the parens, which aren't optional.
  do {
    // There should not be a ;, { or } before the new's open paren.
    if (FormatTok->isOneOf(tok::semi, tok::l_brace, tok::r_brace))
      return;

    // Consume the parens.
    if (FormatTok->is(tok::l_paren)) {
      parseParens();

      // If there is a class body of an anonymous class, consume that as child.
      if (FormatTok->is(tok::l_brace))
        parseChildBlock();
      return;
    }
    nextToken();
  } while (!eof());
}

void UnwrappedLineParser::parseLoopBody(bool KeepBraces, bool WrapRightBrace) {
  keepAncestorBraces();

  if (FormatTok->is(tok::l_brace)) {
    if (!KeepBraces)
      FormatTok->setFinalizedType(TT_ControlStatementLBrace);
    FormatToken *LeftBrace = FormatTok;
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock(/*MustBeDeclaration=*/false, /*AddLevels=*/1u,
               /*MunchSemi=*/true, KeepBraces);
    if (!KeepBraces) {
      assert(!NestedTooDeep.empty());
      if (!NestedTooDeep.back())
        markOptionalBraces(LeftBrace);
    }
    if (WrapRightBrace)
      addUnwrappedLine();
  } else {
    parseUnbracedBody();
  }

  if (!KeepBraces)
    NestedTooDeep.pop_back();
}

void UnwrappedLineParser::parseForOrWhileLoop() {
  assert(FormatTok->isOneOf(tok::kw_for, tok::kw_while, TT_ForEachMacro) &&
         "'for', 'while' or foreach macro expected");
  const bool KeepBraces = !Style.RemoveBracesLLVM ||
                          !FormatTok->isOneOf(tok::kw_for, tok::kw_while);

  nextToken();
  // JS' for await ( ...
  if (Style.isJavaScript() && FormatTok->is(Keywords.kw_await))
    nextToken();
  if (Style.isCpp() && FormatTok->is(tok::kw_co_await))
    nextToken();
  if (FormatTok->is(tok::l_paren))
    parseParens();

  parseLoopBody(KeepBraces, /*WrapRightBrace=*/true);
}

void UnwrappedLineParser::parseDoWhile() {
  assert(FormatTok->is(tok::kw_do) && "'do' expected");
  nextToken();

  parseLoopBody(/*KeepBraces=*/true, Style.BraceWrapping.BeforeWhile);

  // FIXME: Add error handling.
  if (!FormatTok->is(tok::kw_while)) {
    addUnwrappedLine();
    return;
  }

  // If in Whitesmiths mode, the line with the while() needs to be indented
  // to the same level as the block.
  if (Style.BreakBeforeBraces == FormatStyle::BS_Whitesmiths)
    ++Line->Level;

  nextToken();
  parseStructuralElement();
}

void UnwrappedLineParser::parseLabel(bool LeftAlignLabel) {
  nextToken();
  unsigned OldLineLevel = Line->Level;
  if (Line->Level > 1 || (!Line->InPPDirective && Line->Level > 0))
    --Line->Level;
  if (LeftAlignLabel)
    Line->Level = 0;

  if (!Style.IndentCaseBlocks && CommentsBeforeNextToken.empty() &&
      FormatTok->is(tok::l_brace)) {

    CompoundStatementIndenter Indenter(this, Line->Level,
                                       Style.BraceWrapping.AfterCaseLabel,
                                       Style.BraceWrapping.IndentBraces);
    parseBlock();
    if (FormatTok->is(tok::kw_break)) {
      if (Style.BraceWrapping.AfterControlStatement ==
          FormatStyle::BWACS_Always) {
        addUnwrappedLine();
        if (!Style.IndentCaseBlocks &&
            Style.BreakBeforeBraces == FormatStyle::BS_Whitesmiths)
          ++Line->Level;
      }
      parseStructuralElement();
    }
    addUnwrappedLine();
  } else {
    if (FormatTok->is(tok::semi))
      nextToken();
    addUnwrappedLine();
  }
  Line->Level = OldLineLevel;
  if (FormatTok->isNot(tok::l_brace)) {
    parseStructuralElement();
    addUnwrappedLine();
  }
}

void UnwrappedLineParser::parseCaseLabel() {
  assert(FormatTok->is(tok::kw_case) && "'case' expected");

  // FIXME: fix handling of complex expressions here.
  do {
    nextToken();
  } while (!eof() && !FormatTok->is(tok::colon));
  parseLabel();
}

void UnwrappedLineParser::parseSwitch() {
  assert(FormatTok->is(tok::kw_switch) && "'switch' expected");
  nextToken();
  if (FormatTok->is(tok::l_paren))
    parseParens();

  keepAncestorBraces();

  if (FormatTok->is(tok::l_brace)) {
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock();
    addUnwrappedLine();
  } else {
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }

  if (Style.RemoveBracesLLVM)
    NestedTooDeep.pop_back();
}

// Operators that can follow a C variable.
static bool isCOperatorFollowingVar(tok::TokenKind kind) {
  switch (kind) {
  case tok::ampamp:
  case tok::ampequal:
  case tok::arrow:
  case tok::caret:
  case tok::caretequal:
  case tok::comma:
  case tok::ellipsis:
  case tok::equal:
  case tok::equalequal:
  case tok::exclaim:
  case tok::exclaimequal:
  case tok::greater:
  case tok::greaterequal:
  case tok::greatergreater:
  case tok::greatergreaterequal:
  case tok::l_paren:
  case tok::l_square:
  case tok::less:
  case tok::lessequal:
  case tok::lessless:
  case tok::lesslessequal:
  case tok::minus:
  case tok::minusequal:
  case tok::minusminus:
  case tok::percent:
  case tok::percentequal:
  case tok::period:
  case tok::pipe:
  case tok::pipeequal:
  case tok::pipepipe:
  case tok::plus:
  case tok::plusequal:
  case tok::plusplus:
  case tok::question:
  case tok::r_brace:
  case tok::r_paren:
  case tok::r_square:
  case tok::semi:
  case tok::slash:
  case tok::slashequal:
  case tok::star:
  case tok::starequal:
    return true;
  default:
    return false;
  }
}

void UnwrappedLineParser::parseAccessSpecifier() {
  FormatToken *AccessSpecifierCandidate = FormatTok;
  nextToken();
  // Understand Qt's slots.
  if (FormatTok->isOneOf(Keywords.kw_slots, Keywords.kw_qslots))
    nextToken();
  // Otherwise, we don't know what it is, and we'd better keep the next token.
  if (FormatTok->is(tok::colon)) {
    nextToken();
    addUnwrappedLine();
  } else if (!FormatTok->is(tok::coloncolon) &&
             !isCOperatorFollowingVar(FormatTok->Tok.getKind())) {
    // Not a variable name nor namespace name.
    addUnwrappedLine();
  } else if (AccessSpecifierCandidate) {
    // Consider the access specifier to be a C identifier.
    AccessSpecifierCandidate->Tok.setKind(tok::identifier);
  }
}

/// \brief Parses a concept definition.
/// \pre The current token has to be the concept keyword.
///
/// Returns if either the concept has been completely parsed, or if it detects
/// that the concept definition is incorrect.
void UnwrappedLineParser::parseConcept() {
  assert(FormatTok->is(tok::kw_concept) && "'concept' expected");
  nextToken();
  if (!FormatTok->is(tok::identifier))
    return;
  nextToken();
  if (!FormatTok->is(tok::equal))
    return;
  nextToken();
  parseConstraintExpression();
  if (FormatTok->is(tok::semi))
    nextToken();
  addUnwrappedLine();
}

/// \brief Parses a requires, decides if it is a clause or an expression.
/// \pre The current token has to be the requires keyword.
/// \returns true if it parsed a clause.
bool clang::format::UnwrappedLineParser::parseRequires() {
  assert(FormatTok->is(tok::kw_requires) && "'requires' expected");
  auto RequiresToken = FormatTok;

  // We try to guess if it is a requires clause, or a requires expression. For
  // that we first consume the keyword and check the next token.
  nextToken();

  switch (FormatTok->Tok.getKind()) {
  case tok::l_brace:
    // This can only be an expression, never a clause.
    parseRequiresExpression(RequiresToken);
    return false;
  case tok::l_paren:
    // Clauses and expression can start with a paren, it's unclear what we have.
    break;
  default:
    // All other tokens can only be a clause.
    parseRequiresClause(RequiresToken);
    return true;
  }

  // Looking forward we would have to decide if there are function declaration
  // like arguments to the requires expression:
  // requires (T t) {
  // Or there is a constraint expression for the requires clause:
  // requires (C<T> && ...

  // But first let's look behind.
  auto *PreviousNonComment = RequiresToken->getPreviousNonComment();

  if (!PreviousNonComment ||
      PreviousNonComment->is(TT_RequiresExpressionLBrace)) {
    // If there is no token, or an expression left brace, we are a requires
    // clause within a requires expression.
    parseRequiresClause(RequiresToken);
    return true;
  }

  switch (PreviousNonComment->Tok.getKind()) {
  case tok::greater:
  case tok::r_paren:
  case tok::kw_noexcept:
  case tok::kw_const:
    // This is a requires clause.
    parseRequiresClause(RequiresToken);
    return true;
  case tok::amp:
  case tok::ampamp: {
    // This can be either:
    // if (... && requires (T t) ...)
    // Or
    // void member(...) && requires (C<T> ...
    // We check the one token before that for a const:
    // void member(...) const && requires (C<T> ...
    auto PrevPrev = PreviousNonComment->getPreviousNonComment();
    if (PrevPrev && PrevPrev->is(tok::kw_const)) {
      parseRequiresClause(RequiresToken);
      return true;
    }
    break;
  }
  default:
    // It's an expression.
    parseRequiresExpression(RequiresToken);
    return false;
  }

  // Now we look forward and try to check if the paren content is a parameter
  // list. The parameters can be cv-qualified and contain references or
  // pointers.
  // So we want basically to check for TYPE NAME, but TYPE can contain all kinds
  // of stuff: typename, const, *, &, &&, ::, identifiers.

  int NextTokenOffset = 1;
  auto NextToken = Tokens->peekNextToken(NextTokenOffset);
  auto PeekNext = [&NextTokenOffset, &NextToken, this] {
    ++NextTokenOffset;
    NextToken = Tokens->peekNextToken(NextTokenOffset);
  };

  bool FoundType = false;
  bool LastWasColonColon = false;
  int OpenAngles = 0;

  for (; NextTokenOffset < 50; PeekNext()) {
    switch (NextToken->Tok.getKind()) {
    case tok::kw_volatile:
    case tok::kw_const:
    case tok::comma:
      parseRequiresExpression(RequiresToken);
      return false;
    case tok::r_paren:
    case tok::pipepipe:
      parseRequiresClause(RequiresToken);
      return true;
    case tok::eof:
      // Break out of the loop.
      NextTokenOffset = 50;
      break;
    case tok::coloncolon:
      LastWasColonColon = true;
      break;
    case tok::identifier:
      if (FoundType && !LastWasColonColon && OpenAngles == 0) {
        parseRequiresExpression(RequiresToken);
        return false;
      }
      FoundType = true;
      LastWasColonColon = false;
      break;
    case tok::less:
      ++OpenAngles;
      break;
    case tok::greater:
      --OpenAngles;
      break;
    default:
      if (NextToken->isSimpleTypeSpecifier()) {
        parseRequiresExpression(RequiresToken);
        return false;
      }
      break;
    }
  }

  // This seems to be a complicated expression, just assume it's a clause.
  parseRequiresClause(RequiresToken);
  return true;
}

/// \brief Parses a requires clause.
/// \param RequiresToken The requires keyword token, which starts this clause.
/// \pre We need to be on the next token after the requires keyword.
/// \sa parseRequiresExpression
///
/// Returns if it either has finished parsing the clause, or it detects, that
/// the clause is incorrect.
void UnwrappedLineParser::parseRequiresClause(FormatToken *RequiresToken) {
  assert(FormatTok->getPreviousNonComment() == RequiresToken);
  assert(RequiresToken->is(tok::kw_requires) && "'requires' expected");

  // If there is no previous token, we are within a requires expression,
  // otherwise we will always have the template or function declaration in front
  // of it.
  bool InRequiresExpression =
      !RequiresToken->Previous ||
      RequiresToken->Previous->is(TT_RequiresExpressionLBrace);

  RequiresToken->setFinalizedType(InRequiresExpression
                                      ? TT_RequiresClauseInARequiresExpression
                                      : TT_RequiresClause);

  parseConstraintExpression();

  if (!InRequiresExpression)
    FormatTok->Previous->ClosesRequiresClause = true;
}

/// \brief Parses a requires expression.
/// \param RequiresToken The requires keyword token, which starts this clause.
/// \pre We need to be on the next token after the requires keyword.
/// \sa parseRequiresClause
///
/// Returns if it either has finished parsing the expression, or it detects,
/// that the expression is incorrect.
void UnwrappedLineParser::parseRequiresExpression(FormatToken *RequiresToken) {
  assert(FormatTok->getPreviousNonComment() == RequiresToken);
  assert(RequiresToken->is(tok::kw_requires) && "'requires' expected");

  RequiresToken->setFinalizedType(TT_RequiresExpression);

  if (FormatTok->is(tok::l_paren)) {
    FormatTok->setFinalizedType(TT_RequiresExpressionLParen);
    parseParens();
  }

  if (FormatTok->is(tok::l_brace)) {
    FormatTok->setFinalizedType(TT_RequiresExpressionLBrace);
    parseChildBlock(/*CanContainBracedList=*/false,
                    /*NextLBracesType=*/TT_CompoundRequirementLBrace);
  }
}

/// \brief Parses a constraint expression.
///
/// This is either the definition of a concept, or the body of a requires
/// clause. It returns, when the parsing is complete, or the expression is
/// incorrect.
void UnwrappedLineParser::parseConstraintExpression() {
  // The special handling for lambdas is needed since tryToParseLambda() eats a
  // token and if a requires expression is the last part of a requires clause
  // and followed by an attribute like [[nodiscard]] the ClosesRequiresClause is
  // not set on the correct token. Thus we need to be aware if we even expect a
  // lambda to be possible.
  // template <typename T> requires requires { ... } [[nodiscard]] ...;
  bool LambdaNextTimeAllowed = true;
  do {
    bool LambdaThisTimeAllowed = std::exchange(LambdaNextTimeAllowed, false);

    switch (FormatTok->Tok.getKind()) {
    case tok::kw_requires: {
      auto RequiresToken = FormatTok;
      nextToken();
      parseRequiresExpression(RequiresToken);
      break;
    }

    case tok::l_paren:
      parseParens(/*AmpAmpTokenType=*/TT_BinaryOperator);
      break;

    case tok::l_square:
      if (!LambdaThisTimeAllowed || !tryToParseLambda())
        return;
      break;

    case tok::kw_const:
    case tok::semi:
    case tok::kw_class:
    case tok::kw_struct:
    case tok::kw_union:
      return;

    case tok::l_brace:
      // Potential function body.
      return;

    case tok::ampamp:
    case tok::pipepipe:
      FormatTok->setFinalizedType(TT_BinaryOperator);
      nextToken();
      LambdaNextTimeAllowed = true;
      break;

    case tok::comma:
    case tok::comment:
      LambdaNextTimeAllowed = LambdaThisTimeAllowed;
      nextToken();
      break;

    case tok::kw_sizeof:
    case tok::greater:
    case tok::greaterequal:
    case tok::greatergreater:
    case tok::less:
    case tok::lessequal:
    case tok::lessless:
    case tok::equalequal:
    case tok::exclaim:
    case tok::exclaimequal:
    case tok::plus:
    case tok::minus:
    case tok::star:
    case tok::slash:
    case tok::kw_decltype:
      LambdaNextTimeAllowed = true;
      // Just eat them.
      nextToken();
      break;

    case tok::numeric_constant:
    case tok::coloncolon:
    case tok::kw_true:
    case tok::kw_false:
      // Just eat them.
      nextToken();
      break;

    case tok::kw_static_cast:
    case tok::kw_const_cast:
    case tok::kw_reinterpret_cast:
    case tok::kw_dynamic_cast:
      nextToken();
      if (!FormatTok->is(tok::less))
        return;

      nextToken();
      parseBracedList(/*ContinueOnSemicolons=*/false, /*IsEnum=*/false,
                      /*ClosingBraceKind=*/tok::greater);
      break;

    case tok::kw_bool:
      // bool is only allowed if it is directly followed by a paren for a cast:
      // concept C = bool(...);
      // and bool is the only type, all other types as cast must be inside a
      // cast to bool an thus are handled by the other cases.
      nextToken();
      if (FormatTok->isNot(tok::l_paren))
        return;
      parseParens();
      break;

    default:
      if (!FormatTok->Tok.getIdentifierInfo()) {
        // Identifiers are part of the default case, we check for more then
        // tok::identifier to handle builtin type traits.
        return;
      }

      // We need to differentiate identifiers for a template deduction guide,
      // variables, or function return types (the constraint expression has
      // ended before that), and basically all other cases. But it's easier to
      // check the other way around.
      assert(FormatTok->Previous);
      switch (FormatTok->Previous->Tok.getKind()) {
      case tok::coloncolon:  // Nested identifier.
      case tok::ampamp:      // Start of a function or variable for the
      case tok::pipepipe:    // constraint expression.
      case tok::kw_requires: // Initial identifier of a requires clause.
      case tok::equal:       // Initial identifier of a concept declaration.
        break;
      default:
        return;
      }

      // Read identifier with optional template declaration.
      nextToken();
      if (FormatTok->is(tok::less)) {
        nextToken();
        parseBracedList(/*ContinueOnSemicolons=*/false, /*IsEnum=*/false,
                        /*ClosingBraceKind=*/tok::greater);
      }
      break;
    }
  } while (!eof());
}

bool UnwrappedLineParser::parseEnum() {
  const FormatToken &InitialToken = *FormatTok;

  // Won't be 'enum' for NS_ENUMs.
  if (FormatTok->is(tok::kw_enum))
    nextToken();

  // In TypeScript, "enum" can also be used as property name, e.g. in interface
  // declarations. An "enum" keyword followed by a colon would be a syntax
  // error and thus assume it is just an identifier.
  if (Style.isJavaScript() && FormatTok->isOneOf(tok::colon, tok::question))
    return false;

  // In protobuf, "enum" can be used as a field name.
  if (Style.Language == FormatStyle::LK_Proto && FormatTok->is(tok::equal))
    return false;

  // Eat up enum class ...
  if (FormatTok->isOneOf(tok::kw_class, tok::kw_struct))
    nextToken();

  while (FormatTok->Tok.getIdentifierInfo() ||
         FormatTok->isOneOf(tok::colon, tok::coloncolon, tok::less,
                            tok::greater, tok::comma, tok::question)) {
    nextToken();
    // We can have macros or attributes in between 'enum' and the enum name.
    if (FormatTok->is(tok::l_paren))
      parseParens();
    if (FormatTok->is(tok::identifier)) {
      nextToken();
      // If there are two identifiers in a row, this is likely an elaborate
      // return type. In Java, this can be "implements", etc.
      if (Style.isCpp() && FormatTok->is(tok::identifier))
        return false;
    }
  }

  // Just a declaration or something is wrong.
  if (FormatTok->isNot(tok::l_brace))
    return true;
  FormatTok->setFinalizedType(TT_EnumLBrace);
  FormatTok->setBlockKind(BK_Block);

  if (Style.Language == FormatStyle::LK_Java) {
    // Java enums are different.
    parseJavaEnumBody();
    return true;
  }
  if (Style.Language == FormatStyle::LK_Proto) {
    parseBlock(/*MustBeDeclaration=*/true);
    return true;
  }

  if (!Style.AllowShortEnumsOnASingleLine &&
      ShouldBreakBeforeBrace(Style, InitialToken))
    addUnwrappedLine();
  // Parse enum body.
  nextToken();
  if (!Style.AllowShortEnumsOnASingleLine) {
    addUnwrappedLine();
    Line->Level += 1;
  }
  bool HasError = !parseBracedList(/*ContinueOnSemicolons=*/true,
                                   /*IsEnum=*/true);
  if (!Style.AllowShortEnumsOnASingleLine)
    Line->Level -= 1;
  if (HasError) {
    if (FormatTok->is(tok::semi))
      nextToken();
    addUnwrappedLine();
  }
  return true;

  // There is no addUnwrappedLine() here so that we fall through to parsing a
  // structural element afterwards. Thus, in "enum A {} n, m;",
  // "} n, m;" will end up in one unwrapped line.
}

bool UnwrappedLineParser::parseStructLike() {
  // parseRecord falls through and does not yet add an unwrapped line as a
  // record declaration or definition can start a structural element.
  parseRecord();
  // This does not apply to Java, JavaScript and C#.
  if (Style.Language == FormatStyle::LK_Java || Style.isJavaScript() ||
      Style.isCSharp()) {
    if (FormatTok->is(tok::semi))
      nextToken();
    addUnwrappedLine();
    return true;
  }
  return false;
}

namespace {
// A class used to set and restore the Token position when peeking
// ahead in the token source.
class ScopedTokenPosition {
  unsigned StoredPosition;
  FormatTokenSource *Tokens;

public:
  ScopedTokenPosition(FormatTokenSource *Tokens) : Tokens(Tokens) {
    assert(Tokens && "Tokens expected to not be null");
    StoredPosition = Tokens->getPosition();
  }

  ~ScopedTokenPosition() { Tokens->setPosition(StoredPosition); }
};
} // namespace

// Look to see if we have [[ by looking ahead, if
// its not then rewind to the original position.
bool UnwrappedLineParser::tryToParseSimpleAttribute() {
  ScopedTokenPosition AutoPosition(Tokens);
  FormatToken *Tok = Tokens->getNextToken();
  // We already read the first [ check for the second.
  if (!Tok->is(tok::l_square))
    return false;
  // Double check that the attribute is just something
  // fairly simple.
  while (Tok->isNot(tok::eof)) {
    if (Tok->is(tok::r_square))
      break;
    Tok = Tokens->getNextToken();
  }
  if (Tok->is(tok::eof))
    return false;
  Tok = Tokens->getNextToken();
  if (!Tok->is(tok::r_square))
    return false;
  Tok = Tokens->getNextToken();
  if (Tok->is(tok::semi))
    return false;
  return true;
}

void UnwrappedLineParser::parseJavaEnumBody() {
  assert(FormatTok->is(tok::l_brace));
  const FormatToken *OpeningBrace = FormatTok;

  // Determine whether the enum is simple, i.e. does not have a semicolon or
  // constants with class bodies. Simple enums can be formatted like braced
  // lists, contracted to a single line, etc.
  unsigned StoredPosition = Tokens->getPosition();
  bool IsSimple = true;
  FormatToken *Tok = Tokens->getNextToken();
  while (!Tok->is(tok::eof)) {
    if (Tok->is(tok::r_brace))
      break;
    if (Tok->isOneOf(tok::l_brace, tok::semi)) {
      IsSimple = false;
      break;
    }
    // FIXME: This will also mark enums with braces in the arguments to enum
    // constants as "not simple". This is probably fine in practice, though.
    Tok = Tokens->getNextToken();
  }
  FormatTok = Tokens->setPosition(StoredPosition);

  if (IsSimple) {
    nextToken();
    parseBracedList();
    addUnwrappedLine();
    return;
  }

  // Parse the body of a more complex enum.
  // First add a line for everything up to the "{".
  nextToken();
  addUnwrappedLine();
  ++Line->Level;

  // Parse the enum constants.
  while (FormatTok->isNot(tok::eof)) {
    if (FormatTok->is(tok::l_brace)) {
      // Parse the constant's class body.
      parseBlock(/*MustBeDeclaration=*/true, /*AddLevels=*/1u,
                 /*MunchSemi=*/false);
    } else if (FormatTok->is(tok::l_paren)) {
      parseParens();
    } else if (FormatTok->is(tok::comma)) {
      nextToken();
      addUnwrappedLine();
    } else if (FormatTok->is(tok::semi)) {
      nextToken();
      addUnwrappedLine();
      break;
    } else if (FormatTok->is(tok::r_brace)) {
      addUnwrappedLine();
      break;
    } else {
      nextToken();
    }
  }

  // Parse the class body after the enum's ";" if any.
  parseLevel(OpeningBrace, /*CanContainBracedList=*/true);
  nextToken();
  --Line->Level;
  addUnwrappedLine();
}

void UnwrappedLineParser::parseRecord(bool ParseAsExpr) {
  const FormatToken &InitialToken = *FormatTok;
  nextToken();

  // The actual identifier can be a nested name specifier, and in macros
  // it is often token-pasted.
  // An [[attribute]] can be before the identifier.
  while (FormatTok->isOneOf(tok::identifier, tok::coloncolon, tok::hashhash,
                            tok::kw___attribute, tok::kw___declspec,
                            tok::kw_alignas, tok::l_square, tok::r_square) ||
         ((Style.Language == FormatStyle::LK_Java || Style.isJavaScript()) &&
          FormatTok->isOneOf(tok::period, tok::comma))) {
    if (Style.isJavaScript() &&
        FormatTok->isOneOf(Keywords.kw_extends, Keywords.kw_implements)) {
      // JavaScript/TypeScript supports inline object types in
      // extends/implements positions:
      //     class Foo implements {bar: number} { }
      nextToken();
      if (FormatTok->is(tok::l_brace)) {
        tryToParseBracedList();
        continue;
      }
    }
    bool IsNonMacroIdentifier =
        FormatTok->is(tok::identifier) &&
        FormatTok->TokenText != FormatTok->TokenText.upper();
    nextToken();
    // We can have macros or attributes in between 'class' and the class name.
    if (!IsNonMacroIdentifier) {
      if (FormatTok->is(tok::l_paren)) {
        parseParens();
      } else if (FormatTok->is(TT_AttributeSquare)) {
        parseSquare();
        // Consume the closing TT_AttributeSquare.
        if (FormatTok->Next && FormatTok->is(TT_AttributeSquare))
          nextToken();
      }
    }
  }

  // Note that parsing away template declarations here leads to incorrectly
  // accepting function declarations as record declarations.
  // In general, we cannot solve this problem. Consider:
  // class A<int> B() {}
  // which can be a function definition or a class definition when B() is a
  // macro. If we find enough real-world cases where this is a problem, we
  // can parse for the 'template' keyword in the beginning of the statement,
  // and thus rule out the record production in case there is no template
  // (this would still leave us with an ambiguity between template function
  // and class declarations).
  if (FormatTok->isOneOf(tok::colon, tok::less)) {
    do {
      if (FormatTok->is(tok::l_brace)) {
        calculateBraceTypes(/*ExpectClassBody=*/true);
        if (!tryToParseBracedList())
          break;
      }
      if (FormatTok->is(tok::l_square)) {
        FormatToken *Previous = FormatTok->Previous;
        if (!Previous ||
            !(Previous->is(tok::r_paren) || Previous->isTypeOrIdentifier())) {
          // Don't try parsing a lambda if we had a closing parenthesis before,
          // it was probably a pointer to an array: int (*)[].
          if (!tryToParseLambda())
            break;
        } else {
          parseSquare();
          continue;
        }
      }
      if (FormatTok->is(tok::semi))
        return;
      if (Style.isCSharp() && FormatTok->is(Keywords.kw_where)) {
        addUnwrappedLine();
        nextToken();
        parseCSharpGenericTypeConstraint();
        break;
      }
      nextToken();
    } while (!eof());
  }

  auto GetBraceType = [](const FormatToken &RecordTok) {
    switch (RecordTok.Tok.getKind()) {
    case tok::kw_class:
      return TT_ClassLBrace;
    case tok::kw_struct:
      return TT_StructLBrace;
    case tok::kw_union:
      return TT_UnionLBrace;
    default:
      // Useful for e.g. interface.
      return TT_RecordLBrace;
    }
  };
  if (FormatTok->is(tok::l_brace)) {
    FormatTok->setFinalizedType(GetBraceType(InitialToken));
    if (ParseAsExpr) {
      parseChildBlock();
    } else {
      if (ShouldBreakBeforeBrace(Style, InitialToken))
        addUnwrappedLine();

      unsigned AddLevels = Style.IndentAccessModifiers ? 2u : 1u;
      parseBlock(/*MustBeDeclaration=*/true, AddLevels, /*MunchSemi=*/false);
    }
  }
  // There is no addUnwrappedLine() here so that we fall through to parsing a
  // structural element afterwards. Thus, in "class A {} n, m;",
  // "} n, m;" will end up in one unwrapped line.
}

void UnwrappedLineParser::parseObjCMethod() {
  assert(FormatTok->isOneOf(tok::l_paren, tok::identifier) &&
         "'(' or identifier expected.");
  do {
    if (FormatTok->is(tok::semi)) {
      nextToken();
      addUnwrappedLine();
      return;
    } else if (FormatTok->is(tok::l_brace)) {
      if (Style.BraceWrapping.AfterFunction)
        addUnwrappedLine();
      parseBlock();
      addUnwrappedLine();
      return;
    } else {
      nextToken();
    }
  } while (!eof());
}

void UnwrappedLineParser::parseObjCProtocolList() {
  assert(FormatTok->is(tok::less) && "'<' expected.");
  do {
    nextToken();
    // Early exit in case someone forgot a close angle.
    if (FormatTok->isOneOf(tok::semi, tok::l_brace) ||
        FormatTok->isObjCAtKeyword(tok::objc_end))
      return;
  } while (!eof() && FormatTok->isNot(tok::greater));
  nextToken(); // Skip '>'.
}

void UnwrappedLineParser::parseObjCUntilAtEnd() {
  do {
    if (FormatTok->isObjCAtKeyword(tok::objc_end)) {
      nextToken();
      addUnwrappedLine();
      break;
    }
    if (FormatTok->is(tok::l_brace)) {
      parseBlock();
      // In ObjC interfaces, nothing should be following the "}".
      addUnwrappedLine();
    } else if (FormatTok->is(tok::r_brace)) {
      // Ignore stray "}". parseStructuralElement doesn't consume them.
      nextToken();
      addUnwrappedLine();
    } else if (FormatTok->isOneOf(tok::minus, tok::plus)) {
      nextToken();
      parseObjCMethod();
    } else {
      parseStructuralElement();
    }
  } while (!eof());
}

void UnwrappedLineParser::parseObjCInterfaceOrImplementation() {
  assert(FormatTok->Tok.getObjCKeywordID() == tok::objc_interface ||
         FormatTok->Tok.getObjCKeywordID() == tok::objc_implementation);
  nextToken();
  nextToken(); // interface name

  // @interface can be followed by a lightweight generic
  // specialization list, then either a base class or a category.
  if (FormatTok->is(tok::less))
    parseObjCLightweightGenerics();
  if (FormatTok->is(tok::colon)) {
    nextToken();
    nextToken(); // base class name
    // The base class can also have lightweight generics applied to it.
    if (FormatTok->is(tok::less))
      parseObjCLightweightGenerics();
  } else if (FormatTok->is(tok::l_paren))
    // Skip category, if present.
    parseParens();

  if (FormatTok->is(tok::less))
    parseObjCProtocolList();

  if (FormatTok->is(tok::l_brace)) {
    if (Style.BraceWrapping.AfterObjCDeclaration)
      addUnwrappedLine();
    parseBlock(/*MustBeDeclaration=*/true);
  }

  // With instance variables, this puts '}' on its own line.  Without instance
  // variables, this ends the @interface line.
  addUnwrappedLine();

  parseObjCUntilAtEnd();
}

void UnwrappedLineParser::parseObjCLightweightGenerics() {
  assert(FormatTok->is(tok::less));
  // Unlike protocol lists, generic parameterizations support
  // nested angles:
  //
  // @interface Foo<ValueType : id <NSCopying, NSSecureCoding>> :
  //     NSObject <NSCopying, NSSecureCoding>
  //
  // so we need to count how many open angles we have left.
  unsigned NumOpenAngles = 1;
  do {
    nextToken();
    // Early exit in case someone forgot a close angle.
    if (FormatTok->isOneOf(tok::semi, tok::l_brace) ||
        FormatTok->isObjCAtKeyword(tok::objc_end))
      break;
    if (FormatTok->is(tok::less))
      ++NumOpenAngles;
    else if (FormatTok->is(tok::greater)) {
      assert(NumOpenAngles > 0 && "'>' makes NumOpenAngles negative");
      --NumOpenAngles;
    }
  } while (!eof() && NumOpenAngles != 0);
  nextToken(); // Skip '>'.
}

// Returns true for the declaration/definition form of @protocol,
// false for the expression form.
bool UnwrappedLineParser::parseObjCProtocol() {
  assert(FormatTok->Tok.getObjCKeywordID() == tok::objc_protocol);
  nextToken();

  if (FormatTok->is(tok::l_paren))
    // The expression form of @protocol, e.g. "Protocol* p = @protocol(foo);".
    return false;

  // The definition/declaration form,
  // @protocol Foo
  // - (int)someMethod;
  // @end

  nextToken(); // protocol name

  if (FormatTok->is(tok::less))
    parseObjCProtocolList();

  // Check for protocol declaration.
  if (FormatTok->is(tok::semi)) {
    nextToken();
    addUnwrappedLine();
    return true;
  }

  addUnwrappedLine();
  parseObjCUntilAtEnd();
  return true;
}

void UnwrappedLineParser::parseJavaScriptEs6ImportExport() {
  bool IsImport = FormatTok->is(Keywords.kw_import);
  assert(IsImport || FormatTok->is(tok::kw_export));
  nextToken();

  // Consume the "default" in "export default class/function".
  if (FormatTok->is(tok::kw_default))
    nextToken();

  // Consume "async function", "function" and "default function", so that these
  // get parsed as free-standing JS functions, i.e. do not require a trailing
  // semicolon.
  if (FormatTok->is(Keywords.kw_async))
    nextToken();
  if (FormatTok->is(Keywords.kw_function)) {
    nextToken();
    return;
  }

  // For imports, `export *`, `export {...}`, consume the rest of the line up
  // to the terminating `;`. For everything else, just return and continue
  // parsing the structural element, i.e. the declaration or expression for
  // `export default`.
  if (!IsImport && !FormatTok->isOneOf(tok::l_brace, tok::star) &&
      !FormatTok->isStringLiteral())
    return;

  while (!eof()) {
    if (FormatTok->is(tok::semi))
      return;
    if (Line->Tokens.empty()) {
      // Common issue: Automatic Semicolon Insertion wrapped the line, so the
      // import statement should terminate.
      return;
    }
    if (FormatTok->is(tok::l_brace)) {
      FormatTok->setBlockKind(BK_Block);
      nextToken();
      parseBracedList();
    } else {
      nextToken();
    }
  }
}

void UnwrappedLineParser::parseStatementMacro() {
  nextToken();
  if (FormatTok->is(tok::l_paren))
    parseParens();
  if (FormatTok->is(tok::semi))
    nextToken();
  addUnwrappedLine();
}

LLVM_ATTRIBUTE_UNUSED static void printDebugInfo(const UnwrappedLine &Line,
                                                 StringRef Prefix = "") {
  llvm::dbgs() << Prefix << "Line(" << Line.Level
               << ", FSC=" << Line.FirstStartColumn << ")"
               << (Line.InPPDirective ? " MACRO" : "") << ": ";
  for (const auto &Node : Line.Tokens) {
    llvm::dbgs() << Node.Tok->Tok.getName() << "["
                 << "T=" << static_cast<unsigned>(Node.Tok->getType())
                 << ", OC=" << Node.Tok->OriginalColumn << "] ";
  }
  for (const auto &Node : Line.Tokens)
    for (const auto &ChildNode : Node.Children)
      printDebugInfo(ChildNode, "\nChild: ");

  llvm::dbgs() << "\n";
}

void UnwrappedLineParser::addUnwrappedLine(LineLevel AdjustLevel) {
  if (Line->Tokens.empty())
    return;
  LLVM_DEBUG({
    if (CurrentLines == &Lines)
      printDebugInfo(*Line);
  });

  // If this line closes a block when in Whitesmiths mode, remember that
  // information so that the level can be decreased after the line is added.
  // This has to happen after the addition of the line since the line itself
  // needs to be indented.
  bool ClosesWhitesmithsBlock =
      Line->MatchingOpeningBlockLineIndex != UnwrappedLine::kInvalidIndex &&
      Style.BreakBeforeBraces == FormatStyle::BS_Whitesmiths;

  CurrentLines->push_back(std::move(*Line));
  Line->Tokens.clear();
  Line->MatchingOpeningBlockLineIndex = UnwrappedLine::kInvalidIndex;
  Line->FirstStartColumn = 0;

  if (ClosesWhitesmithsBlock && AdjustLevel == LineLevel::Remove)
    --Line->Level;
  if (CurrentLines == &Lines && !PreprocessorDirectives.empty()) {
    CurrentLines->append(
        std::make_move_iterator(PreprocessorDirectives.begin()),
        std::make_move_iterator(PreprocessorDirectives.end()));
    PreprocessorDirectives.clear();
  }
  // Disconnect the current token from the last token on the previous line.
  FormatTok->Previous = nullptr;
}

bool UnwrappedLineParser::eof() const { return FormatTok->is(tok::eof); }

bool UnwrappedLineParser::isOnNewLine(const FormatToken &FormatTok) {
  return (Line->InPPDirective || FormatTok.HasUnescapedNewline) &&
         FormatTok.NewlinesBefore > 0;
}

// Checks if \p FormatTok is a line comment that continues the line comment
// section on \p Line.
static bool
continuesLineCommentSection(const FormatToken &FormatTok,
                            const UnwrappedLine &Line,
                            const llvm::Regex &CommentPragmasRegex) {
  if (Line.Tokens.empty())
    return false;

  StringRef IndentContent = FormatTok.TokenText;
  if (FormatTok.TokenText.startswith("//") ||
      FormatTok.TokenText.startswith("/*"))
    IndentContent = FormatTok.TokenText.substr(2);
  if (CommentPragmasRegex.match(IndentContent))
    return false;

  // If Line starts with a line comment, then FormatTok continues the comment
  // section if its original column is greater or equal to the original start
  // column of the line.
  //
  // Define the min column token of a line as follows: if a line ends in '{' or
  // contains a '{' followed by a line comment, then the min column token is
  // that '{'. Otherwise, the min column token of the line is the first token of
  // the line.
  //
  // If Line starts with a token other than a line comment, then FormatTok
  // continues the comment section if its original column is greater than the
  // original start column of the min column token of the line.
  //
  // For example, the second line comment continues the first in these cases:
  //
  // // first line
  // // second line
  //
  // and:
  //
  // // first line
  //  // second line
  //
  // and:
  //
  // int i; // first line
  //  // second line
  //
  // and:
  //
  // do { // first line
  //      // second line
  //   int i;
  // } while (true);
  //
  // and:
  //
  // enum {
  //   a, // first line
  //    // second line
  //   b
  // };
  //
  // The second line comment doesn't continue the first in these cases:
  //
  //   // first line
  //  // second line
  //
  // and:
  //
  // int i; // first line
  // // second line
  //
  // and:
  //
  // do { // first line
  //   // second line
  //   int i;
  // } while (true);
  //
  // and:
  //
  // enum {
  //   a, // first line
  //   // second line
  // };
  const FormatToken *MinColumnToken = Line.Tokens.front().Tok;

  // Scan for '{//'. If found, use the column of '{' as a min column for line
  // comment section continuation.
  const FormatToken *PreviousToken = nullptr;
  for (const UnwrappedLineNode &Node : Line.Tokens) {
    if (PreviousToken && PreviousToken->is(tok::l_brace) &&
        isLineComment(*Node.Tok)) {
      MinColumnToken = PreviousToken;
      break;
    }
    PreviousToken = Node.Tok;

    // Grab the last newline preceding a token in this unwrapped line.
    if (Node.Tok->NewlinesBefore > 0)
      MinColumnToken = Node.Tok;
  }
  if (PreviousToken && PreviousToken->is(tok::l_brace))
    MinColumnToken = PreviousToken;

  return continuesLineComment(FormatTok, /*Previous=*/Line.Tokens.back().Tok,
                              MinColumnToken);
}

void UnwrappedLineParser::flushComments(bool NewlineBeforeNext) {
  bool JustComments = Line->Tokens.empty();
  for (FormatToken *Tok : CommentsBeforeNextToken) {
    // Line comments that belong to the same line comment section are put on the
    // same line since later we might want to reflow content between them.
    // Additional fine-grained breaking of line comment sections is controlled
    // by the class BreakableLineCommentSection in case it is desirable to keep
    // several line comment sections in the same unwrapped line.
    //
    // FIXME: Consider putting separate line comment sections as children to the
    // unwrapped line instead.
    Tok->ContinuesLineCommentSection =
        continuesLineCommentSection(*Tok, *Line, CommentPragmasRegex);
    if (isOnNewLine(*Tok) && JustComments && !Tok->ContinuesLineCommentSection)
      addUnwrappedLine();
    pushToken(Tok);
  }
  if (NewlineBeforeNext && JustComments)
    addUnwrappedLine();
  CommentsBeforeNextToken.clear();
}

void UnwrappedLineParser::nextToken(int LevelDifference) {
  if (eof())
    return;
  flushComments(isOnNewLine(*FormatTok));
  pushToken(FormatTok);
  FormatToken *Previous = FormatTok;
  if (!Style.isJavaScript())
    readToken(LevelDifference);
  else
    readTokenWithJavaScriptASI();
  FormatTok->Previous = Previous;
}

void UnwrappedLineParser::distributeComments(
    const SmallVectorImpl<FormatToken *> &Comments,
    const FormatToken *NextTok) {
  // Whether or not a line comment token continues a line is controlled by
  // the method continuesLineCommentSection, with the following caveat:
  //
  // Define a trail of Comments to be a nonempty proper postfix of Comments such
  // that each comment line from the trail is aligned with the next token, if
  // the next token exists. If a trail exists, the beginning of the maximal
  // trail is marked as a start of a new comment section.
  //
  // For example in this code:
  //
  // int a; // line about a
  //   // line 1 about b
  //   // line 2 about b
  //   int b;
  //
  // the two lines about b form a maximal trail, so there are two sections, the
  // first one consisting of the single comment "// line about a" and the
  // second one consisting of the next two comments.
  if (Comments.empty())
    return;
  bool ShouldPushCommentsInCurrentLine = true;
  bool HasTrailAlignedWithNextToken = false;
  unsigned StartOfTrailAlignedWithNextToken = 0;
  if (NextTok) {
    // We are skipping the first element intentionally.
    for (unsigned i = Comments.size() - 1; i > 0; --i) {
      if (Comments[i]->OriginalColumn == NextTok->OriginalColumn) {
        HasTrailAlignedWithNextToken = true;
        StartOfTrailAlignedWithNextToken = i;
      }
    }
  }
  for (unsigned i = 0, e = Comments.size(); i < e; ++i) {
    FormatToken *FormatTok = Comments[i];
    if (HasTrailAlignedWithNextToken && i == StartOfTrailAlignedWithNextToken) {
      FormatTok->ContinuesLineCommentSection = false;
    } else {
      FormatTok->ContinuesLineCommentSection =
          continuesLineCommentSection(*FormatTok, *Line, CommentPragmasRegex);
    }
    if (!FormatTok->ContinuesLineCommentSection &&
        (isOnNewLine(*FormatTok) || FormatTok->IsFirst))
      ShouldPushCommentsInCurrentLine = false;
    if (ShouldPushCommentsInCurrentLine)
      pushToken(FormatTok);
    else
      CommentsBeforeNextToken.push_back(FormatTok);
  }
}

void UnwrappedLineParser::readToken(int LevelDifference) {
  SmallVector<FormatToken *, 1> Comments;
  bool PreviousWasComment = false;
  bool FirstNonCommentOnLine = false;
  do {
    FormatTok = Tokens->getNextToken();
    assert(FormatTok);
    while (FormatTok->getType() == TT_ConflictStart ||
           FormatTok->getType() == TT_ConflictEnd ||
           FormatTok->getType() == TT_ConflictAlternative) {
      if (FormatTok->getType() == TT_ConflictStart)
        conditionalCompilationStart(/*Unreachable=*/false);
      else if (FormatTok->getType() == TT_ConflictAlternative)
        conditionalCompilationAlternative();
      else if (FormatTok->getType() == TT_ConflictEnd)
        conditionalCompilationEnd();
      FormatTok = Tokens->getNextToken();
      FormatTok->MustBreakBefore = true;
    }

    auto IsFirstNonCommentOnLine = [](bool FirstNonCommentOnLine,
                                      const FormatToken &Tok,
                                      bool PreviousWasComment) {
      auto IsFirstOnLine = [](const FormatToken &Tok) {
        return Tok.HasUnescapedNewline || Tok.IsFirst;
      };

      // Consider preprocessor directives preceded by block comments as first
      // on line.
      if (PreviousWasComment)
        return FirstNonCommentOnLine || IsFirstOnLine(Tok);
      return IsFirstOnLine(Tok);
    };

    FirstNonCommentOnLine = IsFirstNonCommentOnLine(
        FirstNonCommentOnLine, *FormatTok, PreviousWasComment);
    PreviousWasComment = FormatTok->is(tok::comment);

    while (!Line->InPPDirective && FormatTok->is(tok::hash) &&
           FirstNonCommentOnLine) {
      distributeComments(Comments, FormatTok);
      Comments.clear();
      // If there is an unfinished unwrapped line, we flush the preprocessor
      // directives only after that unwrapped line was finished later.
      bool SwitchToPreprocessorLines = !Line->Tokens.empty();
      ScopedLineState BlockState(*this, SwitchToPreprocessorLines);
      assert((LevelDifference >= 0 ||
              static_cast<unsigned>(-LevelDifference) <= Line->Level) &&
             "LevelDifference makes Line->Level negative");
      Line->Level += LevelDifference;
      // Comments stored before the preprocessor directive need to be output
      // before the preprocessor directive, at the same level as the
      // preprocessor directive, as we consider them to apply to the directive.
      if (Style.IndentPPDirectives == FormatStyle::PPDIS_BeforeHash &&
          PPBranchLevel > 0)
        Line->Level += PPBranchLevel;
      flushComments(isOnNewLine(*FormatTok));
      parsePPDirective();
      PreviousWasComment = FormatTok->is(tok::comment);
      FirstNonCommentOnLine = IsFirstNonCommentOnLine(
          FirstNonCommentOnLine, *FormatTok, PreviousWasComment);
    }

    if (!PPStack.empty() && (PPStack.back().Kind == PP_Unreachable) &&
        !Line->InPPDirective)
      continue;

    if (!FormatTok->is(tok::comment)) {
      distributeComments(Comments, FormatTok);
      Comments.clear();
      return;
    }

    Comments.push_back(FormatTok);
  } while (!eof());

  distributeComments(Comments, nullptr);
  Comments.clear();
}

void UnwrappedLineParser::pushToken(FormatToken *Tok) {
  Line->Tokens.push_back(UnwrappedLineNode(Tok));
  if (MustBreakBeforeNextToken) {
    Line->Tokens.back().Tok->MustBreakBefore = true;
    MustBreakBeforeNextToken = false;
  }
}

} // end namespace format
} // end namespace clang
