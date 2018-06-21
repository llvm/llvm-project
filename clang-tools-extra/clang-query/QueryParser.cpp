//===---- QueryParser.cpp - clang-query command parser --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "QueryParser.h"
#include "Query.h"
#include "QuerySession.h"
#include "clang/ASTMatchers/Dynamic/Parser.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include <set>

using namespace llvm;
using namespace clang::ast_matchers::dynamic;

namespace clang {
namespace query {

// Lex any amount of whitespace followed by a "word" (any sequence of
// non-whitespace characters) from the start of region [Begin,End).  If no word
// is found before End, return StringRef().  Begin is adjusted to exclude the
// lexed region.
StringRef QueryParser::lexWord() {
  while (true) {
    if (Begin == End)
      return StringRef(Begin, 0);

    if (!isWhitespace(*Begin))
      break;

    ++Begin;
  }

  const char *WordBegin = Begin;

  while (true) {
    ++Begin;

    if (Begin == End || isWhitespace(*Begin))
      return StringRef(WordBegin, Begin - WordBegin);
  }
}

// This is the StringSwitch-alike used by lexOrCompleteWord below. See that
// function for details.
template <typename T> struct QueryParser::LexOrCompleteWord {
  StringRef Word;
  StringSwitch<T> Switch;

  QueryParser *P;
  // Set to the completion point offset in Word, or StringRef::npos if
  // completion point not in Word.
  size_t WordCompletionPos;

  // Lexes a word and stores it in Word. Returns a LexOrCompleteWord<T> object
  // that can be used like a llvm::StringSwitch<T>, but adds cases as possible
  // completions if the lexed word contains the completion point.
  LexOrCompleteWord(QueryParser *P, StringRef &OutWord)
      : Word(P->lexWord()), Switch(Word), P(P),
        WordCompletionPos(StringRef::npos) {
    OutWord = Word;
    if (P->CompletionPos && P->CompletionPos <= Word.data() + Word.size()) {
      if (P->CompletionPos < Word.data())
        WordCompletionPos = 0;
      else
        WordCompletionPos = P->CompletionPos - Word.data();
    }
  }

  LexOrCompleteWord &Case(llvm::StringLiteral CaseStr, const T &Value,
                          bool IsCompletion = true) {

    if (WordCompletionPos == StringRef::npos)
      Switch.Case(CaseStr, Value);
    else if (CaseStr.size() != 0 && IsCompletion && WordCompletionPos <= CaseStr.size() &&
             CaseStr.substr(0, WordCompletionPos) ==
                 Word.substr(0, WordCompletionPos))
      P->Completions.push_back(LineEditor::Completion(
          (CaseStr.substr(WordCompletionPos) + " ").str(), CaseStr));
    return *this;
  }

  T Default(T Value) { return Switch.Default(Value); }
};

QueryRef QueryParser::parseSetBool(bool QuerySession::*Var) {
  StringRef ValStr;
  unsigned Value = LexOrCompleteWord<unsigned>(this, ValStr)
                       .Case("false", 0)
                       .Case("true", 1)
                       .Default(~0u);
  if (Value == ~0u) {
    return new InvalidQuery("expected 'true' or 'false', got '" + ValStr + "'");
  }
  return new SetQuery<bool>(Var, Value);
}

QueryRef QueryParser::parseSetOutputKind() {
  StringRef ValStr;
  unsigned OutKind = LexOrCompleteWord<unsigned>(this, ValStr)
                         .Case("diag", OK_Diag)
                         .Case("print", OK_Print)
                         .Case("dump", OK_Dump)
                         .Default(~0u);
  if (OutKind == ~0u) {
    return new InvalidQuery("expected 'diag', 'print' or 'dump', got '" +
                            ValStr + "'");
  }
  return new SetQuery<OutputKind>(&QuerySession::OutKind, OutputKind(OutKind));
}

QueryRef QueryParser::endQuery(QueryRef Q) {
  const char *Extra = Begin;
  if (!lexWord().empty())
    return new InvalidQuery("unexpected extra input: '" +
                            StringRef(Extra, End - Extra) + "'");
  return Q;
}

namespace {

enum ParsedQueryKind {
  PQK_Invalid,
  PQK_NoOp,
  PQK_Help,
  PQK_Let,
  PQK_Match,
  PQK_Set,
  PQK_Unlet,
  PQK_Quit
};

enum ParsedQueryVariable { PQV_Invalid, PQV_Output, PQV_BindRoot };

QueryRef makeInvalidQueryFromDiagnostics(const Diagnostics &Diag) {
  std::string ErrStr;
  llvm::raw_string_ostream OS(ErrStr);
  Diag.printToStreamFull(OS);
  return new InvalidQuery(OS.str());
}

} // namespace

QueryRef QueryParser::completeMatcherExpression() {
  std::vector<MatcherCompletion> Comps = Parser::completeExpression(
      StringRef(Begin, End - Begin), CompletionPos - Begin, nullptr,
      &QS.NamedValues);
  for (auto I = Comps.begin(), E = Comps.end(); I != E; ++I) {
    Completions.push_back(LineEditor::Completion(I->TypedText, I->MatcherDecl));
  }
  return QueryRef();
}

QueryRef QueryParser::doParse() {
  StringRef CommandStr;
  ParsedQueryKind QKind = LexOrCompleteWord<ParsedQueryKind>(this, CommandStr)
                              .Case("", PQK_NoOp)
                              .Case("help", PQK_Help)
                              .Case("m", PQK_Match, /*IsCompletion=*/false)
                              .Case("let", PQK_Let)
                              .Case("match", PQK_Match)
                              .Case("set", PQK_Set)
                              .Case("unlet", PQK_Unlet)
                              .Case("quit", PQK_Quit)
                              .Default(PQK_Invalid);

  switch (QKind) {
  case PQK_NoOp:
    return new NoOpQuery;

  case PQK_Help:
    return endQuery(new HelpQuery);

  case PQK_Quit:
    return endQuery(new QuitQuery);

  case PQK_Let: {
    StringRef Name = lexWord();

    if (Name.empty())
      return new InvalidQuery("expected variable name");

    if (CompletionPos)
      return completeMatcherExpression();

    Diagnostics Diag;
    ast_matchers::dynamic::VariantValue Value;
    if (!Parser::parseExpression(StringRef(Begin, End - Begin), nullptr,
                                 &QS.NamedValues, &Value, &Diag)) {
      return makeInvalidQueryFromDiagnostics(Diag);
    }

    return new LetQuery(Name, Value);
  }

  case PQK_Match: {
    if (CompletionPos)
      return completeMatcherExpression();

    Diagnostics Diag;
    Optional<DynTypedMatcher> Matcher = Parser::parseMatcherExpression(
        StringRef(Begin, End - Begin), nullptr, &QS.NamedValues, &Diag);
    if (!Matcher) {
      return makeInvalidQueryFromDiagnostics(Diag);
    }
    return new MatchQuery(*Matcher);
  }

  case PQK_Set: {
    StringRef VarStr;
    ParsedQueryVariable Var = LexOrCompleteWord<ParsedQueryVariable>(this,
                                                                     VarStr)
                                  .Case("output", PQV_Output)
                                  .Case("bind-root", PQV_BindRoot)
                                  .Default(PQV_Invalid);
    if (VarStr.empty())
      return new InvalidQuery("expected variable name");
    if (Var == PQV_Invalid)
      return new InvalidQuery("unknown variable: '" + VarStr + "'");

    QueryRef Q;
    switch (Var) {
    case PQV_Output:
      Q = parseSetOutputKind();
      break;
    case PQV_BindRoot:
      Q = parseSetBool(&QuerySession::BindRoot);
      break;
    case PQV_Invalid:
      llvm_unreachable("Invalid query kind");
    }

    return endQuery(Q);
  }

  case PQK_Unlet: {
    StringRef Name = lexWord();

    if (Name.empty())
      return new InvalidQuery("expected variable name");

    return endQuery(new LetQuery(Name, VariantValue()));
  }

  case PQK_Invalid:
    return new InvalidQuery("unknown command: " + CommandStr);
  }

  llvm_unreachable("Invalid query kind");
}

QueryRef QueryParser::parse(StringRef Line, const QuerySession &QS) {
  return QueryParser(Line, QS).doParse();
}

std::vector<LineEditor::Completion>
QueryParser::complete(StringRef Line, size_t Pos, const QuerySession &QS) {
  QueryParser P(Line, QS);
  P.CompletionPos = Line.data() + Pos;

  P.doParse();
  return P.Completions;
}

} // namespace query
} // namespace clang
