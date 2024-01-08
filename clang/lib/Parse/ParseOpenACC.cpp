//===--- ParseOpenACC.cpp - OpenACC-specific parsing support --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parsing logic for OpenACC language features.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/OpenACCKinds.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/RAIIObjectsForParser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace llvm;

namespace {
// An enum that contains the extended 'partial' parsed variants. This type
// should never escape the initial parse functionality, but is useful for
// simplifying the implementation.
enum class OpenACCDirectiveKindEx {
  Invalid = static_cast<int>(OpenACCDirectiveKind::Invalid),
  // 'enter data' and 'exit data'
  Enter,
  Exit,
};

// Translate single-token string representations to the OpenACC Directive Kind.
// This doesn't completely comprehend 'Compound Constructs' (as it just
// identifies the first token), and doesn't fully handle 'enter data', 'exit
// data', nor any of the 'atomic' variants, just the first token of each.  So
// this should only be used by `ParseOpenACCDirectiveKind`.
OpenACCDirectiveKindEx getOpenACCDirectiveKind(Token Tok) {
  if (!Tok.is(tok::identifier))
    return OpenACCDirectiveKindEx::Invalid;
  OpenACCDirectiveKind DirKind =
      llvm::StringSwitch<OpenACCDirectiveKind>(
          Tok.getIdentifierInfo()->getName())
          .Case("parallel", OpenACCDirectiveKind::Parallel)
          .Case("serial", OpenACCDirectiveKind::Serial)
          .Case("kernels", OpenACCDirectiveKind::Kernels)
          .Case("data", OpenACCDirectiveKind::Data)
          .Case("host_data", OpenACCDirectiveKind::HostData)
          .Case("loop", OpenACCDirectiveKind::Loop)
          .Case("cache", OpenACCDirectiveKind::Cache)
          .Case("atomic", OpenACCDirectiveKind::Atomic)
          .Case("routine", OpenACCDirectiveKind::Routine)
          .Case("declare", OpenACCDirectiveKind::Declare)
          .Case("init", OpenACCDirectiveKind::Init)
          .Case("shutdown", OpenACCDirectiveKind::Shutdown)
          .Case("set", OpenACCDirectiveKind::Shutdown)
          .Case("update", OpenACCDirectiveKind::Update)
          .Case("wait", OpenACCDirectiveKind::Wait)
          .Default(OpenACCDirectiveKind::Invalid);

  if (DirKind != OpenACCDirectiveKind::Invalid)
    return static_cast<OpenACCDirectiveKindEx>(DirKind);

  return llvm::StringSwitch<OpenACCDirectiveKindEx>(
             Tok.getIdentifierInfo()->getName())
      .Case("enter", OpenACCDirectiveKindEx::Enter)
      .Case("exit", OpenACCDirectiveKindEx::Exit)
      .Default(OpenACCDirectiveKindEx::Invalid);
}

// Translate single-token string representations to the OpenCC Clause Kind.
OpenACCClauseKind getOpenACCClauseKind(Token Tok) {
  // auto is a keyword in some language modes, so make sure we parse it
  // correctly.
  if (Tok.is(tok::kw_auto))
    return OpenACCClauseKind::Auto;

  // default is a keyword, so make sure we parse it correctly.
  if (Tok.is(tok::kw_default))
    return OpenACCClauseKind::Default;

  // if is also a keyword, make sure we parse it correctly.
  if (Tok.is(tok::kw_if))
    return OpenACCClauseKind::If;

  if (!Tok.is(tok::identifier))
    return OpenACCClauseKind::Invalid;

  return llvm::StringSwitch<OpenACCClauseKind>(
             Tok.getIdentifierInfo()->getName())
      .Case("auto", OpenACCClauseKind::Auto)
      .Case("default", OpenACCClauseKind::Default)
      .Case("finalize", OpenACCClauseKind::Finalize)
      .Case("if", OpenACCClauseKind::If)
      .Case("if_present", OpenACCClauseKind::IfPresent)
      .Case("independent", OpenACCClauseKind::Independent)
      .Case("nohost", OpenACCClauseKind::NoHost)
      .Case("self", OpenACCClauseKind::Self)
      .Case("seq", OpenACCClauseKind::Seq)
      .Case("vector", OpenACCClauseKind::Vector)
      .Case("worker", OpenACCClauseKind::Worker)
      .Default(OpenACCClauseKind::Invalid);
}

// Since 'atomic' is effectively a compound directive, this will decode the
// second part of the directive.
OpenACCAtomicKind getOpenACCAtomicKind(Token Tok) {
  if (!Tok.is(tok::identifier))
    return OpenACCAtomicKind::Invalid;
  return llvm::StringSwitch<OpenACCAtomicKind>(
             Tok.getIdentifierInfo()->getName())
      .Case("read", OpenACCAtomicKind::Read)
      .Case("write", OpenACCAtomicKind::Write)
      .Case("update", OpenACCAtomicKind::Update)
      .Case("capture", OpenACCAtomicKind::Capture)
      .Default(OpenACCAtomicKind::Invalid);
}

OpenACCDefaultClauseKind getOpenACCDefaultClauseKind(Token Tok) {
  if (!Tok.is(tok::identifier))
    return OpenACCDefaultClauseKind::Invalid;

  return llvm::StringSwitch<OpenACCDefaultClauseKind>(
             Tok.getIdentifierInfo()->getName())
      .Case("none", OpenACCDefaultClauseKind::None)
      .Case("present", OpenACCDefaultClauseKind::Present)
      .Default(OpenACCDefaultClauseKind::Invalid);
}

enum class OpenACCSpecialTokenKind {
  ReadOnly,
  DevNum,
  Queues,
};

bool isOpenACCSpecialToken(OpenACCSpecialTokenKind Kind, Token Tok) {
  if (!Tok.is(tok::identifier))
    return false;

  switch (Kind) {
  case OpenACCSpecialTokenKind::ReadOnly:
    return Tok.getIdentifierInfo()->isStr("readonly");
  case OpenACCSpecialTokenKind::DevNum:
    return Tok.getIdentifierInfo()->isStr("devnum");
  case OpenACCSpecialTokenKind::Queues:
    return Tok.getIdentifierInfo()->isStr("queues");
  }
  llvm_unreachable("Unknown 'Kind' Passed");
}

bool isOpenACCDirectiveKind(OpenACCDirectiveKind Kind, Token Tok) {
  if (!Tok.is(tok::identifier))
    return false;

  switch (Kind) {
  case OpenACCDirectiveKind::Parallel:
    return Tok.getIdentifierInfo()->isStr("parallel");
  case OpenACCDirectiveKind::Serial:
    return Tok.getIdentifierInfo()->isStr("serial");
  case OpenACCDirectiveKind::Kernels:
    return Tok.getIdentifierInfo()->isStr("kernels");
  case OpenACCDirectiveKind::Data:
    return Tok.getIdentifierInfo()->isStr("data");
  case OpenACCDirectiveKind::HostData:
    return Tok.getIdentifierInfo()->isStr("host_data");
  case OpenACCDirectiveKind::Loop:
    return Tok.getIdentifierInfo()->isStr("loop");
  case OpenACCDirectiveKind::Cache:
    return Tok.getIdentifierInfo()->isStr("cache");

  case OpenACCDirectiveKind::ParallelLoop:
  case OpenACCDirectiveKind::SerialLoop:
  case OpenACCDirectiveKind::KernelsLoop:
  case OpenACCDirectiveKind::EnterData:
  case OpenACCDirectiveKind::ExitData:
    return false;

  case OpenACCDirectiveKind::Atomic:
    return Tok.getIdentifierInfo()->isStr("atomic");
  case OpenACCDirectiveKind::Routine:
    return Tok.getIdentifierInfo()->isStr("routine");
  case OpenACCDirectiveKind::Declare:
    return Tok.getIdentifierInfo()->isStr("declare");
  case OpenACCDirectiveKind::Init:
    return Tok.getIdentifierInfo()->isStr("init");
  case OpenACCDirectiveKind::Shutdown:
    return Tok.getIdentifierInfo()->isStr("shutdown");
  case OpenACCDirectiveKind::Set:
    return Tok.getIdentifierInfo()->isStr("set");
  case OpenACCDirectiveKind::Update:
    return Tok.getIdentifierInfo()->isStr("update");
  case OpenACCDirectiveKind::Wait:
    return Tok.getIdentifierInfo()->isStr("wait");
  case OpenACCDirectiveKind::Invalid:
    return false;
  }
  llvm_unreachable("Unknown 'Kind' Passed");
}

/// Used for cases where we expect an identifier-like token, but don't want to
/// give awkward error messages in cases where it is accidentially a keyword.
bool expectIdentifierOrKeyword(Parser &P) {
  Token Tok = P.getCurToken();

  if (Tok.is(tok::identifier))
    return false;

  if (!Tok.isAnnotation() && Tok.getIdentifierInfo() &&
      Tok.getIdentifierInfo()->isKeyword(P.getLangOpts()))
    return false;

  P.Diag(P.getCurToken(), diag::err_expected) << tok::identifier;
  return true;
}

OpenACCDirectiveKind
ParseOpenACCEnterExitDataDirective(Parser &P, Token FirstTok,
                                   OpenACCDirectiveKindEx ExtDirKind) {
  Token SecondTok = P.getCurToken();

  if (SecondTok.isAnnotation()) {
    P.Diag(FirstTok, diag::err_acc_invalid_directive)
        << 0 << FirstTok.getIdentifierInfo();
    return OpenACCDirectiveKind::Invalid;
  }

  // Consume the second name anyway, this way we can continue on without making
  // this oddly look like a clause.
  P.ConsumeAnyToken();

  if (!isOpenACCDirectiveKind(OpenACCDirectiveKind::Data, SecondTok)) {
    if (!SecondTok.is(tok::identifier))
      P.Diag(SecondTok, diag::err_expected) << tok::identifier;
    else
      P.Diag(FirstTok, diag::err_acc_invalid_directive)
          << 1 << FirstTok.getIdentifierInfo()->getName()
          << SecondTok.getIdentifierInfo()->getName();
    return OpenACCDirectiveKind::Invalid;
  }

  return ExtDirKind == OpenACCDirectiveKindEx::Enter
             ? OpenACCDirectiveKind::EnterData
             : OpenACCDirectiveKind::ExitData;
}

OpenACCAtomicKind ParseOpenACCAtomicKind(Parser &P) {
  Token AtomicClauseToken = P.getCurToken();

  // #pragma acc atomic is equivilent to update:
  if (AtomicClauseToken.isAnnotation())
    return OpenACCAtomicKind::Update;

  OpenACCAtomicKind AtomicKind = getOpenACCAtomicKind(AtomicClauseToken);

  // If we don't know what this is, treat it as 'nothing', and treat the rest of
  // this as a clause list, which, despite being invalid, is likely what the
  // user was trying to do.
  if (AtomicKind == OpenACCAtomicKind::Invalid)
    return OpenACCAtomicKind::Update;

  P.ConsumeToken();
  return AtomicKind;
}

// Parse and consume the tokens for OpenACC Directive/Construct kinds.
OpenACCDirectiveKind ParseOpenACCDirectiveKind(Parser &P) {
  Token FirstTok = P.getCurToken();

  // Just #pragma acc can get us immediately to the end, make sure we don't
  // introspect on the spelling before then.
  if (FirstTok.isNot(tok::identifier)) {
    P.Diag(FirstTok, diag::err_acc_missing_directive);

    if (P.getCurToken().isNot(tok::annot_pragma_openacc_end))
      P.ConsumeAnyToken();

    return OpenACCDirectiveKind::Invalid;
  }

  P.ConsumeToken();

  OpenACCDirectiveKindEx ExDirKind = getOpenACCDirectiveKind(FirstTok);

  // OpenACCDirectiveKindEx is meant to be an extended list
  // over OpenACCDirectiveKind, so any value below Invalid is one of the
  // OpenACCDirectiveKind values.  This switch takes care of all of the extra
  // parsing required for the Extended values.  At the end of this block,
  // ExDirKind can be assumed to be a valid OpenACCDirectiveKind, so we can
  // immediately cast it and use it as that.
  if (ExDirKind >= OpenACCDirectiveKindEx::Invalid) {
    switch (ExDirKind) {
    case OpenACCDirectiveKindEx::Invalid: {
      P.Diag(FirstTok, diag::err_acc_invalid_directive)
          << 0 << FirstTok.getIdentifierInfo();
      return OpenACCDirectiveKind::Invalid;
    }
    case OpenACCDirectiveKindEx::Enter:
    case OpenACCDirectiveKindEx::Exit:
      return ParseOpenACCEnterExitDataDirective(P, FirstTok, ExDirKind);
    }
  }

  OpenACCDirectiveKind DirKind = static_cast<OpenACCDirectiveKind>(ExDirKind);

  // Combined Constructs allows parallel loop, serial loop, or kernels loop. Any
  // other attempt at a combined construct will be diagnosed as an invalid
  // clause.
  Token SecondTok = P.getCurToken();
  if (!SecondTok.isAnnotation() &&
      isOpenACCDirectiveKind(OpenACCDirectiveKind::Loop, SecondTok)) {
    switch (DirKind) {
    default:
      // Nothing to do except in the below cases, as they should be diagnosed as
      // a clause.
      break;
    case OpenACCDirectiveKind::Parallel:
      P.ConsumeToken();
      return OpenACCDirectiveKind::ParallelLoop;
    case OpenACCDirectiveKind::Serial:
      P.ConsumeToken();
      return OpenACCDirectiveKind::SerialLoop;
    case OpenACCDirectiveKind::Kernels:
      P.ConsumeToken();
      return OpenACCDirectiveKind::KernelsLoop;
    }
  }

  return DirKind;
}

bool ClauseHasOptionalParens(OpenACCClauseKind Kind) {
  return Kind == OpenACCClauseKind::Self;
}

bool ClauseHasRequiredParens(OpenACCClauseKind Kind) {
  return Kind == OpenACCClauseKind::Default || Kind == OpenACCClauseKind::If;
}

ExprResult ParseOpenACCConditionalExpr(Parser &P) {
  // FIXME: It isn't clear if the spec saying 'condition' means the same as
  // it does in an if/while/etc (See ParseCXXCondition), however as it was
  // written with Fortran/C in mind, we're going to assume it just means an
  // 'expression evaluating to boolean'.
  return P.getActions().CorrectDelayedTyposInExpr(P.ParseExpression());
}

bool ParseOpenACCClauseParams(Parser &P, OpenACCClauseKind Kind) {
  BalancedDelimiterTracker Parens(P, tok::l_paren,
                                  tok::annot_pragma_openacc_end);

  if (ClauseHasRequiredParens(Kind)) {
    if (Parens.expectAndConsume()) {
      // We are missing a paren, so assume that the person just forgot the
      // parameter.  Return 'false' so we try to continue on and parse the next
      // clause.
      P.SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openacc_end,
                  Parser::StopBeforeMatch);
      return false;
    }

    switch (Kind) {
    case OpenACCClauseKind::Default: {
      Token DefKindTok = P.getCurToken();

      if (expectIdentifierOrKeyword(P))
        break;

      P.ConsumeToken();

      if (getOpenACCDefaultClauseKind(DefKindTok) ==
          OpenACCDefaultClauseKind::Invalid)
        P.Diag(DefKindTok, diag::err_acc_invalid_default_clause_kind);

      break;
    }
    case OpenACCClauseKind::If: {
      ExprResult CondExpr = ParseOpenACCConditionalExpr(P);
      // An invalid expression can be just about anything, so just give up on
      // this clause list.
      if (CondExpr.isInvalid())
        return true;
      break;
    }
    default:
      llvm_unreachable("Not a required parens type?");
    }

    return Parens.consumeClose();
  } else if (ClauseHasOptionalParens(Kind)) {
    if (!Parens.consumeOpen()) {
      switch (Kind) {
      case OpenACCClauseKind::Self: {
        ExprResult CondExpr = ParseOpenACCConditionalExpr(P);
        // An invalid expression can be just about anything, so just give up on
        // this clause list.
        if (CondExpr.isInvalid())
          return true;
        break;
      }
      default:
        llvm_unreachable("Not an optional parens type?");
      }
      Parens.consumeClose();
    }
  }
  return false;
}

// The OpenACC Clause List is a comma or space-delimited list of clauses (see
// the comment on ParseOpenACCClauseList).  The concept of a 'clause' doesn't
// really have its owner grammar and each individual one has its own definition.
// However, they all are named with a single-identifier (or auto/default!)
// token, followed in some cases by either braces or parens.
bool ParseOpenACCClause(Parser &P) {
  // A number of clause names are actually keywords, so accept a keyword that
  // can be converted to a name.
  if (expectIdentifierOrKeyword(P))
    return true;

  OpenACCClauseKind Kind = getOpenACCClauseKind(P.getCurToken());

  if (Kind == OpenACCClauseKind::Invalid)
    return P.Diag(P.getCurToken(), diag::err_acc_invalid_clause)
           << P.getCurToken().getIdentifierInfo();

  // Consume the clause name.
  P.ConsumeToken();

  return ParseOpenACCClauseParams(P, Kind);
}

// Skip until we see the end of pragma token, but don't consume it. This is us
// just giving up on the rest of the pragma so we can continue executing. We
// have to do this because 'SkipUntil' considers paren balancing, which isn't
// what we want.
void SkipUntilEndOfDirective(Parser &P) {
  while (P.getCurToken().isNot(tok::annot_pragma_openacc_end))
    P.ConsumeAnyToken();
}

// OpenACC 3.3, section 1.7:
// To simplify the specification and convey appropriate constraint information,
// a pqr-list is a comma-separated list of pdr items. The one exception is a
// clause-list, which is a list of one or more clauses optionally separated by
// commas.
void ParseOpenACCClauseList(Parser &P) {
  bool FirstClause = true;
  while (P.getCurToken().isNot(tok::annot_pragma_openacc_end)) {
    // Comma is optional in a clause-list.
    if (!FirstClause && P.getCurToken().is(tok::comma))
      P.ConsumeToken();
    FirstClause = false;

    // Recovering from a bad clause is really difficult, so we just give up on
    // error.
    if (ParseOpenACCClause(P)) {
      SkipUntilEndOfDirective(P);
      return;
    }
  }
}

} // namespace

/// OpenACC 3.3, section 2.16:
/// In this section and throughout the specification, the term wait-argument
/// means:
/// [ devnum : int-expr : ] [ queues : ] async-argument-list
bool Parser::ParseOpenACCWaitArgument() {
  // [devnum : int-expr : ]
  if (isOpenACCSpecialToken(OpenACCSpecialTokenKind::DevNum, Tok) &&
      NextToken().is(tok::colon)) {
    // Consume devnum.
    ConsumeToken();
    // Consume colon.
    ConsumeToken();

    ExprResult IntExpr =
        getActions().CorrectDelayedTyposInExpr(ParseAssignmentExpression());
    if (IntExpr.isInvalid())
      return true;

    if (ExpectAndConsume(tok::colon))
      return true;
  }

  // [ queues : ]
  if (isOpenACCSpecialToken(OpenACCSpecialTokenKind::Queues, Tok) &&
      NextToken().is(tok::colon)) {
    // Consume queues.
    ConsumeToken();
    // Consume colon.
    ConsumeToken();
  }

  // OpenACC 3.3, section 2.16:
  // the term 'async-argument' means a nonnegative scalar integer expression, or
  // one of the special values 'acc_async_noval' or 'acc_async_sync', as defined
  // in the C header file and the Fortran opacc module.
  //
  // We are parsing this simply as list of assignment expressions (to avoid
  // comma being troublesome), and will ensure it is an integral type.  The
  // 'special' types are defined as macros, so we can't really check those
  // (other than perhaps as values at one point?), but the standard does say it
  // is implementation-defined to use any other negative value.
  //
  //
  bool FirstArg = true;
  while (!getCurToken().isOneOf(tok::r_paren, tok::annot_pragma_openacc_end)) {
    if (!FirstArg) {
      if (ExpectAndConsume(tok::comma))
        return true;
    }
    FirstArg = false;

    ExprResult CurArg =
        getActions().CorrectDelayedTyposInExpr(ParseAssignmentExpression());

    if (CurArg.isInvalid())
      return true;
  }

  return false;
}

ExprResult Parser::ParseOpenACCIDExpression() {
  ExprResult Res;
  if (getLangOpts().CPlusPlus) {
    Res = ParseCXXIdExpression(/*isAddressOfOperand=*/false);
  } else {
    // There isn't anything quite the same as ParseCXXIdExpression for C, so we
    // need to get the identifier, then call into Sema ourselves.

    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected) << tok::identifier;
      return ExprError();
    }

    Token FuncName = getCurToken();
    UnqualifiedId Name;
    CXXScopeSpec ScopeSpec;
    SourceLocation TemplateKWLoc;
    Name.setIdentifier(FuncName.getIdentifierInfo(), ConsumeToken());

    // Ensure this is a valid identifier. We don't accept causing implicit
    // function declarations per the spec, so always claim to not have trailing
    // L Paren.
    Res = Actions.ActOnIdExpression(getCurScope(), ScopeSpec, TemplateKWLoc,
                                    Name, /*HasTrailingLParen=*/false,
                                    /*isAddressOfOperand=*/false);
  }

  return getActions().CorrectDelayedTyposInExpr(Res);
}

/// OpenACC 3.3, section 2.10:
/// A 'var' in a cache directive must be a single array element or a simple
/// subarray.  In C and C++, a simple subarray is an array name followed by an
/// extended array range specification in brackets, with a start and length such
/// as:
///
/// arr[lower:length]
///
bool Parser::ParseOpenACCCacheVar() {
  ExprResult ArrayName = ParseOpenACCIDExpression();
  if (ArrayName.isInvalid())
    return true;

  // If the expression is invalid, just continue parsing the brackets, there
  // is likely other useful diagnostics we can emit inside of those.

  BalancedDelimiterTracker SquareBrackets(*this, tok::l_square,
                                          tok::annot_pragma_openacc_end);

  // Square brackets are required, so error here, and try to recover by moving
  // until the next comma, or the close paren/end of pragma.
  if (SquareBrackets.expectAndConsume()) {
    SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openacc_end,
              Parser::StopBeforeMatch);
    return true;
  }

  ExprResult Lower = getActions().CorrectDelayedTyposInExpr(ParseExpression());
  if (Lower.isInvalid())
    return true;

  // The 'length' expression is optional, as this could be a single array
  // element. If there is no colon, we can treat it as that.
  if (getCurToken().is(tok::colon)) {
    ConsumeToken();
    ExprResult Length =
        getActions().CorrectDelayedTyposInExpr(ParseExpression());
    if (Length.isInvalid())
      return true;
  }

  // Diagnose the square bracket being in the wrong place and continue.
  return SquareBrackets.consumeClose();
}

/// OpenACC 3.3, section 2.10:
/// In C and C++, the syntax of the cache directive is:
///
/// #pragma acc cache ([readonly:]var-list) new-line
void Parser::ParseOpenACCCacheVarList() {
  // If this is the end of the line, just return 'false' and count on the close
  // paren diagnostic to catch the issue.
  if (getCurToken().isAnnotation())
    return;

  // The VarList is an optional `readonly:` followed by a list of a variable
  // specifications.  First, see if we have `readonly:`, else we back-out and
  // treat it like the beginning of a reference to a potentially-existing
  // `readonly` variable.
  if (isOpenACCSpecialToken(OpenACCSpecialTokenKind::ReadOnly, Tok) &&
      NextToken().is(tok::colon)) {
    // Consume both tokens.
    ConsumeToken();
    ConsumeToken();
    // FIXME: Record that this is a 'readonly' so that we can use that during
    // Sema/AST generation.
  }

  bool FirstArray = true;
  while (!getCurToken().isOneOf(tok::r_paren, tok::annot_pragma_openacc_end)) {
    if (!FirstArray)
      ExpectAndConsume(tok::comma);
    FirstArray = false;
    if (ParseOpenACCCacheVar())
      SkipUntil(tok::r_paren, tok::annot_pragma_openacc_end, tok::comma,
                StopBeforeMatch);
  }
}

void Parser::ParseOpenACCDirective() {
  OpenACCDirectiveKind DirKind = ParseOpenACCDirectiveKind(*this);

  // Once we've parsed the construct/directive name, some have additional
  // specifiers that need to be taken care of. Atomic has an 'atomic-clause'
  // that needs to be parsed.
  if (DirKind == OpenACCDirectiveKind::Atomic)
    ParseOpenACCAtomicKind(*this);

  // We've successfully parsed the construct/directive name, however a few of
  // the constructs have optional parens that contain further details.
  BalancedDelimiterTracker T(*this, tok::l_paren,
                             tok::annot_pragma_openacc_end);

  if (!T.consumeOpen()) {
    switch (DirKind) {
    default:
      Diag(T.getOpenLocation(), diag::err_acc_invalid_open_paren);
      T.skipToEnd();
      break;
    case OpenACCDirectiveKind::Routine: {
      // Routine has an optional paren-wrapped name of a function in the local
      // scope. We parse the name, emitting any diagnostics
      ExprResult RoutineName = ParseOpenACCIDExpression();
      // If the routine name is invalid, just skip until the closing paren to
      // recover more gracefully.
      if (RoutineName.isInvalid())
        T.skipToEnd();
      else
        T.consumeClose();
      break;
    }
    case OpenACCDirectiveKind::Cache:
      ParseOpenACCCacheVarList();
      // The ParseOpenACCCacheVarList function manages to recover from failures,
      // so we can always consume the close.
      T.consumeClose();
      break;
    case OpenACCDirectiveKind::Wait:
      // OpenACC has an optional paren-wrapped 'wait-argument'.
      if (ParseOpenACCWaitArgument())
        T.skipToEnd();
      else
        T.consumeClose();
      break;
    }
  } else if (DirKind == OpenACCDirectiveKind::Cache) {
    // Cache's paren var-list is required, so error here if it isn't provided.
    // We know that the consumeOpen above left the first non-paren here, so
    // diagnose, then continue as if it was completely omitted.
    Diag(Tok, diag::err_expected) << tok::l_paren;
  }

  // Parses the list of clauses, if present.
  ParseOpenACCClauseList(*this);

  Diag(getCurToken(), diag::warn_pragma_acc_unimplemented);
  assert(Tok.is(tok::annot_pragma_openacc_end) &&
         "Didn't parse all OpenACC Clauses");
  ConsumeAnnotationToken();
}

// Parse OpenACC directive on a declaration.
Parser::DeclGroupPtrTy Parser::ParseOpenACCDirectiveDecl() {
  assert(Tok.is(tok::annot_pragma_openacc) && "expected OpenACC Start Token");

  ParsingOpenACCDirectiveRAII DirScope(*this);
  ConsumeAnnotationToken();

  ParseOpenACCDirective();

  return nullptr;
}

// Parse OpenACC Directive on a Statement.
StmtResult Parser::ParseOpenACCDirectiveStmt() {
  assert(Tok.is(tok::annot_pragma_openacc) && "expected OpenACC Start Token");

  ParsingOpenACCDirectiveRAII DirScope(*this);
  ConsumeAnnotationToken();

  ParseOpenACCDirective();

  return StmtEmpty();
}
