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
OpenACCDirectiveKindEx getOpenACCDirectiveKind(StringRef Name) {
  OpenACCDirectiveKind DirKind =
      llvm::StringSwitch<OpenACCDirectiveKind>(Name)
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
          .Default(OpenACCDirectiveKind::Invalid);

  if (DirKind != OpenACCDirectiveKind::Invalid)
    return static_cast<OpenACCDirectiveKindEx>(DirKind);

  return llvm::StringSwitch<OpenACCDirectiveKindEx>(Name)
      .Case("enter", OpenACCDirectiveKindEx::Enter)
      .Case("exit", OpenACCDirectiveKindEx::Exit)
      .Default(OpenACCDirectiveKindEx::Invalid);
}

// Since 'atomic' is effectively a compound directive, this will decode the
// second part of the directive.
OpenACCAtomicKind getOpenACCAtomicKind(StringRef Name) {
  return llvm::StringSwitch<OpenACCAtomicKind>(Name)
      .Case("read", OpenACCAtomicKind::Read)
      .Case("write", OpenACCAtomicKind::Write)
      .Case("update", OpenACCAtomicKind::Update)
      .Case("capture", OpenACCAtomicKind::Capture)
      .Default(OpenACCAtomicKind::Invalid);
}

bool isOpenACCDirectiveKind(OpenACCDirectiveKind Kind, StringRef Tok) {
  switch (Kind) {
  case OpenACCDirectiveKind::Parallel:
    return Tok == "parallel";
  case OpenACCDirectiveKind::Serial:
    return Tok == "serial";
  case OpenACCDirectiveKind::Kernels:
    return Tok == "kernels";
  case OpenACCDirectiveKind::Data:
    return Tok == "data";
  case OpenACCDirectiveKind::HostData:
    return Tok == "host_data";
  case OpenACCDirectiveKind::Loop:
    return Tok == "loop";
  case OpenACCDirectiveKind::Cache:
    return Tok == "cache";

  case OpenACCDirectiveKind::ParallelLoop:
  case OpenACCDirectiveKind::SerialLoop:
  case OpenACCDirectiveKind::KernelsLoop:
  case OpenACCDirectiveKind::EnterData:
  case OpenACCDirectiveKind::ExitData:
    return false;

  case OpenACCDirectiveKind::Atomic:
    return Tok == "atomic";
  case OpenACCDirectiveKind::Routine:
    return Tok == "routine";
  case OpenACCDirectiveKind::Declare:
    return Tok == "declare";
  case OpenACCDirectiveKind::Init:
    return Tok == "init";
  case OpenACCDirectiveKind::Shutdown:
    return Tok == "shutdown";
  case OpenACCDirectiveKind::Set:
    return Tok == "set";
  case OpenACCDirectiveKind::Update:
    return Tok == "update";
  case OpenACCDirectiveKind::Invalid:
    return false;
  }
  llvm_unreachable("Unknown 'Kind' Passed");
}

OpenACCDirectiveKind
ParseOpenACCEnterExitDataDirective(Parser &P, Token FirstTok,
                                   StringRef FirstTokSpelling,
                                   OpenACCDirectiveKindEx ExtDirKind) {
  Token SecondTok = P.getCurToken();

  if (SecondTok.isAnnotation()) {
    P.Diag(FirstTok, diag::err_acc_invalid_directive) << 0 << FirstTokSpelling;
    return OpenACCDirectiveKind::Invalid;
  }

  std::string SecondTokSpelling = P.getPreprocessor().getSpelling(SecondTok);

  if (!isOpenACCDirectiveKind(OpenACCDirectiveKind::Data, SecondTokSpelling)) {
    P.Diag(FirstTok, diag::err_acc_invalid_directive)
        << 1 << FirstTokSpelling << SecondTokSpelling;
    return OpenACCDirectiveKind::Invalid;
  }

  P.ConsumeToken();

  return ExtDirKind == OpenACCDirectiveKindEx::Enter
             ? OpenACCDirectiveKind::EnterData
             : OpenACCDirectiveKind::ExitData;
}

OpenACCAtomicKind ParseOpenACCAtomicKind(Parser &P) {
  Token AtomicClauseToken = P.getCurToken();

  // #pragma acc atomic is equivilent to update:
  if (AtomicClauseToken.isAnnotation())
    return OpenACCAtomicKind::Update;

  std::string AtomicClauseSpelling =
      P.getPreprocessor().getSpelling(AtomicClauseToken);
  OpenACCAtomicKind AtomicKind = getOpenACCAtomicKind(AtomicClauseSpelling);

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
  if (FirstTok.isAnnotation()) {
    P.Diag(FirstTok, diag::err_acc_missing_directive);
    return OpenACCDirectiveKind::Invalid;
  }

  P.ConsumeToken();
  std::string FirstTokSpelling = P.getPreprocessor().getSpelling(FirstTok);

  OpenACCDirectiveKindEx ExDirKind = getOpenACCDirectiveKind(FirstTokSpelling);

  // OpenACCDirectiveKindEx is meant to be an extended list
  // over OpenACCDirectiveKind, so any value below Invalid is one of the
  // OpenACCDirectiveKind values.  This switch takes care of all of the extra
  // parsing required for the Extended values.  At the end of this block,
  // ExDirKind can be assumed to be a valid OpenACCDirectiveKind, so we can
  // immediately cast it and use it as that.
  if (ExDirKind >= OpenACCDirectiveKindEx::Invalid) {
    switch (ExDirKind) {
    case OpenACCDirectiveKindEx::Invalid:
      P.Diag(FirstTok, diag::err_acc_invalid_directive)
          << 0 << FirstTokSpelling;
      return OpenACCDirectiveKind::Invalid;
    case OpenACCDirectiveKindEx::Enter:
    case OpenACCDirectiveKindEx::Exit:
      return ParseOpenACCEnterExitDataDirective(P, FirstTok, FirstTokSpelling,
                                                ExDirKind);
    }
  }

  OpenACCDirectiveKind DirKind = static_cast<OpenACCDirectiveKind>(ExDirKind);

  // Combined Constructs allows parallel loop, serial loop, or kernels loop. Any
  // other attempt at a combined construct will be diagnosed as an invalid
  // clause.
  Token SecondTok = P.getCurToken();
  if (!SecondTok.isAnnotation() &&
      isOpenACCDirectiveKind(OpenACCDirectiveKind::Loop,
                             P.getPreprocessor().getSpelling(SecondTok))) {
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

void ParseOpenACCClauseList(Parser &P) {
  // FIXME: In the future, we'll start parsing the clauses here, but for now we
  // haven't implemented that, so just emit the unimplemented diagnostic and
  // fail reasonably.
  if (P.getCurToken().isNot(tok::annot_pragma_openacc_end))
    P.Diag(P.getCurToken(), diag::warn_pragma_acc_unimplemented_clause_parsing);
}

} // namespace

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

void Parser::ParseOpenACCCacheVar() {
  ExprResult ArrayName = ParseOpenACCIDExpression();
  // FIXME: Pass this to Sema.
  (void)ArrayName;

  // If the expression is invalid, just continue parsing the brackets, there
  // is likely other useful diagnostics we can emit inside of those.

  BalancedDelimiterTracker SquareBrackets(*this, tok::l_square,
                                          tok::annot_pragma_openacc_end);

  // Square brackets are required, so error here, and try to recover by moving
  // until the next comma, or the close paren/end of pragma.
  if (SquareBrackets.expectAndConsume()) {
    SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openacc_end,
              Parser::StopBeforeMatch);
    return;
  }

  ExprResult Lower = getActions().CorrectDelayedTyposInExpr(ParseExpression());
  // FIXME: Pass this to Sema.
  (void)Lower;

  // The 'length' expression is optional, as this could be a single array
  // element. If there is no colon, we can treat it as that.
  if (getCurToken().is(tok::colon)) {
    ConsumeToken();
    ExprResult Length =
        getActions().CorrectDelayedTyposInExpr(ParseExpression());
    // FIXME: Pass this to Sema.
    (void)Length;
  }

  // Diagnose the square bracket being in the wrong place and continue.
  SquareBrackets.consumeClose();
}

void Parser::ParseOpenACCCacheVarList() {
  // If this is the end of the line, just return 'false' and count on the close
  // paren diagnostic to catch the issue.
  if (getCurToken().isAnnotation())
    return;

  // The VarList is an optional `readonly:` followed by a list of a variable
  // specifications.  First, see if we have `readonly:`, else we back-out and
  // treat it like the beginning of a reference to a potentially-existing
  // `readonly` variable.
  if (getPreprocessor().getSpelling(getCurToken()) == "readonly" &&
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
    ParseOpenACCCacheVar();
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
    }
  } else if (DirKind == OpenACCDirectiveKind::Cache) {
    // Cache's paren var-list is required, so error here if it isn't provided.
    // We know that the consumeOpen above left the first non-paren here, so use
    // expectAndConsume to emit the proper dialog, then continue.
    (void)T.expectAndConsume();
  }

  // Parses the list of clauses, if present.
  ParseOpenACCClauseList(*this);

  Diag(getCurToken(), diag::warn_pragma_acc_unimplemented);
  SkipUntil(tok::annot_pragma_openacc_end);
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
