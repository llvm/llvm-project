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
  // 'atomic read', 'atomic write', 'atomic update', and 'atomic capture'.
  Atomic,
};

// Translate single-token string representations to the OpenACC Directive Kind.
// This doesn't completely comprehend 'Compound Constructs' (as it just
// identifies the first token), and doesn't fully handle 'enter data', 'exit
// data', nor any of the 'atomic' variants, just the first token of each.  So
// this should only be used by `ParseOpenACCDirectiveKind`.
OpenACCDirectiveKindEx GetOpenACCDirectiveKind(StringRef Name) {
  OpenACCDirectiveKind DirKind =
      llvm::StringSwitch<OpenACCDirectiveKind>(Name)
          .Case("parallel", OpenACCDirectiveKind::Parallel)
          .Case("serial", OpenACCDirectiveKind::Serial)
          .Case("kernels", OpenACCDirectiveKind::Kernels)
          .Case("data", OpenACCDirectiveKind::Data)
          .Case("host_data", OpenACCDirectiveKind::HostData)
          .Case("loop", OpenACCDirectiveKind::Loop)
          .Case("cache", OpenACCDirectiveKind::Cache)
          .Case("declare", OpenACCDirectiveKind::Declare)
          .Case("init", OpenACCDirectiveKind::Init)
          .Case("shutdown", OpenACCDirectiveKind::Shutdown)
          .Case("set", OpenACCDirectiveKind::Shutdown)
          .Case("update", OpenACCDirectiveKind::Update)
          .Case("wait", OpenACCDirectiveKind::Wait)
          .Case("routine", OpenACCDirectiveKind::Routine)
          .Default(OpenACCDirectiveKind::Invalid);

  if (DirKind != OpenACCDirectiveKind::Invalid)
    return static_cast<OpenACCDirectiveKindEx>(DirKind);

  return llvm::StringSwitch<OpenACCDirectiveKindEx>(Name)
      .Case("enter", OpenACCDirectiveKindEx::Enter)
      .Case("exit", OpenACCDirectiveKindEx::Exit)
      .Case("atomic", OpenACCDirectiveKindEx::Atomic)
      .Default(OpenACCDirectiveKindEx::Invalid);
}

// "enter data" and "exit data" are permitted as their own constructs. Handle
// these, knowing the previous token is either 'enter' or 'exit'. The current
// token should be the one after the "enter" or "exit".
OpenACCDirectiveKind
ParseOpenACCEnterExitDataDirective(Parser &P, Token FirstTok,
                                   StringRef FirstTokSpelling,
                                   OpenACCDirectiveKindEx ExtDirKind) {
  Token SecondTok = P.getCurToken();
  std::string SecondTokSpelling = P.getPreprocessor().getSpelling(SecondTok);

  if (SecondTokSpelling != "data") {
    P.Diag(FirstTok, diag::err_acc_invalid_directive)
        << 1 << FirstTokSpelling << SecondTokSpelling;
    return OpenACCDirectiveKind::Invalid;
  }

  P.ConsumeToken();
  return ExtDirKind == OpenACCDirectiveKindEx::Enter
             ? OpenACCDirectiveKind::EnterData
             : OpenACCDirectiveKind::ExitData;
}

OpenACCDirectiveKind ParseOpenACCAtomicDirective(Parser &P) {
  Token AtomicClauseToken = P.getCurToken();
  std::string AtomicClauseSpelling =
      P.getPreprocessor().getSpelling(AtomicClauseToken);

  OpenACCDirectiveKind DirKind =
      llvm::StringSwitch<OpenACCDirectiveKind>(AtomicClauseSpelling)
          .Case("read", OpenACCDirectiveKind::AtomicRead)
          .Case("write", OpenACCDirectiveKind::AtomicWrite)
          .Case("update", OpenACCDirectiveKind::AtomicUpdate)
          .Case("capture", OpenACCDirectiveKind::AtomicCapture)
          .Default(OpenACCDirectiveKind::Invalid);

  if (DirKind == OpenACCDirectiveKind::Invalid)
    P.Diag(AtomicClauseToken, diag::err_acc_invalid_atomic_clause)
        << AtomicClauseSpelling;

  P.ConsumeToken();
  return DirKind;
}

// Parse and consume the tokens for OpenACC Directive/Construct kinds.
OpenACCDirectiveKind ParseOpenACCDirectiveKind(Parser &P) {
  Token FirstTok = P.getCurToken();
  P.ConsumeToken();
  std::string FirstTokSpelling = P.getPreprocessor().getSpelling(FirstTok);

  OpenACCDirectiveKindEx ExDirKind = GetOpenACCDirectiveKind(FirstTokSpelling);

  Token SecondTok = P.getCurToken();
  // Go through the Extended kinds to see if we can convert this to the
  // non-Extended kinds, and handle invalid.
  switch (ExDirKind) {
  case OpenACCDirectiveKindEx::Invalid:
    P.Diag(FirstTok, diag::err_acc_invalid_directive) << 0 << FirstTokSpelling;
    return OpenACCDirectiveKind::Invalid;
  case OpenACCDirectiveKindEx::Enter:
  case OpenACCDirectiveKindEx::Exit:
    return ParseOpenACCEnterExitDataDirective(P, FirstTok, FirstTokSpelling,
                                              ExDirKind);
  case OpenACCDirectiveKindEx::Atomic:
    return ParseOpenACCAtomicDirective(P);
  }

  // Combined Constructs allows parallel loop, serial loop, or kernels loop. Any
  // other attempt at a combined constructwill be diagnosed as an invalid
  // clause.

  switch (static_cast<OpenACCDirectiveKind>(ExDirKind)) {
  default:
    break;
  case OpenACCDirectiveKind::Parallel:
    if (P.getPreprocessor().getSpelling(SecondTok) == "loop") {
      P.ConsumeToken();
      return OpenACCDirectiveKind::ParallelLoop;
    }
    break;
  case OpenACCDirectiveKind::Serial:
    if (P.getPreprocessor().getSpelling(SecondTok) == "loop") {
      P.ConsumeToken();
      return OpenACCDirectiveKind::SerialLoop;
    }
    break;
  case OpenACCDirectiveKind::Kernels:
    if (P.getPreprocessor().getSpelling(SecondTok) == "loop") {
      P.ConsumeToken();
      return OpenACCDirectiveKind::KernelsLoop;
    }
    break;
  }

  return static_cast<OpenACCDirectiveKind>(ExDirKind);
}

void ParseOpenACCClauseList(Parser &P) {
  // FIXME: In the future, we'll start parsing the clauses here, but for now we
  // haven't implemented that, so just emit the unimplemented diagnostic and
  // fail reasonably.
  if (P.getCurToken().isNot(tok::annot_pragma_openacc_end))
    P.Diag(P.getCurToken(), diag::warn_pragma_acc_unimplemented_clause_parsing);
}

void ParseOpenACCDirective(Parser &P) {
  OpenACCDirectiveKind DirKind = ParseOpenACCDirectiveKind(P);

  if (DirKind == OpenACCDirectiveKind::Invalid) {
    P.SkipUntil(tok::annot_pragma_openacc_end);
    return;
  }

  // We've successfully parsed the construct/directive name, a few require
  // special parsing which we will attempt, then revert to clause parsing.
  BalancedDelimiterTracker T(P, tok::l_paren, tok::annot_pragma_openacc_end);

  // Before we can parse clauses, there are a few directives that have special
  // cases that need parsing before clauses:
  //
  // 1- 'cache' doesn't take clauses, just a 'var list'.
  // 2- 'wait' has an optional wait-argument.
  // 3- 'routine' takes an optional 'name' before the clause list.
  //
  // As these are not implemented, diagnose, and continue.
  if (!T.consumeOpen()) {
    switch (DirKind) {
    default:
      P.Diag(T.getOpenLocation(), diag::err_acc_invalid_open_paren);
      break;
    case OpenACCDirectiveKind::Cache:
      P.Diag(T.getOpenLocation(),
             diag::warn_pragma_acc_unimplemented_construct_parens)
          << 0;
      break;
    case OpenACCDirectiveKind::Wait:
      P.Diag(T.getOpenLocation(),
             diag::warn_pragma_acc_unimplemented_construct_parens)
          << 1;
      break;
    case OpenACCDirectiveKind::Routine:
      P.Diag(T.getOpenLocation(),
             diag::warn_pragma_acc_unimplemented_construct_parens)
          << 2;
      break;
    }
    T.skipToEnd();
  }

  // Parses the list of clauses, if present.
  ParseOpenACCClauseList(P);

  P.Diag(P.getCurToken(), diag::warn_pragma_acc_unimplemented);
  P.SkipUntil(tok::annot_pragma_openacc_end);
}

} // namespace

// Parse OpenACC directive on a declaration.
Parser::DeclGroupPtrTy Parser::ParseOpenACCDirectiveDecl() {
  assert(Tok.is(tok::annot_pragma_openacc) && "expected OpenACC Start Token");

  ParsingOpenACCDirectiveRAII DirScope(*this);
  ConsumeAnnotationToken();

  ParseOpenACCDirective(*this);

  return nullptr;
}

// Parse OpenACC Directive on a Statement.
StmtResult Parser::ParseOpenACCDirectiveStmt() {
  assert(Tok.is(tok::annot_pragma_openacc) && "expected OpenACC Start Token");

  ParsingOpenACCDirectiveRAII DirScope(*this);
  ConsumeAnnotationToken();

  ParseOpenACCDirective(*this);

  return StmtEmpty();
}
