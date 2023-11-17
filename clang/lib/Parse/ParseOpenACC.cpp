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

/// This doesn't completely comprehend 'Compound Constructs' (as it just
/// identifies the first token) just the first token of each.  So
/// this should only be used by `ParseOpenACCDirectiveKind`.
OpenACCDirectiveKind getOpenACCDirectiveKind(StringRef Name) {
  return llvm::StringSwitch<OpenACCDirectiveKind>(Name)
      .Case("parallel", OpenACCDirectiveKind::Parallel)
      .Case("serial", OpenACCDirectiveKind::Serial)
      .Case("kernels", OpenACCDirectiveKind::Kernels)
      .Case("data", OpenACCDirectiveKind::Data)
      .Case("host_data", OpenACCDirectiveKind::HostData)
      .Case("loop", OpenACCDirectiveKind::Loop)
      .Case("declare", OpenACCDirectiveKind::Declare)
      .Case("init", OpenACCDirectiveKind::Init)
      .Case("shutdown", OpenACCDirectiveKind::Shutdown)
      .Case("set", OpenACCDirectiveKind::Shutdown)
      .Case("update", OpenACCDirectiveKind::Update)
      .Default(OpenACCDirectiveKind::Invalid);
}

// Parse and consume the tokens for OpenACC Directive/Construct kinds.
OpenACCDirectiveKind ParseOpenACCDirectiveKind(Parser &P) {
  Token FirstTok = P.getCurToken();
  P.ConsumeToken();
  std::string FirstTokSpelling = P.getPreprocessor().getSpelling(FirstTok);

  OpenACCDirectiveKind DirKind = getOpenACCDirectiveKind(FirstTokSpelling);

  if (DirKind == OpenACCDirectiveKind::Invalid)
    P.Diag(FirstTok, diag::err_acc_invalid_directive) << FirstTokSpelling;

  // Combined Constructs allows parallel loop, serial loop, or kernels loop. Any
  // other attempt at a combined construct will be diagnosed as an invalid
  // clause.
  Token SecondTok = P.getCurToken();
  if (!SecondTok.isAnnotation() &&
      P.getPreprocessor().getSpelling(SecondTok) == "loop") {
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

void ParseOpenACCDirective(Parser &P) {
  ParseOpenACCDirectiveKind(P);

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
