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

#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"

using namespace clang;

Parser::DeclGroupPtrTy Parser::ParseOpenACCDirectiveDecl() {
  Diag(Tok, diag::warn_pragma_acc_unimplemented);
  SkipUntil(tok::annot_pragma_openacc_end);
  return nullptr;
}
StmtResult Parser::ParseOpenACCDirectiveStmt() {
  Diag(Tok, diag::warn_pragma_acc_unimplemented);
  SkipUntil(tok::annot_pragma_openacc_end);
  return StmtEmpty();
}
