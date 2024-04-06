//===----- SemaOpenACC.h - Semantic Analysis for OpenACC constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for OpenACC constructs and
/// clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAOPENACC_H
#define LLVM_CLANG_SEMA_SEMAOPENACC_H

#include "clang/AST/DeclGroup.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/SemaBase.h"

namespace clang {

class SemaOpenACC : public SemaBase {
public:
  SemaOpenACC(Sema &S);

  /// Called after parsing an OpenACC Clause so that it can be checked.
  bool ActOnClause(OpenACCClauseKind ClauseKind, SourceLocation StartLoc);

  /// Called after the construct has been parsed, but clauses haven't been
  /// parsed.  This allows us to diagnose not-implemented, as well as set up any
  /// state required for parsing the clauses.
  void ActOnConstruct(OpenACCDirectiveKind K, SourceLocation StartLoc);

  /// Called after the directive, including its clauses, have been parsed and
  /// parsing has consumed the 'annot_pragma_openacc_end' token. This DOES
  /// happen before any associated declarations or statements have been parsed.
  /// This function is only called when we are parsing a 'statement' context.
  bool ActOnStartStmtDirective(OpenACCDirectiveKind K, SourceLocation StartLoc);

  /// Called after the directive, including its clauses, have been parsed and
  /// parsing has consumed the 'annot_pragma_openacc_end' token. This DOES
  /// happen before any associated declarations or statements have been parsed.
  /// This function is only called when we are parsing a 'Decl' context.
  bool ActOnStartDeclDirective(OpenACCDirectiveKind K, SourceLocation StartLoc);
  /// Called when we encounter an associated statement for our construct, this
  /// should check legality of the statement as it appertains to this Construct.
  StmtResult ActOnAssociatedStmt(OpenACCDirectiveKind K, StmtResult AssocStmt);

  /// Called after the directive has been completely parsed, including the
  /// declaration group or associated statement.
  StmtResult ActOnEndStmtDirective(OpenACCDirectiveKind K,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc, StmtResult AssocStmt);
  /// Called after the directive has been completely parsed, including the
  /// declaration group or associated statement.
  DeclGroupRef ActOnEndDeclDirective();
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAOPENACC_H
