//===--- SemaOpenACC.cpp - Semantic Analysis for OpenACC constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OpenACC constructs and
/// clauses.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace {
bool DiagnoseConstructAppertainment(Sema &S, OpenACCDirectiveKind K,
                                    SourceLocation StartLoc, bool IsStmt) {
  switch (K) {
  default:
  case OpenACCDirectiveKind::Invalid:
    // Nothing to do here, both invalid and unimplemented don't really need to
    // do anything.
    break;
  case OpenACCDirectiveKind::Parallel:
    if (!IsStmt)
      return S.Diag(StartLoc, diag::err_acc_construct_appertainment) << K;
    break;
  }
  return false;
}
} // namespace

bool Sema::ActOnOpenACCClause(OpenACCClauseKind ClauseKind,
                              SourceLocation StartLoc) {
  // TODO OpenACC: this will probably want to take the Directive Kind as well to
  // help with legalization.
  if (ClauseKind == OpenACCClauseKind::Invalid)
    return false;
  // For now just diagnose that it is unsupported and leave the parsing to do
  // whatever it can do. This function will eventually need to start returning
  // some sort of Clause AST type, but for now just return true/false based on
  // success.
  return Diag(StartLoc, diag::warn_acc_clause_unimplemented) << ClauseKind;
}

void Sema::ActOnOpenACCConstruct(OpenACCDirectiveKind K,
                                 SourceLocation StartLoc) {
  switch (K) {
  case OpenACCDirectiveKind::Invalid:
    // Nothing to do here, an invalid kind has nothing we can check here.  We
    // want to continue parsing clauses as far as we can, so we will just
    // ensure that we can still work and don't check any construct-specific
    // rules anywhere.
    break;
  case OpenACCDirectiveKind::Parallel:
    // Nothing to do here, there is no real legalization that needs to happen
    // here as these constructs do not take any arguments.
    break;
  default:
    Diag(StartLoc, diag::warn_acc_construct_unimplemented) << K;
    break;
  }
}

void Sema::ActOnStartOpenACCDeclDirective(OpenACCDirectiveKind K,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  // TODO OpenACC: This should likely return something with the modified
  // declaration. At the moment, only handle appertainment.
  DiagnoseConstructAppertainment(*this, K, StartLoc, /*IsStmt=*/false);
}

void Sema::ActOnEndOpenACCDeclDirective() {
  // TODO OpenACC: Should diagnose anything having to do with the associated
  // statement, or any clause diagnostics that can only be done at the 'end' of
  // the directive.  We should also close any 'block' marking now that the decl
  // parsing is complete.
}

StmtResult Sema::ActOnStartOpenACCStmtDirective(OpenACCDirectiveKind K,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc) {
  if (DiagnoseConstructAppertainment(*this, K, StartLoc, /*IsStmt=*/true))
    return StmtError();
  switch (K) {
  case OpenACCDirectiveKind::Invalid:
    return StmtError();
  default:
    return StmtEmpty();
  case OpenACCDirectiveKind::Parallel:
    return OpenACCComputeConstruct::Create(getASTContext(), K, StartLoc,
                                           EndLoc);
  }
  llvm_unreachable("Unhandled case in directive handling?");
}

StmtResult
Sema::ActOnOpenACCAssociatedStmt(OpenACCAssociatedStmtConstruct *Construct,
                                 Stmt *AssocStmt) {
  assert(Construct && AssocStmt && "Invalid construct or statement");
  switch (Construct->getDirectiveKind()) {
  case OpenACCDirectiveKind::Parallel:
    // There really isn't any checking here that could happen. As long as we
    // have a statement to associate, this should be fine.
    // OpenACC 3.3 Section 6:
    // Structured Block: in C or C++, an executable statement, possibly
    // compound, with a single entry at the top and a single exit at the
    // bottom.
    // FIXME: Should we reject DeclStmt's here? The standard isn't clear, and
    // an interpretation of it is to allow this and treat the initializer as
    // the 'structured block'.
    Context.setOpenACCStructuredBlock(cast<OpenACCComputeConstruct>(Construct),
                                      AssocStmt);
    break;
  default:
    llvm_unreachable("Unimplemented associated statement application");
  }
  // TODO: ERICH: Implement.
  return Construct;
}

void Sema::ActOnEndOpenACCStmtDirective(StmtResult Stmt) {
  // TODO OpenACC: Should diagnose anything having to do with the associated
  // statement, or any clause diagnostics that can only be done at the 'end' of
  // the directive. We should also close any 'block' marking now that the
  // statement parsing is complete.
}
