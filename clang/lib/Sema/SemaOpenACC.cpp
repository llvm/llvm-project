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

#include "clang/Sema/SemaOpenACC.h"
#include "clang/AST/StmtOpenACC.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Casting.h"

using namespace clang;

namespace {
bool diagnoseConstructAppertainment(SemaOpenACC &S, OpenACCDirectiveKind K,
                                    SourceLocation StartLoc, bool IsStmt) {
  switch (K) {
  default:
  case OpenACCDirectiveKind::Invalid:
    // Nothing to do here, both invalid and unimplemented don't really need to
    // do anything.
    break;
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::Kernels:
    if (!IsStmt)
      return S.Diag(StartLoc, diag::err_acc_construct_appertainment) << K;
    break;
  }
  return false;
}

bool doesClauseApplyToDirective(OpenACCDirectiveKind DirectiveKind,
                                OpenACCClauseKind ClauseKind) {
  switch (ClauseKind) {
    // FIXME: For each clause as we implement them, we can add the
    // 'legalization' list here.
  case OpenACCClauseKind::Default:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
    case OpenACCDirectiveKind::Data:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::If:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Data:
    case OpenACCDirectiveKind::EnterData:
    case OpenACCDirectiveKind::ExitData:
    case OpenACCDirectiveKind::HostData:
    case OpenACCDirectiveKind::Init:
    case OpenACCDirectiveKind::Shutdown:
    case OpenACCDirectiveKind::Set:
    case OpenACCDirectiveKind::Update:
    case OpenACCDirectiveKind::Wait:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  case OpenACCClauseKind::Self:
    switch (DirectiveKind) {
    case OpenACCDirectiveKind::Parallel:
    case OpenACCDirectiveKind::Serial:
    case OpenACCDirectiveKind::Kernels:
    case OpenACCDirectiveKind::Update:
    case OpenACCDirectiveKind::ParallelLoop:
    case OpenACCDirectiveKind::SerialLoop:
    case OpenACCDirectiveKind::KernelsLoop:
      return true;
    default:
      return false;
    }
  default:
    // Do nothing so we can go to the 'unimplemented' diagnostic instead.
    return true;
  }
  llvm_unreachable("Invalid clause kind");
}

bool checkAlreadyHasClauseOfKind(
    SemaOpenACC &S, ArrayRef<const OpenACCClause *> ExistingClauses,
    SemaOpenACC::OpenACCParsedClause &Clause) {
  const auto *Itr = llvm::find_if(ExistingClauses, [&](const OpenACCClause *C) {
    return C->getClauseKind() == Clause.getClauseKind();
  });
  if (Itr != ExistingClauses.end()) {
    S.Diag(Clause.getBeginLoc(), diag::err_acc_duplicate_clause_disallowed)
        << Clause.getDirectiveKind() << Clause.getClauseKind();
    S.Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
    return true;
  }
  return false;
}

} // namespace

SemaOpenACC::SemaOpenACC(Sema &S) : SemaBase(S) {}

OpenACCClause *
SemaOpenACC::ActOnClause(ArrayRef<const OpenACCClause *> ExistingClauses,
                         OpenACCParsedClause &Clause) {
  if (Clause.getClauseKind() == OpenACCClauseKind::Invalid)
    return nullptr;

  // Diagnose that we don't support this clause on this directive.
  if (!doesClauseApplyToDirective(Clause.getDirectiveKind(),
                                  Clause.getClauseKind())) {
    Diag(Clause.getBeginLoc(), diag::err_acc_clause_appertainment)
        << Clause.getDirectiveKind() << Clause.getClauseKind();
    return nullptr;
  }

  switch (Clause.getClauseKind()) {
  case OpenACCClauseKind::Default: {
    // Restrictions only properly implemented on 'compute' constructs, and
    // 'compute' constructs are the only construct that can do anything with
    // this yet, so skip/treat as unimplemented in this case.
    if (!isOpenACCComputeDirectiveKind(Clause.getDirectiveKind()))
      break;

    // Don't add an invalid clause to the AST.
    if (Clause.getDefaultClauseKind() == OpenACCDefaultClauseKind::Invalid)
      return nullptr;

    // OpenACC 3.3, Section 2.5.4:
    // At most one 'default' clause may appear, and it must have a value of
    // either 'none' or 'present'.
    // Second half of the sentence is diagnosed during parsing.
    if (checkAlreadyHasClauseOfKind(*this, ExistingClauses, Clause))
      return nullptr;

    return OpenACCDefaultClause::Create(
        getASTContext(), Clause.getDefaultClauseKind(), Clause.getBeginLoc(),
        Clause.getLParenLoc(), Clause.getEndLoc());
  }

  case OpenACCClauseKind::If: {
    // Restrictions only properly implemented on 'compute' constructs, and
    // 'compute' constructs are the only construct that can do anything with
    // this yet, so skip/treat as unimplemented in this case.
    if (!isOpenACCComputeDirectiveKind(Clause.getDirectiveKind()))
      break;

    // There is no prose in the standard that says duplicates aren't allowed,
    // but this diagnostic is present in other compilers, as well as makes
    // sense.
    if (checkAlreadyHasClauseOfKind(*this, ExistingClauses, Clause))
      return nullptr;

    // The parser has ensured that we have a proper condition expr, so there
    // isn't really much to do here.

    // If the 'if' clause is true, it makes the 'self' clause have no effect,
    // diagnose that here.
    // TODO OpenACC: When we add these two to other constructs, we might not
    // want to warn on this (for example, 'update').
    const auto *Itr =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCSelfClause>);
    if (Itr != ExistingClauses.end()) {
      Diag(Clause.getBeginLoc(), diag::warn_acc_if_self_conflict);
      Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
    }

    return OpenACCIfClause::Create(
        getASTContext(), Clause.getBeginLoc(), Clause.getLParenLoc(),
        Clause.getConditionExpr(), Clause.getEndLoc());
  }

  case OpenACCClauseKind::Self: {
    // Restrictions only properly implemented on 'compute' constructs, and
    // 'compute' constructs are the only construct that can do anything with
    // this yet, so skip/treat as unimplemented in this case.
    if (!isOpenACCComputeDirectiveKind(Clause.getDirectiveKind()))
      break;

    // TODO OpenACC: When we implement this for 'update', this takes a
    // 'var-list' instead of a condition expression, so semantics/handling has
    // to happen differently here.

    // There is no prose in the standard that says duplicates aren't allowed,
    // but this diagnostic is present in other compilers, as well as makes
    // sense.
    if (checkAlreadyHasClauseOfKind(*this, ExistingClauses, Clause))
      return nullptr;

    // If the 'if' clause is true, it makes the 'self' clause have no effect,
    // diagnose that here.
    // TODO OpenACC: When we add these two to other constructs, we might not
    // want to warn on this (for example, 'update').
    const auto *Itr =
        llvm::find_if(ExistingClauses, llvm::IsaPred<OpenACCIfClause>);
    if (Itr != ExistingClauses.end()) {
      Diag(Clause.getBeginLoc(), diag::warn_acc_if_self_conflict);
      Diag((*Itr)->getBeginLoc(), diag::note_acc_previous_clause_here);
    }

    return OpenACCSelfClause::Create(
        getASTContext(), Clause.getBeginLoc(), Clause.getLParenLoc(),
        Clause.getConditionExpr(), Clause.getEndLoc());
  }
  default:
    break;
  }

  Diag(Clause.getBeginLoc(), diag::warn_acc_clause_unimplemented)
      << Clause.getClauseKind();
  return nullptr;
}

void SemaOpenACC::ActOnConstruct(OpenACCDirectiveKind K,
                                 SourceLocation StartLoc) {
  switch (K) {
  case OpenACCDirectiveKind::Invalid:
    // Nothing to do here, an invalid kind has nothing we can check here.  We
    // want to continue parsing clauses as far as we can, so we will just
    // ensure that we can still work and don't check any construct-specific
    // rules anywhere.
    break;
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::Kernels:
    // Nothing to do here, there is no real legalization that needs to happen
    // here as these constructs do not take any arguments.
    break;
  default:
    Diag(StartLoc, diag::warn_acc_construct_unimplemented) << K;
    break;
  }
}

bool SemaOpenACC::ActOnStartStmtDirective(OpenACCDirectiveKind K,
                                          SourceLocation StartLoc) {
  return diagnoseConstructAppertainment(*this, K, StartLoc, /*IsStmt=*/true);
}

StmtResult SemaOpenACC::ActOnEndStmtDirective(OpenACCDirectiveKind K,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc,
                                              ArrayRef<OpenACCClause *> Clauses,
                                              StmtResult AssocStmt) {
  switch (K) {
  default:
    return StmtEmpty();
  case OpenACCDirectiveKind::Invalid:
    return StmtError();
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::Kernels:
    // TODO OpenACC: Add clauses to the construct here.
    return OpenACCComputeConstruct::Create(
        getASTContext(), K, StartLoc, EndLoc, Clauses,
        AssocStmt.isUsable() ? AssocStmt.get() : nullptr);
  }
  llvm_unreachable("Unhandled case in directive handling?");
}

StmtResult SemaOpenACC::ActOnAssociatedStmt(OpenACCDirectiveKind K,
                                            StmtResult AssocStmt) {
  switch (K) {
  default:
    llvm_unreachable("Unimplemented associated statement application");
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::Kernels:
    // There really isn't any checking here that could happen. As long as we
    // have a statement to associate, this should be fine.
    // OpenACC 3.3 Section 6:
    // Structured Block: in C or C++, an executable statement, possibly
    // compound, with a single entry at the top and a single exit at the
    // bottom.
    // FIXME: Should we reject DeclStmt's here? The standard isn't clear, and
    // an interpretation of it is to allow this and treat the initializer as
    // the 'structured block'.
    return AssocStmt;
  }
  llvm_unreachable("Invalid associated statement application");
}

bool SemaOpenACC::ActOnStartDeclDirective(OpenACCDirectiveKind K,
                                          SourceLocation StartLoc) {
  return diagnoseConstructAppertainment(*this, K, StartLoc, /*IsStmt=*/false);
}

DeclGroupRef SemaOpenACC::ActOnEndDeclDirective() { return DeclGroupRef{}; }
