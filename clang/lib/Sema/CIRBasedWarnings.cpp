//=- CIRBasedWarnings.cpp - Sema warnings based on libAnalysis -*- C++ ----*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines analysis_warnings::[Policy,Executor].
// Together they are used by Sema to issue warnings based on inexpensive
// static analysis algorithms using ClangIR.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/CIRBasedWarnings.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "clang/CIR/CIRGenerator.h"

#include <algorithm>
#include <deque>
#include <iterator>

using namespace clang;

///
/// CIRBasedWarnings
///
static unsigned isEnabled(DiagnosticsEngine &D, unsigned diag) {
  return (unsigned)!D.isIgnored(diag, SourceLocation());
}

sema::CIRBasedWarnings::CIRBasedWarnings(Sema &s) : S(s) {

  using namespace diag;
  DiagnosticsEngine &D = S.getDiagnostics();

  DefaultPolicy.enableCheckUnreachable =
      isEnabled(D, warn_unreachable) || isEnabled(D, warn_unreachable_break) ||
      isEnabled(D, warn_unreachable_return) ||
      isEnabled(D, warn_unreachable_loop_increment);

  DefaultPolicy.enableThreadSafetyAnalysis = isEnabled(D, warn_double_lock);

  DefaultPolicy.enableConsumedAnalysis =
      isEnabled(D, warn_use_in_invalid_state);

  // TODO: figure out a way to get this properly. This isn't actually reasonable
  // to ask for prior to codegen, so we're just subbing in a blank one.
  CIRGen = std::make_unique<cir::CIRGenerator>(CodeGenOptions());
  CIRGen->Initialize(S.getASTContext());
}

// We need this here for unique_ptr with forward declared class.
sema::CIRBasedWarnings::~CIRBasedWarnings() = default;

static void flushDiagnostics(Sema &S, const sema::FunctionScopeInfo *fscope) {
  for (const auto &D : fscope->PossiblyUnreachableDiags)
    S.Diag(D.Loc, D.PD);
}

void clang::sema::CIRBasedWarnings::IssueWarnings(
    sema::AnalysisBasedWarnings::Policy P, sema::FunctionScopeInfo *fscope,
    const Decl *D, QualType BlockType) {
  // We avoid doing analysis-based warnings when there are errors for
  // two reasons:
  // (1) The CFGs often can't be constructed (if the body is invalid), so
  //     don't bother trying.
  // (2) The code already has problems; running the analysis just takes more
  //     time.
  DiagnosticsEngine &Diags = S.getDiagnostics();

  // Do not do any analysis if we are going to just ignore them.
  if (Diags.getIgnoreAllWarnings() ||
      (Diags.getSuppressSystemWarnings() &&
       S.SourceMgr.isInSystemHeader(D->getLocation())))
    return;

  // For code in dependent contexts, we'll do this at instantiation time.
  if (cast<DeclContext>(D)->isDependentContext())
    return;

  if (S.hasUncompilableErrorOccurred()) {
    // Flush out any possibly unreachable diagnostics.
    flushDiagnostics(S, fscope);
    return;
  }

  const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  assert(FD && "Only know how to handle functions");

  // TODO: up to this point this behaves the same as
  // AnalysisBasedWarnings::IssueWarnings

  // Unlike Clang CFG, we share CIR state between each analyzed function,
  // retrieve or create a new context.
  CIRGen->EmitFunction(FD);
}

void clang::sema::CIRBasedWarnings::PrintStats() const {
  llvm::errs() << "\n*** CIR Based Warnings Stats:\n";
}
