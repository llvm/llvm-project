//===-- AnalysisManager.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"

using namespace clang;
using namespace ento;

void AnalysisManager::anchor() { }

AnalysisManager::AnalysisManager(ASTContext &ASTCtx, Preprocessor &PP,
                                 PathDiagnosticConsumers PDC,
                                 StoreManagerCreator storemgr,
                                 ConstraintManagerCreator constraintmgr,
                                 CheckerManager *checkerMgr,
                                 AnalyzerOptions &Options,
                                 std::unique_ptr<CodeInjector> injector)
    : AnaCtxMgr(
          ASTCtx, Options.UnoptimizedCFG,
          Options.ShouldIncludeImplicitDtorsInCFG,
          /*addInitializers=*/true, Options.ShouldIncludeTemporaryDtorsInCFG,
          Options.ShouldIncludeLifetimeInCFG,
          // Adding LoopExit elements to the CFG is a requirement for loop
          // unrolling.
          Options.ShouldIncludeLoopExitInCFG || Options.ShouldUnrollLoops,
          Options.ShouldIncludeScopesInCFG, Options.ShouldSynthesizeBodies,
          Options.ShouldConditionalizeStaticInitializers,
          /*addCXXNewAllocator=*/true,
          Options.ShouldIncludeRichConstructorsInCFG,
          Options.ShouldElideConstructors,
          /*addVirtualBaseBranches=*/true, std::move(injector)),
      Ctx(ASTCtx), PP(PP), LangOpts(ASTCtx.getLangOpts()),
      PathConsumers(std::move(PDC)), CreateStoreMgr(storemgr),
      CreateConstraintMgr(constraintmgr), CheckerMgr(checkerMgr),
      options(Options) {
  AnaCtxMgr.getCFGBuildOptions().setAllAlwaysAdd();
  AnaCtxMgr.getCFGBuildOptions().OmitImplicitValueInitializers = true;
  AnaCtxMgr.getCFGBuildOptions().AddCXXDefaultInitExprInAggregates =
      Options.ShouldIncludeDefaultInitForAggregates;
}

AnalysisManager::~AnalysisManager() {
  FlushDiagnostics();
}

void AnalysisManager::FlushDiagnostics() {
  PathDiagnosticConsumer::FilesMade filesMade;
  for (const auto &Consumer : PathConsumers) {
    Consumer->FlushDiagnostics(&filesMade);
  }
}
