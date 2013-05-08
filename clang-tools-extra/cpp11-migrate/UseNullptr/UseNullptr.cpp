//===-- LoopConvert/LoopConvert.cpp - C++11 for-loop migration --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation of the UseNullptrTransform
/// class.
///
//===----------------------------------------------------------------------===//

#include "UseNullptr.h"
#include "NullptrActions.h"
#include "NullptrMatchers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

using clang::ast_matchers::MatchFinder;
using namespace clang::tooling;
using namespace clang;

int UseNullptrTransform::apply(const FileContentsByPath &InputStates,
                               RiskLevel MaxRisk,
                               const CompilationDatabase &Database,
                               const std::vector<std::string> &SourcePaths,
                               FileContentsByPath &ResultStates) {
  RefactoringTool UseNullptrTool(Database, SourcePaths);

  for (FileContentsByPath::const_iterator I = InputStates.begin(),
       E = InputStates.end();
       I != E; ++I) {
    UseNullptrTool.mapVirtualFile(I->first, I->second);
  }

  unsigned AcceptedChanges = 0;

  MatchFinder Finder;
  NullptrFixer Fixer(UseNullptrTool.getReplacements(),
                     AcceptedChanges,
                     MaxRisk);

  Finder.addMatcher(makeCastSequenceMatcher(), &Fixer);

  if (int result = UseNullptrTool.run(newFrontendActionFactory(&Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return result;
  }

  RewriterContainer Rewrite(UseNullptrTool.getFiles(), InputStates);

  // FIXME: Do something if some replacements didn't get applied?
  UseNullptrTool.applyAllReplacements(Rewrite.getRewriter());

  collectResults(Rewrite.getRewriter(), InputStates, ResultStates);

  setAcceptedChanges(AcceptedChanges);

  return 0;
}
