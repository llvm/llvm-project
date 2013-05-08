//===-- AddOverride/AddOverride.cpp - add C++11 override -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation of the AddOverrideTransform
/// class.
///
//===----------------------------------------------------------------------===//

#include "AddOverride.h"
#include "AddOverrideActions.h"
#include "AddOverrideMatchers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

using clang::ast_matchers::MatchFinder;
using namespace clang::tooling;
using namespace clang;

int AddOverrideTransform::apply(const FileContentsByPath &InputStates,
                                RiskLevel MaxRisk,
                                const CompilationDatabase &Database,
                                const std::vector<std::string> &SourcePaths,
                                FileContentsByPath &ResultStates) {
  RefactoringTool AddOverrideTool(Database, SourcePaths);

  for (FileContentsByPath::const_iterator I = InputStates.begin(),
       E = InputStates.end();
       I != E; ++I) {
    AddOverrideTool.mapVirtualFile(I->first, I->second);
  }

  unsigned AcceptedChanges = 0;

  MatchFinder Finder;
  AddOverrideFixer Fixer(AddOverrideTool.getReplacements(), AcceptedChanges);

  Finder.addMatcher(makeCandidateForOverrideAttrMatcher(), &Fixer);

  if (int result = AddOverrideTool.run(newFrontendActionFactory(&Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return result;
  }

  RewriterContainer Rewrite(AddOverrideTool.getFiles(), InputStates);

  // FIXME: Do something if some replacements didn't get applied?
  AddOverrideTool.applyAllReplacements(Rewrite.getRewriter());

  collectResults(Rewrite.getRewriter(), InputStates, ResultStates);

  setAcceptedChanges(AcceptedChanges);

  return 0;
}
