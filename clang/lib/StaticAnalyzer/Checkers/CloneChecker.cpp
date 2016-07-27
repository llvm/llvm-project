//===--- CloneChecker.cpp - Clone detection checker -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CloneChecker is a checker that reports clones in the current translation
/// unit.
///
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/Analysis/CloneDetection.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class CloneChecker
    : public Checker<check::ASTCodeBody, check::EndOfTranslationUnit> {
  mutable CloneDetector Detector;

public:
  void checkASTCodeBody(const Decl *D, AnalysisManager &Mgr,
                        BugReporter &BR) const;

  void checkEndOfTranslationUnit(const TranslationUnitDecl *TU,
                                 AnalysisManager &Mgr, BugReporter &BR) const;
};
} // end anonymous namespace

void CloneChecker::checkASTCodeBody(const Decl *D, AnalysisManager &Mgr,
                                    BugReporter &BR) const {
  // Every statement that should be included in the search for clones needs to
  // be passed to the CloneDetector.
  Detector.analyzeCodeBody(D);
}

void CloneChecker::checkEndOfTranslationUnit(const TranslationUnitDecl *TU,
                                             AnalysisManager &Mgr,
                                             BugReporter &BR) const {
  // At this point, every statement in the translation unit has been analyzed by
  // the CloneDetector. The only thing left to do is to report the found clones.

  int MinComplexity = Mgr.getAnalyzerOptions().getOptionAsInteger(
      "MinimumCloneComplexity", 10, this);

  assert(MinComplexity >= 0);

  SourceManager &SM = BR.getSourceManager();

  std::vector<CloneDetector::CloneGroup> CloneGroups;
  Detector.findClones(CloneGroups, MinComplexity);

  DiagnosticsEngine &DiagEngine = Mgr.getDiagnostic();

  unsigned WarnID = DiagEngine.getCustomDiagID(DiagnosticsEngine::Warning,
                                               "Detected code clone.");

  unsigned NoteID = DiagEngine.getCustomDiagID(DiagnosticsEngine::Note,
                                               "Related code clone is here.");

  for (CloneDetector::CloneGroup &Group : CloneGroups) {
    // For readability reasons we sort the clones by line numbers.
    std::sort(Group.Sequences.begin(), Group.Sequences.end(),
              [&SM](const StmtSequence &LHS, const StmtSequence &RHS) {
                return SM.isBeforeInTranslationUnit(LHS.getStartLoc(),
                                                    RHS.getStartLoc()) &&
                       SM.isBeforeInTranslationUnit(LHS.getEndLoc(),
                                                    RHS.getEndLoc());
              });

    // We group the clones by printing the first as a warning and all others
    // as a note.
    DiagEngine.Report(Group.Sequences.front().getStartLoc(), WarnID);
    for (unsigned i = 1; i < Group.Sequences.size(); ++i) {
      DiagEngine.Report(Group.Sequences[i].getStartLoc(), NoteID);
    }
  }
}

//===----------------------------------------------------------------------===//
// Register CloneChecker
//===----------------------------------------------------------------------===//

void ento::registerCloneChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<CloneChecker>();
}
