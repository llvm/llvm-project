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

  /// \brief Reports all clones to the user.
  void reportClones(SourceManager &SM, AnalysisManager &Mgr,
                    int MinComplexity) const;

  /// \brief Reports only suspicious clones to the user along with informaton
  ///        that explain why they are suspicious.
  void reportSuspiciousClones(SourceManager &SM, AnalysisManager &Mgr,
                              int MinComplexity) const;
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

  bool ReportSuspiciousClones = Mgr.getAnalyzerOptions().getBooleanOption(
      "ReportSuspiciousClones", true, this);

  bool ReportNormalClones = Mgr.getAnalyzerOptions().getBooleanOption(
      "ReportNormalClones", true, this);

  if (ReportSuspiciousClones)
    reportSuspiciousClones(BR.getSourceManager(), Mgr, MinComplexity);

  if (ReportNormalClones)
    reportClones(BR.getSourceManager(), Mgr, MinComplexity);
}

void CloneChecker::reportClones(SourceManager &SM, AnalysisManager &Mgr,
                                int MinComplexity) const {

  std::vector<CloneDetector::CloneGroup> CloneGroups;
  Detector.findClones(CloneGroups, MinComplexity);

  DiagnosticsEngine &DiagEngine = Mgr.getDiagnostic();

  unsigned WarnID = DiagEngine.getCustomDiagID(DiagnosticsEngine::Warning,
                                               "Detected code clone.");

  unsigned NoteID = DiagEngine.getCustomDiagID(DiagnosticsEngine::Note,
                                               "Related code clone is here.");

  for (CloneDetector::CloneGroup &Group : CloneGroups) {
    // We group the clones by printing the first as a warning and all others
    // as a note.
    DiagEngine.Report(Group.Sequences.front().getStartLoc(), WarnID);
    for (unsigned i = 1; i < Group.Sequences.size(); ++i) {
      DiagEngine.Report(Group.Sequences[i].getStartLoc(), NoteID);
    }
  }
}

void CloneChecker::reportSuspiciousClones(SourceManager &SM,
                                          AnalysisManager &Mgr,
                                          int MinComplexity) const {

  std::vector<CloneDetector::SuspiciousClonePair> Clones;
  Detector.findSuspiciousClones(Clones, MinComplexity);

  DiagnosticsEngine &DiagEngine = Mgr.getDiagnostic();

  auto SuspiciousCloneWarning = DiagEngine.getCustomDiagID(
      DiagnosticsEngine::Warning, "suspicious code clone detected; did you "
                                  "mean to use %0?");

  auto RelatedCloneNote = DiagEngine.getCustomDiagID(
      DiagnosticsEngine::Note, "suggestion is based on the usage of this "
                               "variable in a similar piece of code");

  auto RelatedSuspiciousCloneNote = DiagEngine.getCustomDiagID(
      DiagnosticsEngine::Note, "suggestion is based on the usage of this "
                               "variable in a similar piece of code; did you "
                               "mean to use %0?");

  for (CloneDetector::SuspiciousClonePair &Pair : Clones) {
    // The first clone always has a suggestion and we report it to the user
    // along with the place where the suggestion should be used.
    DiagEngine.Report(Pair.FirstCloneInfo.VarRange.getBegin(),
                      SuspiciousCloneWarning)
        << Pair.FirstCloneInfo.VarRange << Pair.FirstCloneInfo.Suggestion;

    // The second clone can have a suggestion and if there is one, we report
    // that suggestion to the user.
    if (Pair.SecondCloneInfo.Suggestion) {
      DiagEngine.Report(Pair.SecondCloneInfo.VarRange.getBegin(),
                        RelatedSuspiciousCloneNote)
          << Pair.SecondCloneInfo.VarRange << Pair.SecondCloneInfo.Suggestion;
      continue;
    }

    // If there isn't a suggestion in the second clone, we only inform the
    // user where we got the idea that his code could contain an error.
    DiagEngine.Report(Pair.SecondCloneInfo.VarRange.getBegin(),
                      RelatedCloneNote)
        << Pair.SecondCloneInfo.VarRange;
  }
}

//===----------------------------------------------------------------------===//
// Register CloneChecker
//===----------------------------------------------------------------------===//

void ento::registerCloneChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<CloneChecker>();
}
