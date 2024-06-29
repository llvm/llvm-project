//===- unittests/Analysis/CloneDetectionTest.cpp - Clone detection tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CloneDetection.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace analysis {
namespace {

class CloneDetectionVisitor
    : public RecursiveASTVisitor<CloneDetectionVisitor> {

  CloneDetector &Detector;

public:
  explicit CloneDetectionVisitor(CloneDetector &D) : Detector(D) {}

  bool VisitFunctionDecl(FunctionDecl *D) {
    Detector.analyzeCodeBody(D);
    return true;
  }
};

/// Example constraint for testing purposes.
/// Filters out all statements that are in a function which name starts with
/// "bar".
class NoBarFunctionConstraint {
public:
  void constrain(std::vector<CloneDetector::CloneGroup> &CloneGroups) {
    CloneConstraint::splitCloneGroups(
        CloneGroups, [](const StmtSequence &A, const StmtSequence &B) {
          // Check if one of the sequences is in a function which name starts
          // with "bar".
          for (const StmtSequence &Arg : {A, B}) {
            if (const auto *D =
                    dyn_cast<const FunctionDecl>(Arg.getContainingDecl())) {
              if (D->getName().starts_with("bar"))
                return false;
            }
          }
          return true;
        });
  }
};

TEST(CloneDetector, FilterFunctionsByName) {
  auto ASTUnit =
      clang::tooling::buildASTFromCode("void foo1(int &a1) { a1++; }\n"
                                       "void foo2(int &a2) { a2++; }\n"
                                       "void bar1(int &a3) { a3++; }\n"
                                       "void bar2(int &a4) { a4++; }\n");
  auto TU = ASTUnit->getASTContext().getTranslationUnitDecl();

  CloneDetector Detector;
  // Push all the function bodies into the detector.
  CloneDetectionVisitor Visitor(Detector);
  Visitor.TraverseTranslationUnitDecl(TU);

  // Find clones with the usual settings, but we want to filter out
  // all statements from functions which names start with "bar".
  std::vector<CloneDetector::CloneGroup> CloneGroups;
  Detector.findClones(CloneGroups, NoBarFunctionConstraint(),
                      RecursiveCloneTypeIIHashConstraint(),
                      MinComplexityConstraint(2), MinGroupSizeConstraint(2),
                      RecursiveCloneTypeIIVerifyConstraint(),
                      OnlyLargestCloneConstraint());

  ASSERT_EQ(CloneGroups.size(), 1u);
  ASSERT_EQ(CloneGroups.front().size(), 2u);

  for (auto &Clone : CloneGroups.front()) {
    const auto ND = dyn_cast<const FunctionDecl>(Clone.getContainingDecl());
    ASSERT_TRUE(ND != nullptr);
    // Check that no function name starting with "bar" is in the results...
    ASSERT_TRUE(ND->getNameAsString().find("bar") != 0);
  }

  // Retry above's example without the filter...
  CloneGroups.clear();

  Detector.findClones(CloneGroups, RecursiveCloneTypeIIHashConstraint(),
                      MinComplexityConstraint(2), MinGroupSizeConstraint(2),
                      RecursiveCloneTypeIIVerifyConstraint(),
                      OnlyLargestCloneConstraint());
  ASSERT_EQ(CloneGroups.size(), 1u);
  ASSERT_EQ(CloneGroups.front().size(), 4u);

  // Count how many functions with the bar prefix we have in the results.
  int FoundFunctionsWithBarPrefix = 0;
  for (auto &Clone : CloneGroups.front()) {
    const auto ND = dyn_cast<const FunctionDecl>(Clone.getContainingDecl());
    ASSERT_TRUE(ND != nullptr);
    // This time check that we picked up the bar functions from above
    if (ND->getNameAsString().find("bar") == 0) {
      FoundFunctionsWithBarPrefix++;
    }
  }
  // We should have found the two functions bar1 and bar2.
  ASSERT_EQ(FoundFunctionsWithBarPrefix, 2);
}
} // namespace
} // namespace analysis
} // namespace clang
