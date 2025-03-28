//===- LValueElementTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <fstream>

using namespace clang;
using namespace ento;

namespace {

class LValueElementChecker
    : public Checker<check::PreStmt<ArraySubscriptExpr>> {
public:
  void checkPreStmt(const ArraySubscriptExpr *ASE, CheckerContext &CC) const {
    const Expr *BaseEx = ASE->getBase()->IgnoreParens();
    const Expr *IdxEx = ASE->getIdx()->IgnoreParens();

    SVal BaseVal = CC.getSVal(BaseEx);
    SVal IdxVal = CC.getSVal(IdxEx);

    auto IdxNonLoc = IdxVal.getAs<NonLoc>();
    ASSERT_TRUE(IdxNonLoc) << "Expect NonLoc as index SVal\n";

    QualType ArrayT = ASE->getType();
    SVal LValue =
        CC.getStoreManager().getLValueElement(ArrayT, *IdxNonLoc, BaseVal);

    if (ExplodedNode *Node = CC.generateNonFatalErrorNode(CC.getState())) {
      std::string TmpStr;
      llvm::raw_string_ostream TmpStream{TmpStr};
      LValue.dumpToStream(TmpStream);
      auto Report = std::make_unique<PathSensitiveBugReport>(Bug, TmpStr, Node);
      CC.emitReport(std::move(Report));
    }
  }

private:
  const BugType Bug{this, "LValueElementBug"};
};

void addLValueElementChecker(AnalysisASTConsumer &AnalysisConsumer,
                             AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"LValueElementChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<LValueElementChecker>("LValueElementChecker", "Desc",
                                              "DocsURI");
  });
}

bool runLValueElementChecker(StringRef Code, std::string &Output) {
  return runCheckerOnCode<addLValueElementChecker>(Code.str(), Output,
                                                   /*OnlyEmitWarnings=*/true);
}

TEST(LValueElementTest, IdxConInt) {
  StringRef Code = R"cpp(
const int index = 1;
extern int array[3];

void top() {
  array[index];
})cpp";

  std::string Output;
  ASSERT_TRUE(runLValueElementChecker(Code, Output));
  EXPECT_EQ(Output, "LValueElementChecker: &Element{array,1 S64b,int}\n");
}

TEST(LValueElementTest, IdxSymVal) {
  StringRef Code = R"cpp(
extern int un_index;
extern int array[3];

void top() {
  array[un_index];
})cpp";

  std::string Output;
  ASSERT_TRUE(runLValueElementChecker(Code, Output));
  EXPECT_EQ(Output,
            "LValueElementChecker: &Element{array,reg_$0<int un_index>,int}\n");
}

TEST(LValueElementTest, IdxConIntSymVal) {
  StringRef Code = R"cpp(
extern int un_index;
extern int matrix[3][3];

void top() {
  matrix[1][un_index];
})cpp";

  std::string Output;
  ASSERT_TRUE(runLValueElementChecker(Code, Output));
  EXPECT_EQ(Output, "LValueElementChecker: &Element{Element{matrix,1 "
                    "S64b,int[3]},reg_$0<int un_index>,int}\n"
                    "LValueElementChecker: &Element{matrix,1 S64b,int[3]}\n");
}

TEST(LValueElementTest, IdxSymValConInt) {
  StringRef Code = R"cpp(
extern int un_index;
extern int matrix[3][3];

void top() {
  matrix[un_index][1];
})cpp";

  std::string Output;
  ASSERT_TRUE(runLValueElementChecker(Code, Output));
  EXPECT_EQ(
      Output,
      "LValueElementChecker: &Element{Element{matrix,reg_$0<int "
      "un_index>,int[3]},1 S64b,int}\n"
      "LValueElementChecker: &Element{matrix,reg_$0<int un_index>,int[3]}\n");
}

TEST(LValueElementTest, IdxSymValSymVal) {
  StringRef Code = R"cpp(
extern int un_index;
extern int matrix[3][3];

void top() {
  matrix[un_index][un_index];
})cpp";

  std::string Output;
  ASSERT_TRUE(runLValueElementChecker(Code, Output));
  EXPECT_EQ(
      Output,
      "LValueElementChecker: &Element{Element{matrix,reg_$0<int "
      "un_index>,int[3]},reg_$0<int un_index>,int}\n"
      "LValueElementChecker: &Element{matrix,reg_$0<int un_index>,int[3]}\n");
}

} // namespace
