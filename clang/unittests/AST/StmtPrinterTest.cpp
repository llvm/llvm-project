//===- unittests/AST/StmtPrinterTest.cpp --- Statement printer tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for Stmt::printPretty() and related methods.
//
// Search this file for WRONG to see test cases that are producing something
// completely wrong, invalid C++ or just misleading.
//
// These tests have a coding convention:
// * statements to be printed should be contained within a function named 'A'
//   unless it should have some special name (e.g., 'operator+');
// * additional helper declarations are 'Z', 'Y', 'X' and so on.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ast_matchers;
using namespace tooling;

namespace {

void PrintStmt(raw_ostream &Out, const ASTContext *Context, const Stmt *S) {
  PrintingPolicy Policy = Context->getPrintingPolicy();
  S->printPretty(Out, /*Helper*/ 0, Policy);
}

class PrintMatch : public MatchFinder::MatchCallback {
  SmallString<1024> Printed;
  unsigned NumFoundStmts;

public:
  PrintMatch() : NumFoundStmts(0) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    const Stmt *S = Result.Nodes.getStmtAs<Stmt>("id");
    if (!S)
      return;
    NumFoundStmts++;
    if (NumFoundStmts > 1)
      return;

    llvm::raw_svector_ostream Out(Printed);
    PrintStmt(Out, Result.Context, S);
  }

  StringRef getPrinted() const {
    return Printed;
  }

  unsigned getNumFoundStmts() const {
    return NumFoundStmts;
  }
};

::testing::AssertionResult PrintedStmtMatches(
                                        StringRef Code,
                                        const std::vector<std::string> &Args,
                                        const DeclarationMatcher &NodeMatch,
                                        StringRef ExpectedPrinted) {

  PrintMatch Printer;
  MatchFinder Finder;
  Finder.addMatcher(NodeMatch, &Printer);
  OwningPtr<FrontendActionFactory> Factory(newFrontendActionFactory(&Finder));

  if (!runToolOnCodeWithArgs(Factory->create(), Code, Args))
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";

  if (Printer.getNumFoundStmts() == 0)
    return testing::AssertionFailure()
        << "Matcher didn't find any statements";

  if (Printer.getNumFoundStmts() > 1)
    return testing::AssertionFailure()
        << "Matcher should match only one statement "
           "(found " << Printer.getNumFoundStmts() << ")";

  if (Printer.getPrinted() != ExpectedPrinted)
    return ::testing::AssertionFailure()
      << "Expected \"" << ExpectedPrinted << "\", "
         "got \"" << Printer.getPrinted() << "\"";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult PrintedStmtCXX98Matches(
                                              StringRef Code,
                                              StringRef ContainingFunction,
                                              StringRef ExpectedPrinted) {
  std::vector<std::string> Args;
  Args.push_back("-std=c++98");
  Args.push_back("-Wno-unused-value");
  return PrintedStmtMatches(Code,
                            Args,
                            functionDecl(hasName(ContainingFunction),
                                         has(compoundStmt(has(stmt().bind("id"))))),
                            ExpectedPrinted);
}

::testing::AssertionResult PrintedStmtMSMatches(
                                              StringRef Code,
                                              StringRef ContainingFunction,
                                              StringRef ExpectedPrinted) {
  std::vector<std::string> Args;
  Args.push_back("-std=c++98");
  Args.push_back("-fms-extensions");
  Args.push_back("-Wno-unused-value");
  return PrintedStmtMatches(Code,
                            Args,
                            functionDecl(hasName(ContainingFunction),
                                         has(compoundStmt(has(stmt().bind("id"))))),
                            ExpectedPrinted);
}

} // unnamed namespace

TEST(StmtPrinter, TestIntegerLiteral) {
  ASSERT_TRUE(PrintedStmtCXX98Matches(
    "void A() {"
    "  1, -1, 1U, 1u,"
    "  1L, 1l, -1L, 1UL, 1ul,"
    "  1LL, -1LL, 1ULL;"
    "}",
    "A",
    "1 , -1 , 1U , 1U , "
    "1L , 1L , -1L , 1UL , 1UL , "
    "1LL , -1LL , 1ULL"));
    // Should be: with semicolon
}

TEST(StmtPrinter, TestMSIntegerLiteral) {
  ASSERT_TRUE(PrintedStmtMSMatches(
    "void A() {"
    "  1i8, -1i8, 1ui8, "
    "  1i16, -1i16, 1ui16, "
    "  1i32, -1i32, 1ui32, "
    "  1i64, -1i64, 1ui64, "
    "  1i128, -1i128, 1ui128, 1Ui128,"
    "  0x10000000000000000i128;"
    "}",
    "A",
    "1 , -1 , 1U , "
    "1 , -1 , 1U , "
    "1L , -1L , 1UL , "
    "1LL , -1LL , 1ULL , "
    "1 , -1 , 1U , 1U , "
    "18446744073709551616i128"));
    // Should be: with semicolon
    // WRONG; all 128-bit literals should be printed as 128-bit.
    // (This is because currently we do semantic analysis incorrectly.)
}

TEST(StmtPrinter, TestFloatingPointLiteral) {
  ASSERT_TRUE(PrintedStmtCXX98Matches(
    "void A() { 1.0f, -1.0f, 1.0, -1.0, 1.0l, -1.0l; }",
    "A",
    "1.F , -1.F , 1. , -1. , 1.L , -1.L"));
    // Should be: with semicolon
}

