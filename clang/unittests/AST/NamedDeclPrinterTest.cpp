//===- unittests/AST/NamedDeclPrinterTest.cpp --- NamedDecl printer tests -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for NamedDecl::printQualifiedName().
//
// These tests have a coding convention:
// * declaration to be printed is named 'A' unless it should have some special
// name (e.g., 'operator+');
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

class PrintMatch : public MatchFinder::MatchCallback {
  SmallString<1024> Printed;
  unsigned NumFoundDecls;
  bool SuppressUnwrittenScope;

public:
  explicit PrintMatch(bool suppressUnwrittenScope)
    : NumFoundDecls(0), SuppressUnwrittenScope(suppressUnwrittenScope) {}

  void run(const MatchFinder::MatchResult &Result) override {
    const NamedDecl *ND = Result.Nodes.getNodeAs<NamedDecl>("id");
    if (!ND)
      return;
    NumFoundDecls++;
    if (NumFoundDecls > 1)
      return;

    llvm::raw_svector_ostream Out(Printed);
    PrintingPolicy Policy = Result.Context->getPrintingPolicy();
    Policy.SuppressUnwrittenScope = SuppressUnwrittenScope;
    ND->printQualifiedName(Out, Policy);
  }

  StringRef getPrinted() const {
    return Printed;
  }

  unsigned getNumFoundDecls() const {
    return NumFoundDecls;
  }
};

::testing::AssertionResult
PrintedNamedDeclMatches(StringRef Code, const std::vector<std::string> &Args,
                        bool SuppressUnwrittenScope,
                        const DeclarationMatcher &NodeMatch,
                        StringRef ExpectedPrinted, StringRef FileName) {
  PrintMatch Printer(SuppressUnwrittenScope);
  MatchFinder Finder;
  Finder.addMatcher(NodeMatch, &Printer);
  std::unique_ptr<FrontendActionFactory> Factory =
      newFrontendActionFactory(&Finder);

  if (!runToolOnCodeWithArgs(Factory->create(), Code, Args, FileName))
    return testing::AssertionFailure()
        << "Parsing error in \"" << Code.str() << "\"";

  if (Printer.getNumFoundDecls() == 0)
    return testing::AssertionFailure()
        << "Matcher didn't find any named declarations";

  if (Printer.getNumFoundDecls() > 1)
    return testing::AssertionFailure()
        << "Matcher should match only one named declaration "
           "(found " << Printer.getNumFoundDecls() << ")";

  if (Printer.getPrinted() != ExpectedPrinted)
    return ::testing::AssertionFailure()
        << "Expected \"" << ExpectedPrinted.str() << "\", "
           "got \"" << Printer.getPrinted().str() << "\"";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult
PrintedNamedDeclCXX98Matches(StringRef Code, StringRef DeclName,
                             StringRef ExpectedPrinted) {
  std::vector<std::string> Args(1, "-std=c++98");
  return PrintedNamedDeclMatches(Code,
                                 Args,
                                 /*SuppressUnwrittenScope*/ false,
                                 namedDecl(hasName(DeclName)).bind("id"),
                                 ExpectedPrinted,
                                 "input.cc");
}

::testing::AssertionResult
PrintedWrittenNamedDeclCXX11Matches(StringRef Code, StringRef DeclName,
                                    StringRef ExpectedPrinted) {
  std::vector<std::string> Args(1, "-std=c++11");
  return PrintedNamedDeclMatches(Code,
                                 Args,
                                 /*SuppressUnwrittenScope*/ true,
                                 namedDecl(hasName(DeclName)).bind("id"),
                                 ExpectedPrinted,
                                 "input.cc");
}

} // unnamed namespace

TEST(NamedDeclPrinter, TestNamespace1) {
  ASSERT_TRUE(PrintedNamedDeclCXX98Matches(
    "namespace { int A; }",
    "A",
    "(anonymous namespace)::A"));
}

TEST(NamedDeclPrinter, TestNamespace2) {
  ASSERT_TRUE(PrintedWrittenNamedDeclCXX11Matches(
    "inline namespace Z { namespace { int A; } }",
    "A",
    "A"));
}
