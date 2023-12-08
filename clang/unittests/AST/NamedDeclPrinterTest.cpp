//===- unittests/AST/NamedDeclPrinterTest.cpp --- NamedDecl printer tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "ASTPrint.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ast_matchers;
using namespace tooling;

namespace {

class PrintMatch : public MatchFinder::MatchCallback {
  SmallString<1024> Printed;
  unsigned NumFoundDecls;
  std::function<void(llvm::raw_ostream &OS, const NamedDecl *)> Printer;

public:
  explicit PrintMatch(
      std::function<void(llvm::raw_ostream &OS, const NamedDecl *)> Printer)
      : NumFoundDecls(0), Printer(std::move(Printer)) {}

  void run(const MatchFinder::MatchResult &Result) override {
    const NamedDecl *ND = Result.Nodes.getNodeAs<NamedDecl>("id");
    if (!ND)
      return;
    NumFoundDecls++;
    if (NumFoundDecls > 1)
      return;

    llvm::raw_svector_ostream Out(Printed);
    Printer(Out, ND);
  }

  StringRef getPrinted() const {
    return Printed;
  }

  unsigned getNumFoundDecls() const {
    return NumFoundDecls;
  }
};

::testing::AssertionResult PrintedDeclMatches(
    StringRef Code, const std::vector<std::string> &Args,
    const DeclarationMatcher &NodeMatch, StringRef ExpectedPrinted,
    StringRef FileName,
    std::function<void(llvm::raw_ostream &, const NamedDecl *)> Print) {
  return PrintedNodeMatches<NamedDecl>(
      Code, Args, NodeMatch, ExpectedPrinted, FileName,
      [Print](llvm::raw_ostream &Out, const ASTContext *Context,
              const NamedDecl *ND,
              PrintingPolicyAdjuster PolicyAdjuster) { Print(Out, ND); });
}

::testing::AssertionResult
PrintedNamedDeclMatches(StringRef Code, const std::vector<std::string> &Args,
                        bool SuppressUnwrittenScope,
                        const DeclarationMatcher &NodeMatch,
                        StringRef ExpectedPrinted, StringRef FileName) {
  return PrintedDeclMatches(Code, Args, NodeMatch, ExpectedPrinted, FileName,
                            [=](llvm::raw_ostream &Out, const NamedDecl *ND) {
                              auto Policy =
                                  ND->getASTContext().getPrintingPolicy();
                              Policy.SuppressUnwrittenScope =
                                  SuppressUnwrittenScope;
                              ND->printQualifiedName(Out, Policy);
                            });
}

::testing::AssertionResult
PrintedNamedDeclCXX98Matches(StringRef Code, StringRef DeclName,
                             StringRef ExpectedPrinted) {
  std::vector<std::string> Args(1, "-std=c++98");
  return PrintedNamedDeclMatches(Code, Args,
                                 /*SuppressUnwrittenScope*/ false,
                                 namedDecl(hasName(DeclName)).bind("id"),
                                 ExpectedPrinted, "input.cc");
}

::testing::AssertionResult
PrintedWrittenNamedDeclCXX11Matches(StringRef Code, StringRef DeclName,
                                    StringRef ExpectedPrinted) {
  std::vector<std::string> Args(1, "-std=c++11");
  return PrintedNamedDeclMatches(Code, Args,
                                 /*SuppressUnwrittenScope*/ true,
                                 namedDecl(hasName(DeclName)).bind("id"),
                                 ExpectedPrinted, "input.cc");
}

::testing::AssertionResult
PrintedWrittenPropertyDeclObjCMatches(StringRef Code, StringRef DeclName,
                                   StringRef ExpectedPrinted) {
  std::vector<std::string> Args{"-std=c++11", "-xobjective-c++"};
  return PrintedNamedDeclMatches(Code, Args,
                                 /*SuppressUnwrittenScope*/ true,
                                 objcPropertyDecl(hasName(DeclName)).bind("id"),
                                 ExpectedPrinted, "input.m");
}

::testing::AssertionResult
PrintedNestedNameSpecifierMatches(StringRef Code, StringRef DeclName,
                                  StringRef ExpectedPrinted) {
  std::vector<std::string> Args{"-std=c++11"};
  return PrintedDeclMatches(Code, Args, namedDecl(hasName(DeclName)).bind("id"),
                            ExpectedPrinted, "input.cc",
                            [](llvm::raw_ostream &Out, const NamedDecl *D) {
                              D->printNestedNameSpecifier(Out);
                            });
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

TEST(NamedDeclPrinter, TestUnscopedUnnamedEnum) {
  ASSERT_TRUE(PrintedWrittenNamedDeclCXX11Matches(
    "enum { A };",
    "A",
    "A"));
}

TEST(NamedDeclPrinter, TestNamedEnum) {
  ASSERT_TRUE(PrintedWrittenNamedDeclCXX11Matches(
    "enum X { A };",
    "A",
    "A"));
}

TEST(NamedDeclPrinter, TestScopedNamedEnum) {
  ASSERT_TRUE(PrintedWrittenNamedDeclCXX11Matches(
    "enum class X { A };",
    "A",
    "X::A"));
}

TEST(NamedDeclPrinter, TestClassWithUnscopedUnnamedEnum) {
  ASSERT_TRUE(PrintedWrittenNamedDeclCXX11Matches(
    "class X { enum { A }; };",
    "A",
    "X::A"));
}

TEST(NamedDeclPrinter, TestClassWithUnscopedNamedEnum) {
  ASSERT_TRUE(PrintedWrittenNamedDeclCXX11Matches(
    "class X { enum Y { A }; };",
    "A",
    "X::A"));
}

TEST(NamedDeclPrinter, TestClassWithScopedNamedEnum) {
  ASSERT_TRUE(PrintedWrittenNamedDeclCXX11Matches(
    "class X { enum class Y { A }; };",
    "A",
    "X::Y::A"));
}

TEST(NamedDeclPrinter, TestLinkageInNamespace) {
  ASSERT_TRUE(PrintedWrittenNamedDeclCXX11Matches(
    "namespace X { extern \"C\" { int A; } }",
    "A",
    "X::A"));
}

TEST(NamedDeclPrinter, TestObjCClassExtension) {
  const char *Code =
R"(
  @interface Obj
  @end

  @interface Obj ()
  @property(nonatomic) int property;
  @end
)";
  ASSERT_TRUE(PrintedWrittenPropertyDeclObjCMatches(
    Code,
    "property",
    "Obj::property"));
}

TEST(NamedDeclPrinter, TestInstanceObjCClassExtension) {
  const char *Code =
R"(
@interface ObjC
@end
@interface ObjC () {
  char data; // legal with non-fragile ABI.
}
@end
)";

  std::vector<std::string> Args{
      "-std=c++11", "-xobjective-c++",
      "-fobjc-runtime=macosx" /*force to use non-fragile ABI*/};
  ASSERT_TRUE(PrintedNamedDeclMatches(Code, Args,
                                      /*SuppressUnwrittenScope*/ true,
                                      namedDecl(hasName("data")).bind("id"),
                                      // not "::data"
                                      "ObjC::data", "input.mm"));
}

TEST(NamedDeclPrinter, TestObjCClassExtensionWithGetter) {
  const char *Code =
R"(
  @interface Obj
  @end

  @interface Obj ()
  @property(nonatomic, getter=myPropertyGetter) int property;
  @end
)";
  ASSERT_TRUE(PrintedWrittenPropertyDeclObjCMatches(
    Code,
    "property",
    "Obj::property"));
}

TEST(NamedDeclPrinter, NestedNameSpecifierSimple) {
  const char *Code =
      R"(
  namespace foo { namespace bar { void func(); }  }
)";
  ASSERT_TRUE(PrintedNestedNameSpecifierMatches(Code, "func", "foo::bar::"));
}

TEST(NamedDeclPrinter, NestedNameSpecifierTemplateArgs) {
  const char *Code =
      R"(
        template <class T> struct vector;
        template <> struct vector<int> { int method(); };
)";
  ASSERT_TRUE(
      PrintedNestedNameSpecifierMatches(Code, "method", "vector<int>::"));
}
