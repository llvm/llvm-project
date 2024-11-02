//===-- ChangeNamespaceTests.cpp - Change namespace unit tests ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ChangeNamespace.h"
#include "unittests/Tooling/RewriterTestContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace change_namespace {
namespace {

class ChangeNamespaceTest : public ::testing::Test {
public:
  std::string runChangeNamespaceOnCode(llvm::StringRef Code) {
    clang::RewriterTestContext Context;
    clang::FileID ID = Context.createInMemoryFile(FileName, Code);

    std::map<std::string, tooling::Replacements> FileToReplacements;
    change_namespace::ChangeNamespaceTool NamespaceTool(
        OldNamespace, NewNamespace, FilePattern,
        /*AllowedSymbolPatterns*/ {}, &FileToReplacements);
    ast_matchers::MatchFinder Finder;
    NamespaceTool.registerMatchers(&Finder);
    std::unique_ptr<tooling::FrontendActionFactory> Factory =
        tooling::newFrontendActionFactory(&Finder);
    if (!tooling::runToolOnCodeWithArgs(Factory->create(), Code, {"-std=c++11"},
                                        FileName))
      return "";
    formatAndApplyAllReplacements(FileToReplacements, Context.Rewrite);
    return format(Context.getRewrittenText(ID));
  }

  std::string format(llvm::StringRef Code) {
    tooling::Replacements Replaces = format::reformat(
        format::getLLVMStyle(), Code, {tooling::Range(0, Code.size())});
    auto ChangedCode = tooling::applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(ChangedCode));
    if (!ChangedCode) {
      llvm::errs() << llvm::toString(ChangedCode.takeError());
      return "";
    }
    return *ChangedCode;
  }

protected:
  std::string FileName = "input.cc";
  std::string OldNamespace = "na::nb";
  std::string NewNamespace = "x::y";
  std::string FilePattern = "input.cc";
};

TEST_F(ChangeNamespaceTest, NoMatchingNamespace) {
  std::string Code = "namespace na {\n"
                     "namespace nx {\n"
                     "class A {};\n"
                     "} // namespace nx\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "namespace nx {\n"
                         "class A {};\n"
                         "} // namespace nx\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, SimpleMoveWithoutTypeRefs) {
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "class A {};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "\n\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class A {};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, NewNsNestedInOldNs) {
  NewNamespace = "na::nb::nc";
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "class A {};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "namespace nb {\n"
                         "namespace nc {\n"
                         "class A {};\n"
                         "} // namespace nc\n"
                         "} // namespace nb\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, NewNsNestedInOldNsWithSurroundingNewLines) {
  NewNamespace = "na::nb::nc";
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "\n"
                     "class A {};\n"
                     "\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "namespace nb {\n"
                         "namespace nc {\n"
                         "\n"
                         "class A {};\n"
                         "\n"
                         "} // namespace nc\n"
                         "} // namespace nb\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, MoveOldNsWithSurroundingNewLines) {
  NewNamespace = "nx::ny";
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "\n"
                     "class A {};\n"
                     "\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "\n\n"
                         "namespace nx {\n"
                         "namespace ny {\n"
                         "\n"
                         "class A {};\n"
                         "\n"
                         "} // namespace ny\n"
                         "} // namespace nx\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, NewNsNestedInOldNsWithRefs) {
  NewNamespace = "na::nb::nc";
  std::string Code = "namespace na {\n"
                     "class A {};\n"
                     "namespace nb {\n"
                     "class B {};\n"
                     "class C {};\n"
                     "void f() { A a; B b; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "class A {};\n"
                         "namespace nb {\n"
                         "namespace nc {\n"
                         "class B {};\n"
                         "class C {};\n"
                         "void f() { A a; B b; }\n"
                         "} // namespace nc\n"
                         "} // namespace nb\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, SimpleMoveIntoAnotherNestedNamespace) {
  NewNamespace = "na::nc";
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "class A {};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "\n"
                         "namespace nc {\n"
                         "class A {};\n"
                         "} // namespace nc\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, MoveIntoAnotherNestedNamespaceWithRef) {
  NewNamespace = "na::nc";
  std::string Code = "namespace na {\n"
                     "class A {};\n"
                     "namespace nb {\n"
                     "class X { A a; };\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "class A {};\n"
                         "\n"
                         "namespace nc {\n"
                         "class X { A a; };\n"
                         "} // namespace nc\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, MoveIntoExistingNamespaceAndShortenRefs) {
  std::string Code = "namespace x {\n"
                     "namespace z {\n"
                     "class Z {};\n"
                     "} // namespace z\n"
                     "namespace y {\n"
                     "class T {};\n"
                     "} // namespace y\n"
                     "} // namespace x\n"
                     "namespace na {\n"
                     "class A{};\n"
                     "namespace nb {\n"
                     "class X { A a; x::z::Z zz; x::y::T t; };\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace x {\n"
                         "namespace z {\n"
                         "class Z {};\n"
                         "} // namespace z\n"
                         "namespace y {\n"
                         "class T {};\n"
                         "} // namespace y\n"
                         "} // namespace x\n"
                         "namespace na {\n"
                         "class A {};\n\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class X { na::A a; z::Z zz; T t; };\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, SimpleMoveNestedNamespace) {
  NewNamespace = "na::x::y";
  std::string Code = "namespace na {\n"
                     "class A {};\n"
                     "namespace nb {\n"
                     "class B {};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "class A {};\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class B {};\n"
                         "} // namespace y\n"
                         "} // namespace x\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, SimpleMoveWithTypeRefs) {
  std::string Code = "namespace na {\n"
                     "class C_A {};\n"
                     "namespace nc {\n"
                     "class C_C {};"
                     "} // namespace nc\n"
                     "namespace nb {\n"
                     "class C_X {\n"
                     "public:\n"
                     "  C_A a;\n"
                     "  nc::C_C c;\n"
                     "};\n"
                     "class C_Y {\n"
                     "  C_X x;\n"
                     "};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "class C_A {};\n"
                         "namespace nc {\n"
                         "class C_C {};"
                         "} // namespace nc\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class C_X {\n"
                         "public:\n"
                         "  na::C_A a;\n"
                         "  na::nc::C_C c;\n"
                         "};\n"
                         "class C_Y {\n"
                         "  C_X x;\n"
                         "};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, TypeLocInTemplateSpecialization) {
  std::string Code = "namespace na {\n"
                     "class A {};\n"
                     "template <typename T>\n"
                     "class B {};\n"
                     "template <typename T1, typename T2>\n"
                     "class Two {};\n"
                     "namespace nc { class C {}; }\n"
                     "} // na\n"
                     "\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  B<A> b;\n"
                     "  B<nc::C> b_c;\n"
                     "  Two<A, nc::C> two;\n"
                     "}\n"
                     "} // nb\n"
                     "} // na\n";
  std::string Expected = "namespace na {\n"
                         "class A {};\n"
                         "template <typename T>\n"
                         "class B {};\n"
                         "template <typename T1, typename T2>\n"
                         "class Two {};\n"
                         "namespace nc { class C {}; }\n"
                         "} // na\n"
                         "\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  na::B<na::A> b;\n"
                         "  na::B<na::nc::C> b_c;\n"
                         "  na::Two<na::A, na::nc::C> two;\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, LeaveForwardDeclarationBehind) {
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "class FWD;\n"
                     "class FWD2;\n"
                     "class A {\n"
                     "  FWD *fwd;\n"
                     "};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "namespace nb {\n"
                         "class FWD;\n"
                         "class FWD2;\n"
                         "} // namespace nb\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "\n"
                         "class A {\n"
                         "  na::nb::FWD *fwd;\n"
                         "};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, InsertForwardDeclsProperly) {
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "\n"
                     "class FWD;\n"
                     "class FWD2;\n"
                     "class A {\n"
                     "  FWD *fwd;\n"
                     "};\n"
                     "\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "namespace nb {\n"
                         "class FWD;\n"
                         "class FWD2;\n"
                         "} // namespace nb\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "\n"
                         "class A {\n"
                         "  na::nb::FWD *fwd;\n"
                         "};\n"
                         "\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, TemplateClassForwardDeclaration) {
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "class FWD;\n"
                     "template<typename T> class FWD_TEMP;\n"
                     "class A {\n"
                     "  FWD *fwd;\n"
                     "};\n"
                     "template<typename T> class TEMP {};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "namespace nb {\n"
                         "class FWD;\n"
                         "template<typename T> class FWD_TEMP;\n"
                         "} // namespace nb\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "\n"
                         "class A {\n"
                         "  na::nb::FWD *fwd;\n"
                         "};\n"
                         "template<typename T> class TEMP {};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, DontMoveForwardDeclarationInClass) {
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "class A {\n"
                     "  class FWD;\n"
                     "  FWD *fwd;\n"
                     "  template<typename T> class FWD_TEMP;\n"
                     "};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "\n\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class A {\n"
                         "  class FWD;\n"
                         "  FWD *fwd;\n"
                         "  template<typename T> class FWD_TEMP;\n"
                         "};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, MoveFunctions) {
  std::string Code = "namespace na {\n"
                     "class C_A {};\n"
                     "namespace nc {\n"
                     "class C_C {};"
                     "} // namespace nc\n"
                     "namespace nb {\n"
                     "void fwd();\n"
                     "void f(C_A ca, nc::C_C cc) {\n"
                     "  C_A ca_1 = ca;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace na {\n"
                         "class C_A {};\n"
                         "namespace nc {\n"
                         "class C_C {};"
                         "} // namespace nc\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void fwd();\n"
                         "void f(na::C_A ca, na::nc::C_C cc) {\n"
                         "  na::C_A ca_1 = ca;\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, FixUsingShadowDecl) {
  std::string Code = "class GLOB {};\n"
                     "using BLOG = GLOB;\n"
                     "namespace na {\n"
                     "namespace nc {\n"
                     "class SAME {};\n"
                     "}\n"
                     "namespace nd {\n"
                     "class SAME {};\n"
                     "}\n"
                     "namespace nb {\n"
                     "using nc::SAME;\n"
                     "using YO = nd::SAME;\n"
                     "typedef nd::SAME IDENTICAL;\n"
                     "void f(nd::SAME Same) {}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "class GLOB {};\n"
                         "using BLOG = GLOB;\n"
                         "namespace na {\n"
                         "namespace nc {\n"
                         "class SAME {};\n"
                         "}\n"
                         "namespace nd {\n"
                         "class SAME {};\n"
                         "}\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "using ::na::nc::SAME;\n"
                         "using YO = na::nd::SAME;\n"
                         "typedef na::nd::SAME IDENTICAL;\n"
                         "void f(na::nd::SAME Same) {}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, DontFixUsingShadowDeclInClasses) {
  std::string Code = "namespace na {\n"
                     "class A {};\n"
                     "class Base { public: Base() {} void m() {} };\n"
                     "namespace nb {\n"
                     "class D : public Base {\n"
                     "public:\n"
                     "  using AA = A; using B = Base;\n"
                     "  using Base::m; using Base::Base;\n"
                     "};"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace na {\n"
                         "class A {};\n"
                         "class Base { public: Base() {} void m() {} };\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class D : public na::Base {\n"
                         "public:\n"
                         "  using AA = na::A; using B = na::Base;\n"
                         "  using Base::m; using Base::Base;\n"
                         "};"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, TypeInNestedNameSpecifier) {
  std::string Code =
      "namespace na {\n"
      "class C_A {\n"
      "public:\n"
      "  class Nested {\n"
      "    public:\n"
      "      static int NestedX;\n"
      "      static void nestedFunc() {}\n"
      "  };\n"
      "};\n"
      "namespace nb {\n"
      "class C_X {\n"
      "  C_A na;\n"
      "  C_A::Nested nested;\n"
      "  void f() {\n"
      "    C_A::Nested::nestedFunc();\n"
      "    int X = C_A::Nested::NestedX;\n"
      "  }\n"
      "};\n"
      "}  // namespace nb\n"
      "}  // namespace na\n";
  std::string Expected =
      "namespace na {\n"
      "class C_A {\n"
      "public:\n"
      "  class Nested {\n"
      "    public:\n"
      "      static int NestedX;\n"
      "      static void nestedFunc() {}\n"
      "  };\n"
      "};\n"
      "\n"
      "}  // namespace na\n"
      "namespace x {\n"
      "namespace y {\n"
      "class C_X {\n"
      "  na::C_A na;\n"
      "  na::C_A::Nested nested;\n"
      "  void f() {\n"
      "    na::C_A::Nested::nestedFunc();\n"
      "    int X = na::C_A::Nested::NestedX;\n"
      "  }\n"
      "};\n"
      "}  // namespace y\n"
      "}  // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, FixFunctionNameSpecifiers) {
  std::string Code =
      "namespace na {\n"
      "class A {\n"
      "public:\n"
      "  static void f() {}\n"
      "  static void g();\n"
      "};\n"
      "void A::g() {}"
      "void a_f() {}\n"
      "static void static_f() {}\n"
      "namespace nb {\n"
      "void f() { a_f(); static_f(); A::f(); }\n"
      "void g() { f(); A::g(); }\n"
      "}  // namespace nb\n"
      "}  // namespace na\n";
  std::string Expected =
      "namespace na {\n"
      "class A {\n"
      "public:\n"
      "  static void f() {}\n"
      "  static void g();\n"
      "};\n"
      "void A::g() {}"
      "void a_f() {}\n"
      "static void static_f() {}\n"
      "\n"
      "}  // namespace na\n"
      "namespace x {\n"
      "namespace y {\n"
      "void f() { na::a_f(); na::static_f(); na::A::f(); }\n"
      "void g() { f(); na::A::g(); }\n"
      "}  // namespace y\n"
      "}  // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, FixOverloadedOperatorFunctionNameSpecifiers) {
  std::string Code =
      "namespace na {\n"
      "class A {\n"
      "public:\n"
      "  int x;\n"
      "  bool operator==(const A &RHS) const { return x == RHS.x; }\n"
      "};\n"
      "bool operator<(const A &LHS, const A &RHS) { return LHS.x == RHS.x; }\n"
      "namespace nb {\n"
      "bool f() {\n"
      "  A x, y;\n"
      "  auto f = operator<;\n"
      "  return (x == y) && (x < y) && (operator<(x, y));\n"
      "}\n"
      "}  // namespace nb\n"
      "}  // namespace na\n";
  std::string Expected =
      "namespace na {\n"
      "class A {\n"
      "public:\n"
      "  int x;\n"
      "  bool operator==(const A &RHS) const { return x == RHS.x; }\n"
      "};\n"
      "bool operator<(const A &LHS, const A &RHS) { return LHS.x == RHS.x; }\n"
      "\n"
      "}  // namespace na\n"
      "namespace x {\n"
      "namespace y {\n"
      "bool f() {\n"
      "  na::A x, y;\n"
      "  auto f = na::operator<;\n"
      // FIXME: function calls to overloaded operators are not fixed now even if
      // they are referenced by qualified names.
      "  return (x == y) && (x < y) && (operator<(x,y));\n"
      "}\n"
      "}  // namespace y\n"
      "}  // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, FixNonCallingFunctionReferences) {
  std::string Code = "namespace na {\n"
                     "class A {\n"
                     "public:\n"
                     "  static void f() {}\n"
                     "};\n"
                     "void a_f() {}\n"
                     "static void s_f() {}\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  auto *ref1 = A::f;\n"
                     "  auto *ref2 = a_f;\n"
                     "  auto *ref3 = s_f;\n"
                     "}\n"
                     "}  // namespace nb\n"
                     "}  // namespace na\n";
  std::string Expected =
      "namespace na {\n"
      "class A {\n"
      "public:\n"
      "  static void f() {}\n"
      "};\n"
      "void a_f() {}\n"
      "static void s_f() {}\n"
      "\n"
      "}  // namespace na\n"
      "namespace x {\n"
      "namespace y {\n"
      "void f() {\n"
      "  auto *ref1 = na::A::f;\n"
      "  auto *ref2 = na::a_f;\n"
      "  auto *ref3 = na::s_f;\n"
      "}\n"
      "}  // namespace y\n"
      "}  // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, MoveAndFixGlobalVariables) {
  std::string Code = "namespace na {\n"
                     "int GlobA;\n"
                     "static int GlobAStatic = 0;\n"
                     "namespace nc { int GlobC; }\n"
                     "namespace nb {\n"
                     "int GlobB;\n"
                     "void f() {\n"
                     "  int a = GlobA;\n"
                     "  int b = GlobAStatic;\n"
                     "  int c = nc::GlobC;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace na {\n"
                         "int GlobA;\n"
                         "static int GlobAStatic = 0;\n"
                         "namespace nc { int GlobC; }\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "int GlobB;\n"
                         "void f() {\n"
                         "  int a = na::GlobA;\n"
                         "  int b = na::GlobAStatic;\n"
                         "  int c = na::nc::GlobC;\n"
                         "}\n"
                         "}  // namespace y\n"
                         "}  // namespace x\n";

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, DoNotFixStaticVariableOfClass) {
  std::string Code = "namespace na {\n"
                     "class A {\n"
                     "public:\n"
                     "static int A1;\n"
                     "static int A2;\n"
                     "};\n"
                     "int A::A1 = 0;\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  int a = A::A1; int b = A::A2;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace na {\n"
                         "class A {\n"
                         "public:\n"
                         "static int A1;\n"
                         "static int A2;\n"
                         "};\n"
                         "int A::A1 = 0;\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  int a = na::A::A1; int b = na::A::A2;\n"
                         "}\n"
                         "}  // namespace y\n"
                         "}  // namespace x\n";

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, NoMisplaceAtEOF) {
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "class A;\n"
                     "class B {};\n"
                     "}"
                     "}";
  std::string Expected = "namespace na {\n"
                         "namespace nb {\n"
                         "class A;\n"
                         "}\n"
                         "}\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "\n"
                         "class B {};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, CommentsBeforeMovedClass) {
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "\n\n"
                     "// Wild comments.\n"
                     "\n"
                     "// Comments.\n"
                     "// More comments.\n"
                     "class B {\n"
                     "  // Private comments.\n"
                     "  int a;\n"
                     "};\n"
                     "}\n"
                     "}";
  std::string Expected = "\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "\n\n"
                         "// Wild comments.\n"
                         "\n"
                         "// Comments.\n"
                         "// More comments.\n"
                         "class B {\n"
                         "  // Private comments.\n"
                         "  int a;\n"
                         "};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingShadowDeclInGlobal) {
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "void GFunc() {}\n"
                     "}\n"
                     "using glob::Glob;\n"
                     "using glob::GFunc;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() { Glob g; GFunc(); }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "void GFunc() {}\n"
                         "}\n"
                         "using glob::Glob;\n"
                         "using glob::GFunc;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() { Glob g; GFunc(); }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingShadowDeclsInAnonymousNamespaces) {
  std::string Code = "namespace util {\n"
                     "class Util {};\n"
                     "void func() {}\n"
                     "}\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "namespace {\n"
                     "using ::util::Util;\n"
                     "using ::util::func;\n"
                     "void f() { Util u; func(); }\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace util {\n"
                         "class Util {};\n"
                         "void func() {}\n"
                         "} // namespace util\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "namespace {\n"
                         "using ::util::Util;\n"
                         "using ::util::func;\n"
                         "void f() { Util u; func(); }\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingNamespaceInGlobal) {
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "}\n"
                     "using namespace glob;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() { Glob g; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "}\n"
                         "using namespace glob;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() { Glob g; }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, NamespaceAliasInGlobal) {
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "}\n"
                     "namespace glob2 { class Glob2 {}; }\n"
                     "namespace gl = glob;\n"
                     "namespace gl2 = glob2;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() { gl::Glob g; gl2::Glob2 g2; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected =
      "namespace glob {\n"
      "class Glob {};\n"
      "}\n"
      "namespace glob2 { class Glob2 {}; }\n"
      "namespace gl = glob;\n"
      "namespace gl2 = glob2;\n"
      "\n"
      "namespace x {\n"
      "namespace y {\n"
      "void f() { gl::Glob g; gl2::Glob2 g2; }\n"
      "} // namespace y\n"
      "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, NamespaceAliasInNamespace) {
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "}\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "namespace gl = glob;\n"
                     "void f() { gl::Glob g; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "}\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "namespace gl = glob;\n"
                         "void f() { gl::Glob g; }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, NamespaceAliasInAncestorNamespace) {
  NewNamespace = "na::nx";
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "}\n"
                     "namespace other { namespace gl = glob; }\n"
                     "namespace na {\n"
                     "namespace ga = glob;\n"
                     "namespace nb {\n"
                     "void f() { ga::Glob g; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "}\n"
                         "namespace other { namespace gl = glob; }\n"
                         "namespace na {\n"
                         "namespace ga = glob;\n"
                         "\n"
                         "namespace nx {\n"
                         "void f() { ga::Glob g; }\n"
                         "} // namespace nx\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, NamespaceAliasInOtherNamespace) {
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "}\n"
                     "namespace other { namespace gl = glob; }\n"
                     "namespace na {\n"
                     "namespace ga = glob;\n"
                     "namespace nb {\n"
                     "void f() { glob::Glob g; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "}\n"
                         "namespace other { namespace gl = glob; }\n"
                         "namespace na {\n"
                         "namespace ga = glob;\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() { glob::Glob g; }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingDeclAfterReference) {
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "}\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() { glob::Glob g; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n"
                     "using glob::Glob;\n"
                     "using namespace glob;\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "}\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() { glob::Glob g; }\n"
                         "} // namespace y\n"
                         "} // namespace x\n"
                         "using glob::Glob;\n"
                         "using namespace glob;\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingNamespaceAfterReference) {
  NewNamespace = "na::nc";
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "}\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() { glob::Glob g; }\n"
                     "} // namespace nb\n"
                     "using namespace glob;\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "}\n"
                         "namespace na {\n"
                         "\n"
                         "namespace nc {\n"
                         "void f() { glob::Glob g; }\n"
                         "} // namespace nc\n"
                         "using namespace glob;\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingNamespaceAndUsingShadowInGlobal) {
  std::string Code = "namespace glob1 {\n"
                     "namespace glob2 {\n"
                     "class Glob {};\n"
                     "}\n"
                     "}\n"
                     "using glob1::glob2::Glob;\n"
                     "using namespace glob1;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() { Glob g; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob1 {\n"
                         "namespace glob2 {\n"
                         "class Glob {};\n"
                         "}\n"
                         "}\n"
                         "using glob1::glob2::Glob;\n"
                         "using namespace glob1;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() { Glob g; }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingAliasInGlobal) {
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "}\n"
                     "using GLB = glob::Glob;\n"
                     "using BLG = glob::Glob;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() { GLB g; BLG blg; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "}\n"
                         "using GLB = glob::Glob;\n"
                         "using BLG = glob::Glob;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() { GLB g; BLG blg; }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingShadowDeclAndMovedNamespace) {
  std::string Code = "namespace na { class C_A {};\n }\n"
                     "using na::C_A;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "class C_X {\n"
                     "public:\n"
                     "  C_A a;\n"
                     "};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na { class C_A {};\n }\n"
                         "using na::C_A;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class C_X {\n"
                         "public:\n"
                         "  C_A a;\n"
                         "};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingNamespaceDeclAndMovedNamespace) {
  std::string Code = "namespace na { class C_A {};\n }\n"
                     "using namespace na;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "class C_X {\n"
                     "public:\n"
                     "  C_A ca;\n"
                     "};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na { class C_A {};\n }\n"
                         "using namespace na;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class C_X {\n"
                         "public:\n"
                         "  C_A ca;\n"
                         "};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingShadowDeclInFunction) {
  std::string Code = "namespace glob {\n"
                     "class Glob {};\n"
                     "}\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  using glob::Glob;\n"
                     "  Glob g;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "}\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  using ::glob::Glob;\n"
                         "  Glob g;\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingShadowDeclInClass) {
  std::string Code = "namespace na { class C_A {}; }\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  using ::na::C_A;\n"
                     "  C_A ca;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na { class C_A {}; }\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  using ::na::C_A;\n"
                         "  C_A ca;\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingDeclInMovedNamespace) {
  std::string Code = "namespace nx { void f(); }\n"
                     "namespace na {\n"
                     "using nx::f;\n"
                     "namespace nb {\n"
                     "void d() { f(); }\n"
                     "} // nb\n"
                     "} // na\n";

  std::string Expected = "namespace nx { void f(); }\n"
                         "namespace na {\n"
                         "using nx::f;\n"
                         "\n"
                         "} // na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void d() { nx::f(); }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingDeclInMovedNamespaceNotNested) {
  OldNamespace = "na";
  std::string Code = "namespace nx { void f(); }\n"
                     "namespace na {\n"
                     "using ::nx::f;\n"
                     "void d() { f(); }\n"
                     "} // na\n";

  std::string Expected = "namespace nx { void f(); }\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "using ::nx::f;\n"
                         "void d() { f(); }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingDeclInMovedNamespaceMultiNested) {
  OldNamespace = "a::b::c::d";
  NewNamespace = "a::b::x::y";
  std::string Code = "namespace nx { void f(); void g(); }\n"
                     "namespace a {\n"
                     "namespace b {\n"
                     "using ::nx::f;\n"
                     "namespace c {\n"
                     "using ::nx::g;\n"
                     "namespace d {\n"
                     "void d() { f(); g(); }\n"
                     "} // d\n"
                     "} // c\n"
                     "} // b\n"
                     "} // a\n";

  std::string Expected = "namespace nx { void f(); void g(); }\n"
                         "namespace a {\n"
                         "namespace b {\n"
                         "using ::nx::f;\n"
                         "namespace c {\n"
                         "using ::nx::g;\n"
                         "\n"
                         "} // c\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void d() { f(); nx::g(); }\n"
                         "} // namespace y\n"
                         "} // namespace x\n"
                         "} // b\n"
                         "} // a\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingShadowDeclInTheParentOfOldNamespace) {
  OldNamespace = "nb::nc";
  NewNamespace = "nb::nd";
  std::string Code = "namespace na { class A {}; }\n"
                     "namespace nb {\n"
                     "using na::A;\n"
                     "namespace nc {\n"
                     "void d() { A a; }\n"
                     "} // nc\n"
                     "} // nb\n";

  std::string Expected = "namespace na { class A {}; }\n"
                         "namespace nb {\n"
                         "using na::A;\n"
                         "\n"
                         "namespace nd {\n"
                         "void d() { A a; }\n"
                         "} // namespace nd\n"
                         "} // nb\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingShadowDeclInOldNamespace) {
  OldNamespace = "nb";
  NewNamespace = "nc";
  std::string Code = "namespace na { class A {}; }\n"
                     "namespace nb {\n"
                     "using na::A;\n"
                     "void d() { A a; }\n"
                     "struct X { A a; };\n"
                     "} // nb\n";

  std::string Expected = "namespace na { class A {}; }\n"
                         "\n"
                         "namespace nc {\n"
                         "using ::na::A;\n"
                         "void d() { A a; }\n"
                         "struct X { A a; };\n"
                         "} // namespace nc\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingShadowDeclOfTemplateClass) {
  OldNamespace = "nb";
  NewNamespace = "nc";
  std::string Code = "namespace na {\n"
                     "template <typename T>\n"
                     "class A { T t; };\n"
                     "} // namespace na\n"
                     "namespace nb {\n"
                     "using na::A;\n"
                     "void d() { A<int> a; }\n"
                     "} // nb\n";

  std::string Expected = "namespace na {\n"
                         "template <typename T>\n"
                         "class A { T t; };\n"
                         "} // namespace na\n"
                         "\n"
                         "namespace nc {\n"
                         "using ::na::A;\n"
                         "void d() { A<int> a; }\n"
                         "} // namespace nc\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingShadowDeclOfTemplateFunction) {
  OldNamespace = "nb";
  NewNamespace = "nc";
  std::string Code = "namespace na {\n"
                     "template <typename T>\n"
                     "void f() { T t; };\n"
                     "} // namespace na\n"
                     "namespace nb {\n"
                     "using na::f;\n"
                     "void d() { f<int>(); }\n"
                     "} // nb\n";

  std::string Expected = "namespace na {\n"
                         "template <typename T>\n"
                         "void f() { T t; };\n"
                         "} // namespace na\n"
                         "\n"
                         "namespace nc {\n"
                         "using ::na::f;\n"
                         "void d() { f<int>(); }\n"
                         "} // namespace nc\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingAliasDecl) {
  std::string Code =
      "namespace nx { namespace ny { class X {}; } }\n"
      "namespace na {\n"
      "namespace nb {\n"
      "using Y = nx::ny::X;\n"
      "void f() { Y y; }\n"
      "} // namespace nb\n"
      "} // namespace na\n";

  std::string Expected = "namespace nx { namespace ny { class X {}; } }\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "using Y = nx::ny::X;\n"
                         "void f() { Y y; }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingAliasDeclInGlobal) {
  std::string Code =
      "namespace nx { namespace ny { class X {}; } }\n"
      "using Y = nx::ny::X;\n"
      "namespace na {\n"
      "namespace nb {\n"
      "void f() { Y y; }\n"
      "} // namespace nb\n"
      "} // namespace na\n";

  std::string Expected = "namespace nx { namespace ny { class X {}; } }\n"
                         "using Y = nx::ny::X;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() { Y y; }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}


TEST_F(ChangeNamespaceTest, TypedefAliasDecl) {
  std::string Code =
      "namespace nx { namespace ny { class X {}; } }\n"
      "namespace na {\n"
      "namespace nb {\n"
      "typedef nx::ny::X Y;\n"
      "void f() { Y y; }\n"
      "} // namespace nb\n"
      "} // namespace na\n";

  std::string Expected = "namespace nx { namespace ny { class X {}; } }\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "typedef nx::ny::X Y;\n"
                         "void f() { Y y; }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, DerivedClassWithConstructors) {
  std::string Code =
      "namespace nx { namespace ny { class X { public: X(int i) {} }; } }\n"
      "namespace na {\n"
      "namespace nb {\n"
      "class A : public nx::ny::X {\n"
      "public:\n"
      "  A() : X(0) {}\n"
      "  A(int i);\n"
      "};\n"
      "A::A(int i) : X(i) {}\n"
      "} // namespace nb\n"
      "} // namespace na\n";
  std::string Expected =
      "namespace nx { namespace ny { class X { public: X(int i) {} }; } }\n"
      "\n\n"
      "namespace x {\n"
      "namespace y {\n"
      "class A : public nx::ny::X {\n"
      "public:\n"
      "  A() : X(0) {}\n"
      "  A(int i);\n"
      "};\n"
      "A::A(int i) : X(i) {}\n"
      "} // namespace y\n"
      "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, DerivedClassWithQualifiedConstructors) {
  std::string Code =
      "namespace nx { namespace ny { class X { public: X(int i) {} }; } }\n"
      "namespace na {\n"
      "namespace nb {\n"
      "class A : public nx::ny::X {\n"
      "public:\n"
      "  A() : X::X(0) {}\n"
      "  A(int i);\n"
      "};\n"
      "A::A(int i) : X::X(i) {}\n"
      "} // namespace nb\n"
      "} // namespace na\n";
  std::string Expected =
      "namespace nx { namespace ny { class X { public: X(int i) {} }; } }\n"
      "\n\n"
      "namespace x {\n"
      "namespace y {\n"
      "class A : public nx::ny::X {\n"
      "public:\n"
      "  A() : X::X(0) {}\n"
      "  A(int i);\n"
      "};\n"
      "A::A(int i) : X::X(i) {}\n"
      "} // namespace y\n"
      "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, DerivedClassWithConstructorsAndTypeRefs) {
  std::string Code =
      "namespace nx { namespace ny { class X { public: X(int i) {} }; } }\n"
      "namespace na {\n"
      "namespace nb {\n"
      "class A : public nx::ny::X {\n"
      "public:\n"
      "  A() : X(0) {}\n"
      "  A(int i);\n"
      "};\n"
      "A::A(int i) : X(i) { X x(1);}\n"
      "} // namespace nb\n"
      "} // namespace na\n";
  std::string Expected =
      "namespace nx { namespace ny { class X { public: X(int i) {} }; } }\n"
      "\n\n"
      "namespace x {\n"
      "namespace y {\n"
      "class A : public nx::ny::X {\n"
      "public:\n"
      "  A() : X(0) {}\n"
      "  A(int i);\n"
      "};\n"
      "A::A(int i) : X(i) { nx::ny::X x(1);}\n"
      "} // namespace y\n"
      "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, MoveToGlobalNamespace) {
  NewNamespace = "";
  std::string Code = "namespace na {\n"
                     "class C_A {};\n"
                     "namespace nc {\n"
                     "class C_C {};"
                     "} // namespace nc\n"
                     "namespace nb {\n"
                     "class C_X {\n"
                     "public:\n"
                     "  C_A a;\n"
                     "  nc::C_C c;\n"
                     "};\n"
                     "class C_Y {\n"
                     "  C_X x;\n"
                     "};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "class C_A {};\n"
                         "namespace nc {\n"
                         "class C_C {};"
                         "} // namespace nc\n"
                         "\n"
                         "} // namespace na\n"
                         "class C_X {\n"
                         "public:\n"
                         "  na::C_A a;\n"
                         "  na::nc::C_C c;\n"
                         "};\n"
                         "class C_Y {\n"
                         "  C_X x;\n"
                         "};\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, KeepGlobalSpecifier) {
  std::string Code = "class Glob {};\n"
                      "namespace na {\n"
                     "class C_A {};\n"
                     "namespace nc {\n"
                     "class C_C {};"
                     "} // namespace nc\n"
                     "namespace nb {\n"
                     "class C_X {\n"
                     "public:\n"
                     "  ::Glob glob_1;\n"
                     "  Glob glob_2;\n"
                     "  C_A a_1;\n"
                     "  ::na::C_A a_2;\n"
                     "  nc::C_C c;\n"
                     "};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "class Glob {};\n"
                         "namespace na {\n"
                         "class C_A {};\n"
                         "namespace nc {\n"
                         "class C_C {};"
                         "} // namespace nc\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class C_X {\n"
                         "public:\n"
                         "  ::Glob glob_1;\n"
                         "  Glob glob_2;\n"
                         "  na::C_A a_1;\n"
                         "  ::na::C_A a_2;\n"
                         "  na::nc::C_C c;\n"
                         "};\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, UsingAliasInTemplate) {
  NewNamespace = "na::nb::nc";
  std::string Code = "namespace some_ns {\n"
                     "template <typename T, typename S>\n"
                     "class G {};\n"
                     "} // namespace some_ns\n"
                     "namespace na {\n"
                     "template<typename P>\n"
                     "using GG = some_ns::G<int, P>;\n"
                     "} // namespace na\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  GG<float> g;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace some_ns {\n"
                         "template <typename T, typename S>\n"
                         "class G {};\n"
                         "} // namespace some_ns\n"
                         "namespace na {\n"
                         "template<typename P>\n"
                         "using GG = some_ns::G<int, P>;\n"
                         "} // namespace na\n"
                         "namespace na {\n"
                         "namespace nb {\n"
                         "namespace nc {\n"
                         "void f() {\n"
                         "  GG<float> g;\n"
                         "}\n"
                         "} // namespace nc\n"
                         "} // namespace nb\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, TemplateUsingAliasInBaseClass) {
  NewNamespace = "na::nb::nc";
  std::string Code = "namespace some_ns {\n"
                     "template <typename T, typename S>\n"
                     "class G {};\n"
                     "} // namespace some_ns\n"
                     "namespace na {\n"
                     "class Base {\n"
                     "public:\n"
                     "  template<typename P>\n"
                     "  using GG = some_ns::G<int, P>;\n"
                     "\n"
                     "  struct Nested {};\n"
                     "};\n"
                     "class Derived : public Base {};\n"
                     "} // namespace na\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  Derived::GG<float> g;\n"
                     "  const Derived::GG<int> gg;\n"
                     "  const Derived::GG<int>* gg_ptr;\n"
                     "  struct Derived::Nested nested;\n"
                     "  const struct Derived::Nested *nested_ptr;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace some_ns {\n"
                         "template <typename T, typename S>\n"
                         "class G {};\n"
                         "} // namespace some_ns\n"
                         "namespace na {\n"
                         "class Base {\n"
                         "public:\n"
                         "  template<typename P>\n"
                         "  using GG = some_ns::G<int, P>;\n"
                         "\n"
                         "  struct Nested {};\n"
                         "};\n"
                         "class Derived : public Base {};\n"
                         "} // namespace na\n"
                         "namespace na {\n"
                         "namespace nb {\n"
                         "namespace nc {\n"
                         "void f() {\n"
                         "  Derived::GG<float> g;\n"
                         "  const Derived::GG<int> gg;\n"
                         "  const Derived::GG<int>* gg_ptr;\n"
                         "  struct Derived::Nested nested;\n"
                         "  const struct Derived::Nested *nested_ptr;\n"
                         "}\n"
                         "} // namespace nc\n"
                         "} // namespace nb\n"
                         "} // namespace na\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, ExistingNamespaceConflictWithNewNamespace) {
  OldNamespace = "nx";
  NewNamespace = "ny::na::nc";
  std::string Code = "namespace na {\n"
                     "class A {};\n"
                     "} // namespace na\n"
                     "namespace nb {\n"
                     "class B {};\n"
                     "} // namespace nb\n"
                     "namespace nx {\n"
                     "class X {\n"
                     " na::A a; nb::B b;\n"
                     "};\n"
                     "} // namespace nx\n";
  std::string Expected = "namespace na {\n"
                         "class A {};\n"
                         "} // namespace na\n"
                         "namespace nb {\n"
                         "class B {};\n"
                         "} // namespace nb\n"
                         "\n"
                         "namespace ny {\n"
                         "namespace na {\n"
                         "namespace nc {\n"
                         "class X {\n"
                         " ::na::A a; nb::B b;\n"
                         "};\n"
                         "} // namespace nc\n"
                         "} // namespace na\n"
                         "} // namespace ny\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, SymbolConflictWithNewNamespace) {
  OldNamespace = "nx";
  NewNamespace = "ny::na::nc";
  std::string Code = "namespace na {\n"
                     "class A {};\n"
                     "namespace nb {\n"
                     "class B {};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n"
                     "namespace ny {\n"
                     "class Y {};\n"
                     "}\n"
                     "namespace nx {\n"
                     "class X {\n"
                     "  na::A a; na::nb::B b;\n"
                     "  ny::Y y;"
                     "};\n"
                     "} // namespace nx\n";
  std::string Expected = "namespace na {\n"
                         "class A {};\n"
                         "namespace nb {\n"
                         "class B {};\n"
                         "} // namespace nb\n"
                         "} // namespace na\n"
                         "namespace ny {\n"
                         "class Y {};\n"
                         "}\n"
                         "\n"
                         "namespace ny {\n"
                         "namespace na {\n"
                         "namespace nc {\n"
                         "class X {\n"
                         "  ::na::A a; ::na::nb::B b;\n"
                         "  Y y;\n"
                         "};\n"
                         "} // namespace nc\n"
                         "} // namespace na\n"
                         "} // namespace ny\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, ShortenNamespaceSpecifier) {
  OldNamespace = "nx";
  NewNamespace = "ny::na";
  std::string Code = "class G {};\n"
                     "namespace ny {\n"
                     "class Y {};\n"
                     "namespace na {\n"
                     "class A {};\n"
                     "namespace nc { class C {}; } // namespace nc\n"
                     "}\n // namespace na\n"
                     "}\n // namespace ny\n"
                     "namespace nx {\n"
                     "class X {\n"
                     " G g; ny::Y y; ny::na::A a; ny::na::nc::C c;\n"
                     "};\n"
                     "} // namespace nx\n";
  std::string Expected = "class G {};\n"
                         "namespace ny {\n"
                         "class Y {};\n"
                         "namespace na {\n"
                         "class A {};\n"
                         "namespace nc { class C {}; } // namespace nc\n"
                         "}\n // namespace na\n"
                         "}\n // namespace ny\n"
                         "\n"
                         "namespace ny {\n"
                         "namespace na {\n"
                         "class X {\n"
                         " G g; Y y; A a; nc::C c;\n"
                         "};\n"
                         "} // namespace na\n"
                         "} // namespace ny\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, ShortenNamespaceSpecifierInAnonymousNamespace) {
  OldNamespace = "nx";
  NewNamespace = "ny::na";
  std::string Code = "class G {};\n"
                     "namespace ny {\n"
                     "class Y {};\n"
                     "namespace na {\n"
                     "class A {};\n"
                     "namespace nc { class C {}; } // namespace nc\n"
                     "}\n // namespace na\n"
                     "}\n // namespace ny\n"
                     "namespace nx {\n"
                     "namespace {\n"
                     "class X {\n"
                     " G g; ::ny::Y y; ::ny::na::A a; ::ny::na::nc::C c;\n"
                     "};\n"
                     "} // namespace\n"
                     "} // namespace nx\n";
  std::string Expected = "class G {};\n"
                         "namespace ny {\n"
                         "class Y {};\n"
                         "namespace na {\n"
                         "class A {};\n"
                         "namespace nc { class C {}; } // namespace nc\n"
                         "}\n // namespace na\n"
                         "}\n // namespace ny\n"
                         "\n"
                         "namespace ny {\n"
                         "namespace na {\n"
                         "namespace {\n"
                         "class X {\n"
                         " G g; Y y; A a; nc::C c;\n"
                         "};\n"
                         "} // namespace\n"
                         "} // namespace na\n"
                         "} // namespace ny\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, SimpleMoveEnum) {
  std::string Code = "namespace na {\n"
                     "namespace nb {\n"
                     "enum class X { X1, X2 };\n"
                     "enum Y { Y1, Y2 };\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "\n\nnamespace x {\n"
                         "namespace y {\n"
                         "enum class X { X1, X2 };\n"
                         "enum Y { Y1, Y2 };\n"
                         "} // namespace y\n"
                         "} // namespace x\n";

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, ReferencesToEnums) {
  std::string Code = "enum Glob { G1, G2 };\n"
                     "namespace na {\n"
                     "enum class X { X1 };\n"
                     "enum Y { Y1, Y2 };\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  Glob g1 = Glob::G1;\n"
                     "  Glob g2 = G2;\n"
                     "  X x1 = X::X1;\n"
                     "  Y y1 = Y::Y1;\n"
                     "  Y y2 = Y2;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "enum Glob { G1, G2 };\n"
                         "namespace na {\n"
                         "enum class X { X1 };\n"
                         "enum Y { Y1, Y2 };\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  Glob g1 = Glob::G1;\n"
                         "  Glob g2 = G2;\n"
                         "  na::X x1 = na::X::X1;\n"
                         "  na::Y y1 = na::Y::Y1;\n"
                         "  na::Y y2 = na::Y2;\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, NoRedundantEnumUpdate) {
  std::string Code = "namespace ns {\n"
                     "enum class X { X1 };\n"
                     "enum Y { Y1, Y2 };\n"
                     "} // namespace ns\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  ns::X x1 = ns::X::X1;\n"
                     "  ns::Y y1 = ns::Y::Y1;\n"
                     "  ns::Y y2 = ns::Y2;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace ns {\n"
                         "enum class X { X1 };\n"
                         "enum Y { Y1, Y2 };\n"
                         "} // namespace ns\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  ns::X x1 = ns::X::X1;\n"
                         "  ns::Y y1 = ns::Y::Y1;\n"
                         "  ns::Y y2 = ns::Y2;\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  ;

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, EnumsAndUsingShadows) {
  std::string Code = "namespace ns {\n"
                     "enum class X { X1 };\n"
                     "enum Y { Y1, Y2, Y3 };\n"
                     "} // namespace ns\n"
                     "using ns::X;\n"
                     "using ns::Y;\n"
                     "using ns::Y::Y2;\n"
                     "using ns::Y::Y3;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  X x1 = X::X1;\n"
                     "  Y y1 = Y::Y1;\n"
                     "  Y y2 = Y2;\n"
                     "  Y y3 = Y3;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace ns {\n"
                         "enum class X { X1 };\n"
                         "enum Y { Y1, Y2, Y3 };\n"
                         "} // namespace ns\n"
                         "using ns::X;\n"
                         "using ns::Y;\n"
                         "using ns::Y::Y2;\n"
                         "using ns::Y::Y3;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  X x1 = X::X1;\n"
                         "  Y y1 = Y::Y1;\n"
                         "  Y y2 = Y2;\n"
                         "  Y y3 = Y3;\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, EnumsAndAliases) {
  std::string Code = "namespace ns {\n"
                     "enum class X { X1 };\n"
                     "enum Y { Y1, Y2, Y3 };\n"
                     "} // namespace ns\n"
                     "typedef ns::X TX;\n"
                     "typedef ns::Y TY;\n"
                     "using UX = ns::X;\n"
                     "using UY = ns::Y;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  ns::X x1 = ns::X::X1;\n"
                     "  TX tx1 = TX::X1;\n"
                     "  UX ux1 = UX::X1;\n"
                     "  ns::Y y1 = ns::Y::Y1;\n"
                     "  TY ty1 = TY::Y1;\n"
                     "  UY uy1 = UY::Y1;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace ns {\n"
                         "enum class X { X1 };\n"
                         "enum Y { Y1, Y2, Y3 };\n"
                         "} // namespace ns\n"
                         "typedef ns::X TX;\n"
                         "typedef ns::Y TY;\n"
                         "using UX = ns::X;\n"
                         "using UY = ns::Y;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  ns::X x1 = ns::X::X1;\n"
                         "  TX tx1 = TX::X1;\n"
                         "  UX ux1 = UX::X1;\n"
                         "  ns::Y y1 = ns::Y::Y1;\n"
                         "  TY ty1 = TY::Y1;\n"
                         "  UY uy1 = UY::Y1;\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, EnumInClass) {
  std::string Code = "namespace na {\n"
                     "struct X { enum E { E1 }; };\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  X::E e = X::E1;\n"
                     "  X::E ee = X::E::E1;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "struct X { enum E { E1 }; };\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  na::X::E e = na::X::E1;\n"
                         "  na::X::E ee = na::X::E::E1;\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, TypeAsTemplateParameter) {
  std::string Code = "namespace na {\n"
                     "struct X {};\n"
                     "namespace nb {\n"
                     "template <typename TT>\n"
                     "void TempTemp(const TT& t) {\n"
                     "  TT tmp;\n"
                     "}\n"
                     "template <typename T>\n"
                     "void Temp(const T& t) {\n"
                     "  T tmp = t;\n"
                     "  TempTemp(tmp);\n"
                     "  TempTemp(t);\n"
                     "}\n"
                     "void f() {\n"
                     "  X x;\n"
                     "  Temp(x);\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "struct X {};\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "template <typename TT>\n"
                         "void TempTemp(const TT& t) {\n"
                         "  TT tmp;\n"
                         "}\n"
                         "template <typename T>\n"
                         "void Temp(const T& t) {\n"
                         "  T tmp = t;\n"
                         "  TempTemp(tmp);\n"
                         "  TempTemp(t);\n"
                         "}\n"
                         "void f() {\n"
                         "  na::X x;\n"
                         "  Temp(x);\n"
                         "}\n"
                         "} // namespace y\n"
                         "} // namespace x\n";

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, DefaultMoveConstructors) {
  std::string Code = "namespace na {\n"
                     "class B {\n"
                     " public:\n"
                     "  B() = default;\n"
                     "  // Allow move only.\n"
                     "  B(B&&) = default;\n"
                     "  B& operator=(B&&) = default;\n"
                     "  B(const B&) = delete;\n"
                     "  B& operator=(const B&) = delete;\n"
                     " private:\n"
                     "  int ref_;\n"
                     "};\n"
                     "} // namespace na\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "class A {\n"
                     "public:\n"
                     "  A() = default;\n"
                     "  A(A&&) = default;\n"
                     "  A& operator=(A&&) = default;\n"
                     "private:\n"
                     "  B b;\n"
                     "  A(const A&) = delete;\n"
                     "  A& operator=(const A&) = delete;\n"
                     "};\n"
                     "void f() { A a; a = A(); A aa = A(); }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "class B {\n"
                         " public:\n"
                         "  B() = default;\n"
                         "  // Allow move only.\n"
                         "  B(B&&) = default;\n"
                         "  B& operator=(B&&) = default;\n"
                         "  B(const B&) = delete;\n"
                         "  B& operator=(const B&) = delete;\n"
                         " private:\n"
                         "  int ref_;\n"
                         "};\n"
                         "} // namespace na\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "class A {\n"
                         "public:\n"
                         "  A() = default;\n"
                         "  A(A&&) = default;\n"
                         "  A& operator=(A&&) = default;\n"
                         "private:\n"
                         "  na::B b;\n"
                         "  A(const A&) = delete;\n"
                         "  A& operator=(const A&) = delete;\n"
                         "};\n"
                         "void f() { A a; a = A(); A aa = A(); }\n"
                         "} // namespace y\n"
                         "} // namespace x\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, InjectedClassNameInFriendDecl) {
  OldNamespace = "d";
  NewNamespace = "e";
  std::string Code = "namespace a{\n"
                     "template <typename T>\n"
                     "class Base {\n"
                     " public:\n"
                     "  void f() {\n"
                     "    T t;\n"
                     "    t.priv();\n"
                     "  }\n"
                     "};\n"
                     "}  // namespace a\n"
                     "namespace d {\n"
                     "class D : public a::Base<D> {\n"
                     " private:\n"
                     "  friend class Base<D>;\n"
                     "  void priv() {}\n"
                     "  Base b;\n"
                     "};\n"
                     "\n"
                     "void f() {\n"
                     "  D d;\n"
                     "  a:: Base<D> b;\n"
                     "  b.f();\n"
                     "}\n"
                     "}  // namespace d\n";
  std::string Expected = "namespace a{\n"
                         "template <typename T>\n"
                         "class Base {\n"
                         " public:\n"
                         "  void f() {\n"
                         "    T t;\n"
                         "    t.priv();\n"
                         "  }\n"
                         "};\n"
                         "}  // namespace a\n"
                         "\n"
                         "namespace e {\n"
                         "class D : public a::Base<D> {\n"
                         " private:\n"
                         "  friend class a::Base<D>;\n"
                         "  void priv() {}\n"
                         "  a::Base b;\n"
                         "};\n"
                         "\n"
                         "void f() {\n"
                         "  D d;\n"
                         "  a::Base<D> b;\n"
                         "  b.f();\n"
                         "}\n"
                         "}  // namespace e\n";
  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

TEST_F(ChangeNamespaceTest, FullyQualifyConflictNamespace) {
  std::string Code =
      "namespace x { namespace util { class Some {}; } }\n"
      "namespace x { namespace y {namespace base { class Base {}; } } }\n"
      "namespace util { class Status {}; }\n"
      "namespace base { class Base {}; }\n"
      "namespace na {\n"
      "namespace nb {\n"
      "void f() {\n"
      "  util::Status s1; x::util::Some s2;\n"
      "  base::Base b1; x::y::base::Base b2;\n"
      "}\n"
      "} // namespace nb\n"
      "} // namespace na\n";

  std::string Expected =
      "namespace x { namespace util { class Some {}; } }\n"
      "namespace x { namespace y {namespace base { class Base {}; } } }\n"
      "namespace util { class Status {}; }\n"
      "namespace base { class Base {}; }\n"
      "\n"
      "namespace x {\n"
      "namespace y {\n"
      "void f() {\n"
      "  ::util::Status s1; util::Some s2;\n"
      "  ::base::Base b1; base::Base b2;\n"
      "}\n"
      "} // namespace y\n"
      "} // namespace x\n";

  EXPECT_EQ(format(Expected), runChangeNamespaceOnCode(Code));
}

} // anonymous namespace
} // namespace change_namespace
} // namespace clang
