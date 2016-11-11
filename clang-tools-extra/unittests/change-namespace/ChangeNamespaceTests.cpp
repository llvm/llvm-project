//===-- ChangeNamespaceTests.cpp - Change namespace unit tests ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ChangeNamespace.h"
#include "unittests/Tooling/RewriterTestContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
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
        OldNamespace, NewNamespace, FilePattern, &FileToReplacements);
    ast_matchers::MatchFinder Finder;
    NamespaceTool.registerMatchers(&Finder);
    std::unique_ptr<tooling::FrontendActionFactory> Factory =
        tooling::newFrontendActionFactory(&Finder);
    tooling::runToolOnCodeWithArgs(Factory->create(), Code, {"-std=c++11"},
                                   FileName);
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
                         "\n"
                         "} // namespace nb\n"
                         "} // namespace na\n";
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
                         "\n"
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
                     "class A {\n"
                     "  FWD *fwd;\n"
                     "};\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na {\n"
                         "namespace nb {\n"
                         "class FWD;\n"
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

TEST_F(ChangeNamespaceTest, DoNotCrashWithLambdaAsParameter) {
  std::string Code =
      "#include <functional>\n"
      "void f(std::function<void(int)> func, int param) { func(param); } "
      "void g() { f([](int x) {}, 1); }";

  std::string Expected =
      "#include <functional>\n"
      "void f(std::function<void(int)> func, int param) { func(param); } "
      "void g() { f([](int x) {}, 1); }";
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
                     "}\n"
                     "static int A::A1 = 0;\n"
                     "namespace nb {\n"
                     "void f() { int a = A::A1; int b = A::A2; }"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace na {\n"
                         "class A {\n"
                         "public:\n"
                         "static int A1;\n"
                         "static int A2;\n"
                         "}\n"
                         "static int A::A1 = 0;\n"
                         "\n"
                         "} // namespace na\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() { int a = na::A::A1; int b = na::A::A2; }"
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
                     "}\n"
                     "using glob::Glob;\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() { Glob g; }\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";

  std::string Expected = "namespace glob {\n"
                         "class Glob {};\n"
                         "}\n"
                         "using glob::Glob;\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() { Glob g; }\n"
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
  std::string Code = "namespace na { class C_A {};\n }\n"
                     "namespace na {\n"
                     "namespace nb {\n"
                     "void f() {\n"
                     "  using na::CA;\n"
                     "  CA ca;\n"
                     "}\n"
                     "} // namespace nb\n"
                     "} // namespace na\n";
  std::string Expected = "namespace na { class C_A {};\n }\n"
                         "\n"
                         "namespace x {\n"
                         "namespace y {\n"
                         "void f() {\n"
                         "  using na::CA;\n"
                         "  CA ca;\n"
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

TEST_F(ChangeNamespaceTest, UsingDeclInTheParentOfOldNamespace) {
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

} // anonymous namespace
} // namespace change_namespace
} // namespace clang
