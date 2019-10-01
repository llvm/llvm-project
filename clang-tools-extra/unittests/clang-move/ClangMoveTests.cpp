//===-- ClangMoveTests.cpp - clang-move unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Move.h"
#include "unittests/Tooling/RewriterTestContext.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

namespace clang {
namespace move {
namespace {

const char TestHeader[] = "namespace a {\n"
                          "class C1; // test\n"
                          "template <typename T> class C2;\n"
                          "namespace b {\n"
                          "// This is a Foo class\n"
                          "// which is used in\n"
                          "// test.\n"
                          "class Foo {\n"
                          "public:\n"
                          "  void f();\n"
                          "\n"
                          "private:\n"
                          "  C1 *c1;\n"
                          "  static int b;\n"
                          "}; // abc\n"
                          "\n"
                          "class Foo2 {\n"
                          "public:\n"
                          "  int f();\n"
                          "};\n"
                          "} // namespace b\n"
                          "} // namespace a\n";

const char TestCC[] = "#include \"foo.h\"\n"
                      "namespace a {\n"
                      "namespace b {\n"
                      "namespace {\n"
                      "// comment1.\n"
                      "void f1() {}\n"
                      "/// comment2.\n"
                      "int kConstInt1 = 0;\n"
                      "} // namespace\n"
                      "\n"
                      "/* comment 3*/\n"
                      "static int kConstInt2 = 1;\n"
                      "\n"
                      "/** comment4\n"
                      " */\n"
                      "static int help() {\n"
                      "  int a = 0;\n"
                      "  return a;\n"
                      "}\n"
                      "\n"
                      "// comment5\n"
                      "// comment5\n"
                      "void Foo::f() {\n"
                      "  f1();\n"
                      "  kConstInt1;\n"
                      "  kConstInt2;\n"
                      "  help();\n"
                      "}\n"
                      "\n"
                      "/////////////\n"
                      "// comment //\n"
                      "/////////////\n"
                      "int Foo::b = 2;\n"
                      "int Foo2::f() {\n"
                      "  kConstInt1;\n"
                      "  kConstInt2;\n"
                      "  help();\n"
                      "  f1();\n"
                      "  return 1;\n"
                      "}\n"
                      "} // namespace b\n"
                      "} // namespace a\n";

const char ExpectedTestHeader[] = "namespace a {\n"
                                  "class C1; // test\n"
                                  "template <typename T> class C2;\n"
                                  "namespace b {\n"
                                  "\n"
                                  "class Foo2 {\n"
                                  "public:\n"
                                  "  int f();\n"
                                  "};\n"
                                  "} // namespace b\n"
                                  "} // namespace a\n";

const char ExpectedTestCC[] = "#include \"foo.h\"\n"
                              "namespace a {\n"
                              "namespace b {\n"
                              "namespace {\n"
                              "// comment1.\n"
                              "void f1() {}\n"
                              "/// comment2.\n"
                              "int kConstInt1 = 0;\n"
                              "} // namespace\n"
                              "\n"
                              "/* comment 3*/\n"
                              "static int kConstInt2 = 1;\n"
                              "\n"
                              "/** comment4\n"
                              " */\n"
                              "static int help() {\n"
                              "  int a = 0;\n"
                              "  return a;\n"
                              "}\n"
                              "\n"
                              "int Foo2::f() {\n"
                              "  kConstInt1;\n"
                              "  kConstInt2;\n"
                              "  help();\n"
                              "  f1();\n"
                              "  return 1;\n"
                              "}\n"
                              "} // namespace b\n"
                              "} // namespace a\n";

const char ExpectedNewHeader[] = "#ifndef NEW_FOO_H\n"
                                 "#define NEW_FOO_H\n"
                                 "\n"
                                 "namespace a {\n"
                                 "class C1; // test\n"
                                 "\n"
                                 "template <typename T> class C2;\n"
                                 "namespace b {\n"
                                 "// This is a Foo class\n"
                                 "// which is used in\n"
                                 "// test.\n"
                                 "class Foo {\n"
                                 "public:\n"
                                 "  void f();\n"
                                 "\n"
                                 "private:\n"
                                 "  C1 *c1;\n"
                                 "  static int b;\n"
                                 "}; // abc\n"
                                 "} // namespace b\n"
                                 "} // namespace a\n"
                                 "\n"
                                 "#endif // NEW_FOO_H\n";

const char ExpectedNewCC[] = "namespace a {\n"
                             "namespace b {\n"
                             "namespace {\n"
                             "// comment1.\n"
                             "void f1() {}\n"
                             "\n"
                             "/// comment2.\n"
                             "int kConstInt1 = 0;\n"
                             "} // namespace\n"
                             "\n"
                             "/* comment 3*/\n"
                             "static int kConstInt2 = 1;\n"
                             "\n"
                             "/** comment4\n"
                             " */\n"
                             "static int help() {\n"
                             "  int a = 0;\n"
                             "  return a;\n"
                             "}\n"
                             "\n"
                             "// comment5\n"
                             "// comment5\n"
                             "void Foo::f() {\n"
                             "  f1();\n"
                             "  kConstInt1;\n"
                             "  kConstInt2;\n"
                             "  help();\n"
                             "}\n"
                             "\n"
                             "/////////////\n"
                             "// comment //\n"
                             "/////////////\n"
                             "int Foo::b = 2;\n"
                             "} // namespace b\n"
                             "} // namespace a\n";

#ifdef _WIN32
const char WorkingDir[] = "C:\\test";
#else
const char WorkingDir[] = "/test";
#endif

const char TestHeaderName[] = "foo.h";
const char TestCCName[] = "foo.cc";

std::map<std::string, std::string>
runClangMoveOnCode(const move::MoveDefinitionSpec &Spec,
                   const char *const Header = TestHeader,
                   const char *const CC = TestCC,
                   DeclarationReporter *const Reporter = nullptr) {
  clang::RewriterTestContext Context;

  Context.InMemoryFileSystem->setCurrentWorkingDirectory(WorkingDir);

  std::map<llvm::StringRef, clang::FileID> FileToFileID;

  auto CreateFiles = [&Context, &FileToFileID](llvm::StringRef Name,
                                               llvm::StringRef Code) {
    if (!Name.empty()) {
      FileToFileID[Name] = Context.createInMemoryFile(Name, Code);
    }
  };
  CreateFiles(Spec.NewCC, "");
  CreateFiles(Spec.NewHeader, "");
  CreateFiles(TestHeaderName, Header);
  CreateFiles(TestCCName, CC);

  std::map<std::string, tooling::Replacements> FileToReplacements;
  ClangMoveContext MoveContext = {Spec, FileToReplacements, WorkingDir, "LLVM",
                                  Reporter != nullptr};

  auto Factory = llvm::make_unique<clang::move::ClangMoveActionFactory>(
      &MoveContext, Reporter);

 // std::string IncludeArg = Twine("-I" + WorkingDir;
  tooling::runToolOnCodeWithArgs(
      Factory->create(), CC, Context.InMemoryFileSystem,
      {"-std=c++11", "-fparse-all-comments", "-I."}, TestCCName, "clang-move",
      std::make_shared<PCHContainerOperations>());
  formatAndApplyAllReplacements(FileToReplacements, Context.Rewrite, "llvm");
  // The Key is file name, value is the new code after moving the class.
  std::map<std::string, std::string> Results;
  for (const auto &It : FileToReplacements) {
    // The path may come out as "./foo.h", normalize to "foo.h".
    SmallString<32> FilePath (It.first);
    llvm::sys::path::remove_dots(FilePath);
    Results[FilePath.str().str()] = Context.getRewrittenText(FileToFileID[FilePath]);
  }
  return Results;
}

TEST(ClangMove, MoveHeaderAndCC) {
  move::MoveDefinitionSpec Spec;
  Spec.Names = {std::string("a::b::Foo")};
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  std::string ExpectedHeader = "#include \"" + Spec.NewHeader + "\"\n\n";
  auto Results = runClangMoveOnCode(Spec);
  EXPECT_EQ(ExpectedTestHeader, Results[Spec.OldHeader]);
  EXPECT_EQ(ExpectedTestCC, Results[Spec.OldCC]);
  EXPECT_EQ(ExpectedNewHeader, Results[Spec.NewHeader]);
  EXPECT_EQ(ExpectedHeader + ExpectedNewCC, Results[Spec.NewCC]);
}

TEST(ClangMove, MoveHeaderOnly) {
  move::MoveDefinitionSpec Spec;
  Spec.Names = {std::string("a::b::Foo")};
  Spec.OldHeader = "foo.h";
  Spec.NewHeader = "new_foo.h";
  auto Results = runClangMoveOnCode(Spec);
  EXPECT_EQ(2u, Results.size());
  EXPECT_EQ(ExpectedTestHeader, Results[Spec.OldHeader]);
  EXPECT_EQ(ExpectedNewHeader, Results[Spec.NewHeader]);
}

TEST(ClangMove, MoveCCOnly) {
  move::MoveDefinitionSpec Spec;
  Spec.Names = {std::string("a::b::Foo")};
  Spec.OldCC = "foo.cc";
  Spec.NewCC = "new_foo.cc";
  std::string ExpectedHeader = "#include \"foo.h\"\n\n";
  auto Results = runClangMoveOnCode(Spec);
  EXPECT_EQ(2u, Results.size());
  EXPECT_EQ(ExpectedTestCC, Results[Spec.OldCC]);
  EXPECT_EQ(ExpectedHeader + ExpectedNewCC, Results[Spec.NewCC]);
}

TEST(ClangMove, MoveNonExistClass) {
  move::MoveDefinitionSpec Spec;
  Spec.Names = {std::string("NonExistFoo")};
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  auto Results = runClangMoveOnCode(Spec);
  EXPECT_EQ(0u, Results.size());
}

TEST(ClangMove, HeaderIncludeSelf) {
  move::MoveDefinitionSpec Spec;
  Spec.Names = {std::string("Foo")};
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";

  const char TestHeader[] = "#ifndef FOO_H\n"
                            "#define FOO_H\n"
                            "#include \"foo.h\"\n"
                            "class Foo {};\n"
                            "#endif\n";
  const char TestCode[] = "#include \"foo.h\"";
  const char ExpectedNewHeader[] = "#ifndef FOO_H\n"
                                   "#define FOO_H\n"
                                   "#include \"new_foo.h\"\n"
                                   "class Foo {};\n"
                                   "#endif\n";
  const char ExpectedNewCC[] = "#include \"new_foo.h\"";
  auto Results = runClangMoveOnCode(Spec, TestHeader, TestCode);
  EXPECT_EQ("", Results[Spec.OldHeader]);
  EXPECT_EQ(ExpectedNewHeader, Results[Spec.NewHeader]);
  EXPECT_EQ(ExpectedNewCC, Results[Spec.NewCC]);
}

TEST(ClangMove, MoveAll) {
  std::vector<std::string> TestHeaders = {
    "class A {\npublic:\n  int f();\n};",
    // forward declaration.
    "class B;\nclass A {\npublic:\n  int f();\n};",
    // template forward declaration.
    "template <typename T> class B;\nclass A {\npublic:\n  int f();\n};",
    "namespace a {}\nclass A {\npublic:\n  int f();\n};",
    "namespace a {}\nusing namespace a;\nclass A {\npublic:\n  int f();\n};",
  };
  const char Code[] = "#include \"foo.h\"\nint A::f() { return 0; }";
  move::MoveDefinitionSpec Spec;
  Spec.Names.push_back("A");
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  for (const auto& Header : TestHeaders) {
    auto Results = runClangMoveOnCode(Spec, Header.c_str(), Code);
    EXPECT_EQ(Header, Results[Spec.NewHeader]);
    EXPECT_EQ("", Results[Spec.OldHeader]);
    EXPECT_EQ("", Results[Spec.OldCC]);
  }
}

TEST(ClangMove, MoveAllMultipleClasses) {
  move::MoveDefinitionSpec Spec;
  std::vector<std::string> TestHeaders = {
    "class C;\nclass A {\npublic:\n  int f();\n};\nclass B {};",
    "class C;\nclass B;\nclass A {\npublic:\n  int f();\n};\nclass B {};",
  };
  const char Code[] = "#include \"foo.h\"\nint A::f() { return 0; }";
  Spec.Names = {std::string("A"), std::string("B")};
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  for (const auto& Header : TestHeaders) {
    auto Results = runClangMoveOnCode(Spec, Header.c_str(), Code);
    EXPECT_EQ(Header, Results[Spec.NewHeader]);
    EXPECT_EQ("", Results[Spec.OldHeader]);
    EXPECT_EQ("", Results[Spec.OldCC]);
  }
}

TEST(ClangMove, DontMoveAll) {
  const char ExpectedHeader[] = "#ifndef NEW_FOO_H\n"
                                "#define NEW_FOO_H\n"
                                "\n"
                                "class A {\npublic:\n  int f();\n};\n"
                                "\n"
                                "#endif // NEW_FOO_H\n";
  const char Code[] = "#include \"foo.h\"\nint A::f() { return 0; }";
  std::vector<std::string> TestHeaders = {
    "class B {};\nclass A {\npublic:\n  int f();\n};\n",
    "void f() {};\nclass A {\npublic:\n  int f();\n};\n",
  };
  move::MoveDefinitionSpec Spec;
  Spec.Names.push_back("A");
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  for (const auto& Header : TestHeaders) {
    auto Results = runClangMoveOnCode(Spec, Header.c_str(), Code);
    EXPECT_EQ(ExpectedHeader, Results[Spec.NewHeader]);
    // The expected old header should not contain class A definition.
    std::string ExpectedOldHeader = Header.substr(0, Header.size() - 32);
    EXPECT_EQ(ExpectedOldHeader, Results[Spec.OldHeader]);
  }
}

TEST(ClangMove, IgnoreMacroSymbolsAndMoveAll) {
  const char TestCode[] = "#include \"foo.h\"";
  std::vector<std::string> TestHeaders = {
    "#define DEFINE_Foo int Foo = 1;\nDEFINE_Foo;\nclass Bar {};\n",
    "#define DEFINE(x) int var_##x = 1;\nDEFINE(foo);\nclass Bar {};\n",
  };
  move::MoveDefinitionSpec Spec;
  Spec.Names.push_back("Bar");
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";

  for (const auto& Header : TestHeaders) {
    auto Results = runClangMoveOnCode(Spec, Header.c_str(), TestCode);
    EXPECT_EQ("", Results[Spec.OldHeader]);
    EXPECT_EQ(Header, Results[Spec.NewHeader]);
  }
}

TEST(ClangMove, MacroInFunction) {
  const char TestHeader[] = "#define INT int\n"
                            "class A {\npublic:\n  int f();\n};\n"
                            "class B {};\n";
  const char TestCode[] = "#include \"foo.h\"\n"
                          "INT A::f() { return 0; }\n";
  const char ExpectedNewCode[] = "#include \"new_foo.h\"\n\n"
                                 "INT A::f() { return 0; }\n";
  move::MoveDefinitionSpec Spec;
  Spec.Names.push_back("A");
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  auto Results = runClangMoveOnCode(Spec, TestHeader, TestCode);
  EXPECT_EQ(ExpectedNewCode, Results[Spec.NewCC]);
}

TEST(ClangMove, DefinitionInMacro) {
  const char TestHeader[] = "#define DEF(CLASS) void CLASS##_::f() {}\n"
                            "#define DEF2(CLASS, ...) void CLASS##_::f2() {}\n"
                            "class A_ {\nvoid f();\nvoid f2();\n};\n"
                            "class B {};\n";
  const char TestCode[] = "#include \"foo.h\"\n"
                          "DEF(A)\n\n"
                          "DEF2(A,\n"
                          "     123)\n";
  const char ExpectedNewCode[] = "#include \"new_foo.h\"\n\n"
                                 "DEF(A)\n\n"
                                 "DEF2(A, 123)\n";
  move::MoveDefinitionSpec Spec;
  Spec.Names.push_back("A_");
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  auto Results = runClangMoveOnCode(Spec, TestHeader, TestCode);
  EXPECT_EQ(ExpectedNewCode, Results[Spec.NewCC]);
}

TEST(ClangMove, WellFormattedCode) {
  const std::string CommonHeader =
      "namespace a {\n"
      "namespace b {\n"
      "namespace c {\n"
      "class C;\n"
      "\n"
      "class A {\npublic:\n  void f();\n  void f2();\n};\n"
      "} // namespace c\n"
      "} // namespace b\n"
      "\n"
      "namespace d {\n"
      "namespace e {\n"
      "class B {\npublic:\n  void f();\n};\n"
      "} // namespace e\n"
      "} // namespace d\n"
      "} // namespace a\n";
  const std::string CommonCode = "\n"
                                 "namespace a {\n"
                                 "namespace b {\n"
                                 "namespace c {\n"
                                 "void A::f() {}\n"
                                 "\n"
                                 "void A::f2() {}\n"
                                 "} // namespace c\n"
                                 "} // namespace b\n"
                                 "\n"
                                 "namespace d {\n"
                                 "namespace e {\n"
                                 "void B::f() {}\n"
                                 "} // namespace e\n"
                                 "} // namespace d\n"
                                 "} // namespace a\n";
  // Add dummy class to prevent behavior of moving all declarations from header.
  const std::string TestHeader = CommonHeader + "class D {};\n";
  const std::string TestCode = "#include \"foo.h\"\n" + CommonCode;
  const std::string ExpectedNewHeader = "#ifndef NEW_FOO_H\n"
                                        "#define NEW_FOO_H\n"
                                        "\n" +
                                        CommonHeader +
                                        "\n"
                                        "#endif // NEW_FOO_H\n";
  const std::string ExpectedNewCC = "#include \"new_foo.h\"\n" + CommonCode;
  move::MoveDefinitionSpec Spec;
  Spec.Names.push_back("a::b::c::A");
  Spec.Names.push_back("a::d::e::B");
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  auto Results = runClangMoveOnCode(Spec, TestHeader.c_str(), TestCode.c_str());
  EXPECT_EQ(ExpectedNewCC, Results[Spec.NewCC]);
  EXPECT_EQ(ExpectedNewHeader, Results[Spec.NewHeader]);
}

TEST(ClangMove, AddDependentNewHeader) {
  const char TestHeader[] = "class A {};\n"
                            "class B {};\n";
  const char TestCode[] = "#include \"foo.h\"\n";
  const char ExpectedOldHeader[] = "#include \"new_foo.h\"\nclass B {};\n";
  const char ExpectedNewHeader[] = "#ifndef NEW_FOO_H\n"
                                   "#define NEW_FOO_H\n"
                                   "\n"
                                   "class A {};\n"
                                   "\n"
                                   "#endif // NEW_FOO_H\n";
  move::MoveDefinitionSpec Spec;
  Spec.Names.push_back("A");
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  Spec.OldDependOnNew = true;
  auto Results = runClangMoveOnCode(Spec, TestHeader, TestCode);
  EXPECT_EQ(ExpectedOldHeader, Results[Spec.OldHeader]);
  EXPECT_EQ(ExpectedNewHeader, Results[Spec.NewHeader]);
}

TEST(ClangMove, AddDependentOldHeader) {
  const char TestHeader[] = "class A {};\n"
                            "class B {};\n";
  const char TestCode[] = "#include \"foo.h\"\n";
  const char ExpectedNewHeader[] = "#ifndef NEW_FOO_H\n"
                                   "#define NEW_FOO_H\n"
                                   "\n"
                                   "#include \"foo.h\"\n"
                                   "\n"
                                   "class B {};\n"
                                   "\n"
                                   "#endif // NEW_FOO_H\n";
  const char ExpectedOldHeader[] = "class A {};\n";
  move::MoveDefinitionSpec Spec;
  Spec.Names.push_back("B");
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  Spec.NewDependOnOld = true;
  auto Results = runClangMoveOnCode(Spec, TestHeader, TestCode);
  EXPECT_EQ(ExpectedNewHeader, Results[Spec.NewHeader]);
  EXPECT_EQ(ExpectedOldHeader, Results[Spec.OldHeader]);
}

TEST(ClangMove, DumpDecls) {
  const char TestHeader[] = "template <typename T>\n"
                            "class A {\n"
                            " public:\n"
                            "  void f();\n"
                            "  template <typename U> void h();\n"
                            "  static int b;\n"
                            "};\n"
                            "\n"
                            "template <typename T> void A<T>::f() {}\n"
                            "\n"
                            "template <typename T>\n"
                            "template <typename U>\n"
                            "void A<T>::h() {}\n"
                            "\n"
                            "template <typename T> int A<T>::b = 2;\n"
                            "\n"
                            "template <> class A<int> {};\n"
                            "\n"
                            "class B {};\n"
                            "\n"
                            "namespace a {\n"
                            "class Move1 {};\n"
                            "void f1() {}\n"
                            "template <typename T>"
                            "void f2(T t);\n"
                            "} // namespace a\n"
                            "\n"
                            "class ForwardClass;\n"
                            "namespace a {\n"
                            "namespace b {\n"
                            "class Move1 { public : void f(); };\n"
                            "void f() {}\n"
                            "enum E1 { Green };\n"
                            "enum class E2 { Red };\n"
                            "typedef int Int2;\n"
                            "typedef A<double> A_d;"
                            "using Int = int;\n"
                            "template <typename T>\n"
                            "using AA = A<T>;\n"
                            "extern int kGlobalInt;\n"
                            "extern const char* const kGlobalStr;\n"
                            "} // namespace b\n"
                            "} // namespace a\n"
                            "#define DEFINE_FOO class Foo {};\n"
                            "DEFINE_FOO\n";
  const char TestCode[] = "#include \"foo.h\"\n";
  move::MoveDefinitionSpec Spec;
  Spec.Names.push_back("B");
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  DeclarationReporter Reporter;
  std::vector<DeclarationReporter::Declaration> ExpectedDeclarations = {
      {"A", "Class", true},
      {"B", "Class", false},
      {"a::Move1", "Class", false},
      {"a::f1", "Function", false},
      {"a::f2", "Function", true},
      {"a::b::Move1", "Class", false},
      {"a::b::f", "Function", false},
      {"a::b::E1", "Enum", false},
      {"a::b::E2", "Enum", false},
      {"a::b::Int2", "TypeAlias", false},
      {"a::b::A_d", "TypeAlias", false},
      {"a::b::Int", "TypeAlias", false},
      {"a::b::AA", "TypeAlias", true},
      {"a::b::kGlobalInt", "Variable", false},
      {"a::b::kGlobalStr", "Variable", false}};
  runClangMoveOnCode(Spec, TestHeader, TestCode, &Reporter);
  std::vector<DeclarationReporter::Declaration> Results;
  for (const auto &DelPair : Reporter.getDeclarationList())
    Results.push_back(DelPair);
  EXPECT_THAT(ExpectedDeclarations,
              testing::UnorderedElementsAreArray(Results));
}

} // namespace
} // namespce move
} // namespace clang
