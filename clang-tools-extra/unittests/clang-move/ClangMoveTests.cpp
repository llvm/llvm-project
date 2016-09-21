//===-- ClangMoveTest.cpp - clang-move unit tests -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangMove.h"
#include "unittests/Tooling/RewriterTestContext.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

namespace clang {
namespace move {
namespace {

const char TestHeaderName[] = "foo.h";

const char TestCCName[] = "foo.cc";

const char TestHeader[] = "namespace a {\n"
                          "class C1;\n"
                          "namespace b {\n"
                          "class Foo {\n"
                          "public:\n"
                          "  void f();\n"
                          "\n"
                          "private:\n"
                          "  C1 *c1;\n"
                          "  static int b;\n"
                          "};\n"
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
                      "void f1() {}\n"
                      "int kConstInt1 = 0;\n"
                      "} // namespace\n"
                      "\n"
                      "static int kConstInt2 = 1;\n"
                      "\n"
                      "static int help() {\n"
                      "  int a = 0;\n"
                      "  return a;\n"
                      "}\n"
                      "\n"
                      "void Foo::f() { f1(); }\n"
                      "\n"
                      "int Foo::b = 2;\n"
                      "int Foo2::f() {\n"
                      "  f1();\n"
                      "  return 1;\n"
                      "}\n"
                      "} // namespace b\n"
                      "} // namespace a\n";

const char ExpectedTestHeader[] = "namespace a {\n"
                                  "class C1;\n"
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
                              "void f1() {}\n"
                              "int kConstInt1 = 0;\n"
                              "} // namespace\n"
                              "\n"
                              "static int kConstInt2 = 1;\n"
                              "\n"
                              "static int help() {\n"
                              "  int a = 0;\n"
                              "  return a;\n"
                              "}\n"
                              "\n"
                              "int Foo2::f() {\n"
                              "  f1();\n"
                              "  return 1;\n"
                              "}\n"
                              "} // namespace b\n"
                              "} // namespace a\n";

const char ExpectedNewHeader[] = "namespace a {\n"
                                 "class C1;\n"
                                 "namespace b {\n"
                                 "class Foo {\n"
                                 "public:\n"
                                 "  void f();\n"
                                 "\n"
                                 "private:\n"
                                 "  C1 *c1;\n"
                                 "  static int b;\n"
                                 "};\n"
                                 "} // namespace b\n"
                                 "} // namespace a\n";

const char ExpectedNewCC[] = "#include \"foo.h\"\n"
                             "namespace a {\n"
                             "namespace b {\n"
                             "namespace {\n"
                             "void f1() {}\n"
                             "int kConstInt1 = 0;\n"
                             "} // namespace\n"
                             "static int kConstInt2 = 1;\n"
                             "static int help() {\n"
                             "  int a = 0;\n"
                             "  return a;\n"
                             "}\n"
                             "void Foo::f() { f1(); }\n"
                             "int Foo::b = 2;\n"
                             "} // namespace b\n"
                             "} // namespace a\n";

std::map<std::string, std::string>
runClangMoveOnCode(const move::ClangMoveTool::MoveDefinitionSpec &Spec) {
  clang::RewriterTestContext Context;

  std::map<llvm::StringRef, clang::FileID> FileToFileID;
  std::vector<std::pair<std::string, std::string>> FileToSourceText = {
      {TestHeaderName, TestHeader}, {TestCCName, TestCC}};

  auto CreateFiles = [&FileToSourceText, &Context, &FileToFileID](
      llvm::StringRef Name, llvm::StringRef Code) {
    if (!Name.empty()) {
      FileToSourceText.emplace_back(Name, Code);
      FileToFileID[Name] = Context.createInMemoryFile(Name, Code);
    }
  };
  CreateFiles(Spec.NewCC, "");
  CreateFiles(Spec.NewHeader, "");
  CreateFiles(Spec.OldHeader, TestHeader);
  CreateFiles(Spec.OldCC, TestCC);

  std::map<std::string, tooling::Replacements> FileToReplacements;
  ClangMoveTool MoveTool(Spec, FileToReplacements);
  auto Factory = llvm::make_unique<clang::move::ClangMoveActionFactory>(
      Spec, FileToReplacements);

  tooling::runToolOnCodeWithArgs(
      Factory->create(), TestCC, {"-std=c++11"}, TestCCName, "clang-move",
      std::make_shared<PCHContainerOperations>(), FileToSourceText);
  formatAndApplyAllReplacements(FileToReplacements, Context.Rewrite, "llvm");
  // The Key is file name, value is the new code after moving the class.
  std::map<std::string, std::string> Results;
  for (const auto &It : FileToReplacements) {
    StringRef FilePath = It.first;
    Results[FilePath] = Context.getRewrittenText(FileToFileID[FilePath]);
  }
  return Results;
}

TEST(ClangMove, MoveHeaderAndCC) {
  move::ClangMoveTool::MoveDefinitionSpec Spec;
  Spec.Name = "a::b::Foo";
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  auto Results = runClangMoveOnCode(Spec);
  EXPECT_EQ(ExpectedTestHeader, Results[Spec.OldHeader]);
  EXPECT_EQ(ExpectedTestCC, Results[Spec.OldCC]);
  EXPECT_EQ(ExpectedNewHeader, Results[Spec.NewHeader]);
  EXPECT_EQ(ExpectedNewCC, Results[Spec.NewCC]);
}

TEST(ClangMove, MoveHeaderOnly) {
  move::ClangMoveTool::MoveDefinitionSpec Spec;
  Spec.Name = "a::b::Foo";
  Spec.OldHeader = "foo.h";
  Spec.NewHeader = "new_foo.h";
  auto Results = runClangMoveOnCode(Spec);
  EXPECT_EQ(2u, Results.size());
  EXPECT_EQ(ExpectedTestHeader, Results[Spec.OldHeader]);
  EXPECT_EQ(ExpectedNewHeader, Results[Spec.NewHeader]);
}

TEST(ClangMove, MoveCCOnly) {
  move::ClangMoveTool::MoveDefinitionSpec Spec;
  Spec.Name = "a::b::Foo";
  Spec.OldCC = "foo.cc";
  Spec.NewCC = "new_foo.cc";
  auto Results = runClangMoveOnCode(Spec);
  EXPECT_EQ(2u, Results.size());
  EXPECT_EQ(ExpectedTestCC, Results[Spec.OldCC]);
  EXPECT_EQ(ExpectedNewCC, Results[Spec.NewCC]);
}

TEST(ClangMove, MoveNonExistClass) {
  move::ClangMoveTool::MoveDefinitionSpec Spec;
  Spec.Name = "NonExistFoo";
  Spec.OldHeader = "foo.h";
  Spec.OldCC = "foo.cc";
  Spec.NewHeader = "new_foo.h";
  Spec.NewCC = "new_foo.cc";
  auto Results = runClangMoveOnCode(Spec);
  EXPECT_EQ(0u, Results.size());
}

} // namespace
} // namespce move
} // namespace clang
