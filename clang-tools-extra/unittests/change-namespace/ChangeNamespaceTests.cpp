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

} // anonymous namespace
} // namespace change_namespace
} // namespace clang
