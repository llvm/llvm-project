//===- unittest/Tooling/ExecutionTest.cpp - Tool execution tests. --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Execution.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/AllTUsExecution.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/StandaloneExecution.h"
#include "clang/Tooling/ToolExecutorPluginRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>

namespace clang {
namespace tooling {

namespace {

// This traverses the AST and outputs function name as key and "1" as value for
// each function declaration.
class ASTConsumerWithResult : public ASTConsumer,
                              public DynamicRecursiveASTVisitor {
public:
  explicit ASTConsumerWithResult(ExecutionContext *Context) : Context(Context) {
    assert(Context != nullptr);
  }

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    TraverseDecl(Context.getTranslationUnitDecl());
  }

  bool TraverseFunctionDecl(clang::FunctionDecl *Decl) override {
    Context->reportResult(Decl->getNameAsString(),
                          Context->getRevision() + ":" + Context->getCorpus() +
                              ":" + Context->getCurrentCompilationUnit() +
                              "/1");
    return DynamicRecursiveASTVisitor::TraverseFunctionDecl(Decl);
  }

private:
  ExecutionContext *const Context;
};

class ReportResultAction : public ASTFrontendAction {
public:
  explicit ReportResultAction(ExecutionContext *Context) : Context(Context) {
    assert(Context != nullptr);
  }

protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler,
                    StringRef /* dummy */) override {
    std::unique_ptr<clang::ASTConsumer> ast_consumer{
        new ASTConsumerWithResult(Context)};
    return ast_consumer;
  }

private:
  ExecutionContext *const Context;
};

class ReportResultActionFactory : public FrontendActionFactory {
public:
  ReportResultActionFactory(ExecutionContext *Context) : Context(Context) {}
  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<ReportResultAction>(Context);
  }

private:
  ExecutionContext *const Context;
};

} // namespace

class TestToolExecutor : public ToolExecutor {
public:
  static const char *ExecutorName;

  TestToolExecutor(CommonOptionsParser Options)
      : OptionsParser(std::move(Options)) {}

  StringRef getExecutorName() const override { return ExecutorName; }

  llvm::Error
  execute(llvm::ArrayRef<std::pair<std::unique_ptr<FrontendActionFactory>,
                                   ArgumentsAdjuster>>) override {
    return llvm::Error::success();
  }

  ExecutionContext *getExecutionContext() override { return nullptr; };

  ToolResults *getToolResults() override { return nullptr; }

  llvm::ArrayRef<std::string> getSourcePaths() const {
    return OptionsParser.getSourcePathList();
  }

  void mapVirtualFile(StringRef FilePath, StringRef Content) override {
    VFS[std::string(FilePath)] = std::string(Content);
  }

private:
  CommonOptionsParser OptionsParser;
  std::string SourcePaths;
  std::map<std::string, std::string> VFS;
};

const char *TestToolExecutor::ExecutorName = "test-executor";

class TestToolExecutorPlugin : public ToolExecutorPlugin {
public:
  llvm::Expected<std::unique_ptr<ToolExecutor>>
  create(CommonOptionsParser &OptionsParser) override {
    return std::make_unique<TestToolExecutor>(std::move(OptionsParser));
  }
};

static ToolExecutorPluginRegistry::Add<TestToolExecutorPlugin>
    X("test-executor", "Plugin for TestToolExecutor.");

llvm::cl::OptionCategory TestCategory("execution-test options");

TEST(CreateToolExecutorTest, FailedCreateExecutorUndefinedFlag) {
  std::vector<const char *> argv = {"prog", "--fake_flag_no_no_no", "f"};
  int argc = argv.size();
  auto Executor = internal::createExecutorFromCommandLineArgsImpl(
      argc, &argv[0], TestCategory);
  ASSERT_FALSE((bool)Executor);
  llvm::consumeError(Executor.takeError());
}

TEST(CreateToolExecutorTest, RegisterFlagsBeforeReset) {
  llvm::cl::opt<std::string> BeforeReset(
      "before_reset", llvm::cl::desc("Defined before reset."),
      llvm::cl::init(""));

  llvm::cl::ResetAllOptionOccurrences();

  std::vector<const char *> argv = {"prog", "--before_reset=set", "f"};
  int argc = argv.size();
  auto Executor = internal::createExecutorFromCommandLineArgsImpl(
      argc, &argv[0], TestCategory);
  ASSERT_TRUE((bool)Executor);
  EXPECT_EQ(BeforeReset, "set");
  BeforeReset.removeArgument();
}

TEST(CreateToolExecutorTest, CreateStandaloneToolExecutor) {
  std::vector<const char *> argv = {"prog", "standalone.cpp"};
  int argc = argv.size();
  auto Executor = internal::createExecutorFromCommandLineArgsImpl(
      argc, &argv[0], TestCategory);
  ASSERT_TRUE((bool)Executor);
  EXPECT_EQ(Executor->get()->getExecutorName(),
            StandaloneToolExecutor::ExecutorName);
}

TEST(CreateToolExecutorTest, CreateTestToolExecutor) {
  std::vector<const char *> argv = {"prog", "test.cpp",
                                    "--executor=test-executor"};
  int argc = argv.size();
  auto Executor = internal::createExecutorFromCommandLineArgsImpl(
      argc, &argv[0], TestCategory);
  ASSERT_TRUE((bool)Executor);
  EXPECT_EQ(Executor->get()->getExecutorName(), TestToolExecutor::ExecutorName);
}

TEST(StandaloneToolTest, SynctaxOnlyActionOnSimpleCode) {
  FixedCompilationDatabase Compilations(".", std::vector<std::string>());
  StandaloneToolExecutor Executor(Compilations,
                                  std::vector<std::string>(1, "a.cc"));
  Executor.mapVirtualFile("a.cc", "int x = 0;");

  auto Err = Executor.execute(newFrontendActionFactory<SyntaxOnlyAction>(),
                              getClangSyntaxOnlyAdjuster());
  ASSERT_TRUE(!Err);
}

TEST(StandaloneToolTest, SimpleAction) {
  FixedCompilationDatabase Compilations(".", std::vector<std::string>());
  StandaloneToolExecutor Executor(Compilations,
                                  std::vector<std::string>(1, "a.cc"));
  Executor.mapVirtualFile("a.cc", "int x = 0;");

  auto Err = Executor.execute(std::unique_ptr<FrontendActionFactory>(
      new ReportResultActionFactory(Executor.getExecutionContext())));
  ASSERT_TRUE(!Err);
  auto KVs = Executor.getToolResults()->AllKVResults();
  ASSERT_EQ(KVs.size(), 0u);
}

TEST(StandaloneToolTest, SimpleActionWithResult) {
  FixedCompilationDatabase Compilations(".", std::vector<std::string>());
  StandaloneToolExecutor Executor(Compilations,
                                  std::vector<std::string>(1, "a.cc"));
  Executor.mapVirtualFile("a.cc", "int x = 0; void f() {}");

  auto Err = Executor.execute(std::unique_ptr<FrontendActionFactory>(
      new ReportResultActionFactory(Executor.getExecutionContext())));
  ASSERT_TRUE(!Err);
  auto KVs = Executor.getToolResults()->AllKVResults();
  ASSERT_EQ(KVs.size(), 1u);
  EXPECT_EQ("f", KVs[0].first);
  // Currently the standlone executor returns empty corpus, revision, and
  // compilation unit.
  EXPECT_EQ("::/1", KVs[0].second);

  Executor.getToolResults()->forEachResult(
      [](StringRef, StringRef Value) { EXPECT_EQ("::/1", Value); });
}

class FixedCompilationDatabaseWithFiles : public CompilationDatabase {
public:
  FixedCompilationDatabaseWithFiles(Twine Directory,
                                    ArrayRef<std::string> Files,
                                    ArrayRef<std::string> CommandLine)
      : FixedCompilations(Directory, CommandLine), Files(Files) {}

  std::vector<CompileCommand>
  getCompileCommands(StringRef FilePath) const override {
    return FixedCompilations.getCompileCommands(FilePath);
  }

  std::vector<std::string> getAllFiles() const override { return Files; }

private:
  FixedCompilationDatabase FixedCompilations;
  std::vector<std::string> Files;
};

MATCHER_P(Named, Name, "") { return arg.first == Name; }

TEST(AllTUsToolTest, AFewFiles) {
  FixedCompilationDatabaseWithFiles Compilations(
      ".", {"a.cc", "b.cc", "c.cc", "ignore.cc"}, std::vector<std::string>());
  AllTUsToolExecutor Executor(Compilations, /*ThreadCount=*/0);
  Filter.setValue("[a-c].cc");
  Executor.mapVirtualFile("a.cc", "void x() {}");
  Executor.mapVirtualFile("b.cc", "void y() {}");
  Executor.mapVirtualFile("c.cc", "void z() {}");
  Executor.mapVirtualFile("ignore.cc", "void d() {}");

  auto Err = Executor.execute(std::unique_ptr<FrontendActionFactory>(
      new ReportResultActionFactory(Executor.getExecutionContext())));
  ASSERT_TRUE(!Err);
  EXPECT_THAT(
      Executor.getToolResults()->AllKVResults(),
      ::testing::UnorderedElementsAre(Named("x"), Named("y"), Named("z")));
  Filter.setValue(".*"); // reset to default value.
}

TEST(AllTUsToolTest, ManyFiles) {
  unsigned NumFiles = 100;
  std::vector<std::string> Files;
  std::map<std::string, std::string> FileToContent;
  std::vector<std::string> ExpectedSymbols;
  for (unsigned i = 1; i <= NumFiles; ++i) {
    std::string File = "f" + std::to_string(i) + ".cc";
    std::string Symbol = "looong_function_name_" + std::to_string(i);
    Files.push_back(File);
    FileToContent[File] = "void " + Symbol + "() {}";
    ExpectedSymbols.push_back(Symbol);
  }
  FixedCompilationDatabaseWithFiles Compilations(".", Files,
                                                 std::vector<std::string>());
  AllTUsToolExecutor Executor(Compilations, /*ThreadCount=*/0);
  for (const auto &FileAndContent : FileToContent) {
    Executor.mapVirtualFile(FileAndContent.first, FileAndContent.second);
  }

  auto Err = Executor.execute(std::unique_ptr<FrontendActionFactory>(
      new ReportResultActionFactory(Executor.getExecutionContext())));
  ASSERT_TRUE(!Err);
  std::vector<std::string> Results;
  Executor.getToolResults()->forEachResult(
      [&](StringRef Name, StringRef) { Results.push_back(std::string(Name)); });
  EXPECT_THAT(ExpectedSymbols, ::testing::UnorderedElementsAreArray(Results));
}

} // end namespace tooling
} // end namespace clang
