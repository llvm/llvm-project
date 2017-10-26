//===- unittest/Tooling/ExecutionTest.cpp - Tool execution tests. --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/StandaloneExecution.h"
#include "clang/Tooling/ToolExecutorPluginRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>

namespace clang {
namespace tooling {

namespace {

// This traverses the AST and outputs function name as key and "1" as value for
// each function declaration.
class ASTConsumerWithResult
    : public ASTConsumer,
      public RecursiveASTVisitor<ASTConsumerWithResult> {
public:
  using ASTVisitor = RecursiveASTVisitor<ASTConsumerWithResult>;

  explicit ASTConsumerWithResult(ExecutionContext *Context) : Context(Context) {
    assert(Context != nullptr);
  }

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    TraverseDecl(Context.getTranslationUnitDecl());
  }

  bool TraverseFunctionDecl(clang::FunctionDecl *Decl) {
    Context->reportResult(Decl->getNameAsString(), "1");
    return ASTVisitor::TraverseFunctionDecl(Decl);
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
  FrontendAction *create() override { return new ReportResultAction(Context); }

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
    VFS[FilePath] = Content;
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
    return llvm::make_unique<TestToolExecutor>(std::move(OptionsParser));
  }
};

// This anchor is used to force the linker to link in the generated object file
// and thus register the plugin.
extern volatile int ToolExecutorPluginAnchorSource;

static int LLVM_ATTRIBUTE_UNUSED TestToolExecutorPluginAnchorDest =
    ToolExecutorPluginAnchorSource;

static ToolExecutorPluginRegistry::Add<TestToolExecutorPlugin>
    X("test-executor", "Plugin for TestToolExecutor.");

llvm::cl::OptionCategory TestCategory("execution-test options");

TEST(CreateToolExecutorTest, FailedCreateExecutorUndefinedFlag) {
  std::vector<const char *> argv = {"prog", "--fake_flag_no_no_no", "f"};
  int argc = argv.size();
  auto Executor =
      createExecutorFromCommandLineArgs(argc, &argv[0], TestCategory);
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
  auto Executor =
      createExecutorFromCommandLineArgs(argc, &argv[0], TestCategory);
  ASSERT_TRUE((bool)Executor);
  EXPECT_EQ(BeforeReset, "set");
  BeforeReset.removeArgument();
}

TEST(CreateToolExecutorTest, CreateStandaloneToolExecutor) {
  std::vector<const char *> argv = {"prog", "standalone.cpp"};
  int argc = argv.size();
  auto Executor =
      createExecutorFromCommandLineArgs(argc, &argv[0], TestCategory);
  ASSERT_TRUE((bool)Executor);
  EXPECT_EQ(Executor->get()->getExecutorName(),
            StandaloneToolExecutor::ExecutorName);
}

TEST(CreateToolExecutorTest, CreateTestToolExecutor) {
  std::vector<const char *> argv = {"prog", "test.cpp",
                                    "--executor=test-executor"};
  int argc = argv.size();
  auto Executor =
      createExecutorFromCommandLineArgs(argc, &argv[0], TestCategory);
  ASSERT_TRUE((bool)Executor);
  EXPECT_EQ(Executor->get()->getExecutorName(), TestToolExecutor::ExecutorName);
}

TEST(StandaloneToolTest, SynctaxOnlyActionOnSimpleCode) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  StandaloneToolExecutor Executor(Compilations,
                                  std::vector<std::string>(1, "/a.cc"));
  Executor.mapVirtualFile("/a.cc", "int x = 0;");

  auto Err = Executor.execute(newFrontendActionFactory<SyntaxOnlyAction>(),
                              getClangSyntaxOnlyAdjuster());
  ASSERT_TRUE(!Err);
}

TEST(StandaloneToolTest, SimpleAction) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  StandaloneToolExecutor Executor(Compilations,
                                  std::vector<std::string>(1, "/a.cc"));
  Executor.mapVirtualFile("/a.cc", "int x = 0;");

  auto Err = Executor.execute(std::unique_ptr<FrontendActionFactory>(
      new ReportResultActionFactory(Executor.getExecutionContext())));
  ASSERT_TRUE(!Err);
  auto KVs = Executor.getToolResults()->AllKVResults();
  ASSERT_EQ(KVs.size(), 0u);
}

TEST(StandaloneToolTest, SimpleActionWithResult) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  StandaloneToolExecutor Executor(Compilations,
                                  std::vector<std::string>(1, "/a.cc"));
  Executor.mapVirtualFile("/a.cc", "int x = 0; void f() {}");

  auto Err = Executor.execute(std::unique_ptr<FrontendActionFactory>(
      new ReportResultActionFactory(Executor.getExecutionContext())));
  ASSERT_TRUE(!Err);
  auto KVs = Executor.getToolResults()->AllKVResults();
  ASSERT_EQ(KVs.size(), 1u);
  EXPECT_EQ("f", KVs[0].first);
  EXPECT_EQ("1", KVs[0].second);

  Executor.getToolResults()->forEachResult(
      [](StringRef, StringRef Value) { EXPECT_EQ("1", Value); });
}

} // end namespace tooling
} // end namespace clang
