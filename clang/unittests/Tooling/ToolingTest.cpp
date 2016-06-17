//===- unittest/Tooling/ToolingTest.cpp - Tooling unit tests --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>

namespace clang {
namespace tooling {

namespace {
/// Takes an ast consumer and returns it from CreateASTConsumer. This only
/// works with single translation unit compilations.
class TestAction : public clang::ASTFrontendAction {
public:
  /// Takes ownership of TestConsumer.
  explicit TestAction(std::unique_ptr<clang::ASTConsumer> TestConsumer)
      : TestConsumer(std::move(TestConsumer)) {}

protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler,
                    StringRef dummy) override {
    /// TestConsumer will be deleted by the framework calling us.
    return std::move(TestConsumer);
  }

private:
  std::unique_ptr<clang::ASTConsumer> TestConsumer;
};

class FindTopLevelDeclConsumer : public clang::ASTConsumer {
 public:
  explicit FindTopLevelDeclConsumer(bool *FoundTopLevelDecl)
      : FoundTopLevelDecl(FoundTopLevelDecl) {}
  bool HandleTopLevelDecl(clang::DeclGroupRef DeclGroup) override {
    *FoundTopLevelDecl = true;
    return true;
  }
 private:
  bool * const FoundTopLevelDecl;
};
} // end namespace

TEST(runToolOnCode, FindsNoTopLevelDeclOnEmptyCode) {
  bool FoundTopLevelDecl = false;
  EXPECT_TRUE(
      runToolOnCode(new TestAction(llvm::make_unique<FindTopLevelDeclConsumer>(
                        &FoundTopLevelDecl)),
                    ""));
  EXPECT_FALSE(FoundTopLevelDecl);
}

namespace {
class FindClassDeclXConsumer : public clang::ASTConsumer {
 public:
  FindClassDeclXConsumer(bool *FoundClassDeclX)
      : FoundClassDeclX(FoundClassDeclX) {}
  bool HandleTopLevelDecl(clang::DeclGroupRef GroupRef) override {
    if (CXXRecordDecl* Record = dyn_cast<clang::CXXRecordDecl>(
            *GroupRef.begin())) {
      if (Record->getName() == "X") {
        *FoundClassDeclX = true;
      }
    }
    return true;
  }
 private:
  bool *FoundClassDeclX;
};
bool FindClassDeclX(ASTUnit *AST) {
  for (std::vector<Decl *>::iterator i = AST->top_level_begin(),
                                     e = AST->top_level_end();
       i != e; ++i) {
    if (CXXRecordDecl* Record = dyn_cast<clang::CXXRecordDecl>(*i)) {
      if (Record->getName() == "X") {
        return true;
      }
    }
  }
  return false;
}
} // end namespace

TEST(runToolOnCode, FindsClassDecl) {
  bool FoundClassDeclX = false;
  EXPECT_TRUE(
      runToolOnCode(new TestAction(llvm::make_unique<FindClassDeclXConsumer>(
                        &FoundClassDeclX)),
                    "class X;"));
  EXPECT_TRUE(FoundClassDeclX);

  FoundClassDeclX = false;
  EXPECT_TRUE(
      runToolOnCode(new TestAction(llvm::make_unique<FindClassDeclXConsumer>(
                        &FoundClassDeclX)),
                    "class Y;"));
  EXPECT_FALSE(FoundClassDeclX);
}

TEST(buildASTFromCode, FindsClassDecl) {
  std::unique_ptr<ASTUnit> AST = buildASTFromCode("class X;");
  ASSERT_TRUE(AST.get());
  EXPECT_TRUE(FindClassDeclX(AST.get()));

  AST = buildASTFromCode("class Y;");
  ASSERT_TRUE(AST.get());
  EXPECT_FALSE(FindClassDeclX(AST.get()));
}

TEST(newFrontendActionFactory, CreatesFrontendActionFactoryFromType) {
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory<SyntaxOnlyAction>());
  std::unique_ptr<FrontendAction> Action(Factory->create());
  EXPECT_TRUE(Action.get() != nullptr);
}

struct IndependentFrontendActionCreator {
  std::unique_ptr<ASTConsumer> newASTConsumer() {
    return llvm::make_unique<FindTopLevelDeclConsumer>(nullptr);
  }
};

TEST(newFrontendActionFactory, CreatesFrontendActionFactoryFromFactoryType) {
  IndependentFrontendActionCreator Creator;
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Creator));
  std::unique_ptr<FrontendAction> Action(Factory->create());
  EXPECT_TRUE(Action.get() != nullptr);
}

TEST(ToolInvocation, TestMapVirtualFile) {
  llvm::IntrusiveRefCntPtr<vfs::OverlayFileSystem> OverlayFileSystem(
      new vfs::OverlayFileSystem(vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
      new vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions(), OverlayFileSystem));
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  Args.push_back("-Idef");
  Args.push_back("-fsyntax-only");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(Args, new SyntaxOnlyAction,
                                            Files.get());
  InMemoryFileSystem->addFile(
      "test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("#include <abc>\n"));
  InMemoryFileSystem->addFile("def/abc", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  EXPECT_TRUE(Invocation.run());
}

TEST(ToolInvocation, TestVirtualModulesCompilation) {
  // FIXME: Currently, this only tests that we don't exit with an error if a
  // mapped module.map is found on the include path. In the future, expand this
  // test to run a full modules enabled compilation, so we make sure we can
  // rerun modules compilations with a virtual file system.
  llvm::IntrusiveRefCntPtr<vfs::OverlayFileSystem> OverlayFileSystem(
      new vfs::OverlayFileSystem(vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
      new vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions(), OverlayFileSystem));
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  Args.push_back("-Idef");
  Args.push_back("-fsyntax-only");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(Args, new SyntaxOnlyAction,
                                            Files.get());
  InMemoryFileSystem->addFile(
      "test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("#include <abc>\n"));
  InMemoryFileSystem->addFile("def/abc", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  // Add a module.map file in the include directory of our header, so we trigger
  // the module.map header search logic.
  InMemoryFileSystem->addFile("def/module.map", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  EXPECT_TRUE(Invocation.run());
}

struct VerifyEndCallback : public SourceFileCallbacks {
  VerifyEndCallback() : BeginCalled(0), EndCalled(0), Matched(false) {}
  bool handleBeginSource(CompilerInstance &CI, StringRef Filename) override {
    ++BeginCalled;
    return true;
  }
  void handleEndSource() override { ++EndCalled; }
  std::unique_ptr<ASTConsumer> newASTConsumer() {
    return llvm::make_unique<FindTopLevelDeclConsumer>(&Matched);
  }
  unsigned BeginCalled;
  unsigned EndCalled;
  bool Matched;
};

#if !defined(LLVM_ON_WIN32)
TEST(newFrontendActionFactory, InjectsSourceFileCallbacks) {
  VerifyEndCallback EndCallback;

  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  std::vector<std::string> Sources;
  Sources.push_back("/a.cc");
  Sources.push_back("/b.cc");
  ClangTool Tool(Compilations, Sources);

  Tool.mapVirtualFile("/a.cc", "void a() {}");
  Tool.mapVirtualFile("/b.cc", "void b() {}");

  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory(&EndCallback, &EndCallback));
  Tool.run(Action.get());

  EXPECT_TRUE(EndCallback.Matched);
  EXPECT_EQ(2u, EndCallback.BeginCalled);
  EXPECT_EQ(2u, EndCallback.EndCalled);
}
#endif

struct SkipBodyConsumer : public clang::ASTConsumer {
  /// Skip the 'skipMe' function.
  bool shouldSkipFunctionBody(Decl *D) override {
    NamedDecl *F = dyn_cast<NamedDecl>(D);
    return F && F->getNameAsString() == "skipMe";
  }
};

struct SkipBodyAction : public clang::ASTFrontendAction {
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef) override {
    Compiler.getFrontendOpts().SkipFunctionBodies = true;
    return llvm::make_unique<SkipBodyConsumer>();
  }
};

TEST(runToolOnCode, TestSkipFunctionBody) {
  std::vector<std::string> Args = {"-std=c++11"};
  std::vector<std::string> Args2 = {"-fno-delayed-template-parsing"};

  EXPECT_TRUE(runToolOnCode(new SkipBodyAction,
                            "int skipMe() { an_error_here }"));
  EXPECT_FALSE(runToolOnCode(new SkipBodyAction,
                             "int skipMeNot() { an_error_here }"));

  // Test constructors with initializers
  EXPECT_TRUE(runToolOnCodeWithArgs(
      new SkipBodyAction,
      "struct skipMe { skipMe() : an_error() { more error } };", Args));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      new SkipBodyAction, "struct skipMe { skipMe(); };"
                          "skipMe::skipMe() : an_error([](){;}) { more error }",
      Args));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      new SkipBodyAction, "struct skipMe { skipMe(); };"
                          "skipMe::skipMe() : an_error{[](){;}} { more error }",
      Args));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      new SkipBodyAction,
      "struct skipMe { skipMe(); };"
      "skipMe::skipMe() : a<b<c>(e)>>(), f{}, g() { error }",
      Args));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      new SkipBodyAction, "struct skipMe { skipMe() : bases()... { error } };",
      Args));

  EXPECT_FALSE(runToolOnCodeWithArgs(
      new SkipBodyAction, "struct skipMeNot { skipMeNot() : an_error() { } };",
      Args));
  EXPECT_FALSE(runToolOnCodeWithArgs(new SkipBodyAction,
                                     "struct skipMeNot { skipMeNot(); };"
                                     "skipMeNot::skipMeNot() : an_error() { }",
                                     Args));

  // Try/catch
  EXPECT_TRUE(runToolOnCode(
      new SkipBodyAction,
      "void skipMe() try { an_error() } catch(error) { error };"));
  EXPECT_TRUE(runToolOnCode(
      new SkipBodyAction,
      "struct S { void skipMe() try { an_error() } catch(error) { error } };"));
  EXPECT_TRUE(
      runToolOnCode(new SkipBodyAction,
                    "void skipMe() try { an_error() } catch(error) { error; }"
                    "catch(error) { error } catch (error) { }"));
  EXPECT_FALSE(runToolOnCode(
      new SkipBodyAction,
      "void skipMe() try something;")); // don't crash while parsing

  // Template
  EXPECT_TRUE(runToolOnCode(
      new SkipBodyAction, "template<typename T> int skipMe() { an_error_here }"
                          "int x = skipMe<int>();"));
  EXPECT_FALSE(runToolOnCodeWithArgs(
      new SkipBodyAction,
      "template<typename T> int skipMeNot() { an_error_here }", Args2));
}

TEST(runToolOnCodeWithArgs, TestNoDepFile) {
  llvm::SmallString<32> DepFilePath;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("depfile", "d", DepFilePath));
  std::vector<std::string> Args;
  Args.push_back("-MMD");
  Args.push_back("-MT");
  Args.push_back(DepFilePath.str());
  Args.push_back("-MF");
  Args.push_back(DepFilePath.str());
  EXPECT_TRUE(runToolOnCodeWithArgs(new SkipBodyAction, "", Args));
  EXPECT_FALSE(llvm::sys::fs::exists(DepFilePath.str()));
  EXPECT_FALSE(llvm::sys::fs::remove(DepFilePath.str()));
}

TEST(ClangToolTest, ArgumentAdjusters) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());

  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "void a() {}");

  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory<SyntaxOnlyAction>());

  bool Found = false;
  bool Ran = false;
  ArgumentsAdjuster CheckSyntaxOnlyAdjuster =
      [&Found, &Ran](const CommandLineArguments &Args, StringRef /*unused*/) {
    Ran = true;
    if (std::find(Args.begin(), Args.end(), "-fsyntax-only") != Args.end())
      Found = true;
    return Args;
  };
  Tool.appendArgumentsAdjuster(CheckSyntaxOnlyAdjuster);
  Tool.run(Action.get());
  EXPECT_TRUE(Ran);
  EXPECT_TRUE(Found);

  Ran = Found = false;
  Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(CheckSyntaxOnlyAdjuster);
  Tool.appendArgumentsAdjuster(getClangSyntaxOnlyAdjuster());
  Tool.run(Action.get());
  EXPECT_TRUE(Ran);
  EXPECT_FALSE(Found);
}

namespace {
/// Find a target name such that looking for it in TargetRegistry by that name
/// returns the same target. We expect that there is at least one target
/// configured with this property.
std::string getAnyTarget() {
  llvm::InitializeAllTargets();
  for (const auto &Target : llvm::TargetRegistry::targets()) {
    std::string Error;
    StringRef TargetName(Target.getName());
    if (TargetName == "x86-64")
      TargetName = "x86_64";
    if (llvm::TargetRegistry::lookupTarget(TargetName, Error) == &Target) {
      return TargetName;
    }
  }
  return "";
}
}

TEST(addTargetAndModeForProgramName, AddsTargetAndMode) {
  std::string Target = getAnyTarget();
  ASSERT_FALSE(Target.empty());

  std::vector<std::string> Args = {"clang", "-foo"};
  addTargetAndModeForProgramName(Args, "");
  EXPECT_EQ((std::vector<std::string>{"clang", "-foo"}), Args);
  addTargetAndModeForProgramName(Args, Target + "-g++");
  EXPECT_EQ((std::vector<std::string>{"clang", "-target", Target,
                                      "--driver-mode=g++", "-foo"}),
            Args);
}

TEST(addTargetAndModeForProgramName, PathIgnored) {
  std::string Target = getAnyTarget();
  ASSERT_FALSE(Target.empty());

  SmallString<32> ToolPath;
  llvm::sys::path::append(ToolPath, "foo", "bar", Target + "-g++");

  std::vector<std::string> Args = {"clang", "-foo"};
  addTargetAndModeForProgramName(Args, ToolPath);
  EXPECT_EQ((std::vector<std::string>{"clang", "-target", Target,
                                      "--driver-mode=g++", "-foo"}),
            Args);
}

TEST(addTargetAndModeForProgramName, IgnoresExistingTarget) {
  std::string Target = getAnyTarget();
  ASSERT_FALSE(Target.empty());

  std::vector<std::string> Args = {"clang", "-foo", "-target", "something"};
  addTargetAndModeForProgramName(Args, Target + "-g++");
  EXPECT_EQ((std::vector<std::string>{"clang", "--driver-mode=g++", "-foo",
                                      "-target", "something"}),
            Args);

  std::vector<std::string> ArgsAlt = {"clang", "-foo", "-target=something"};
  addTargetAndModeForProgramName(ArgsAlt, Target + "-g++");
  EXPECT_EQ((std::vector<std::string>{"clang", "--driver-mode=g++", "-foo",
                                      "-target=something"}),
            ArgsAlt);
}

TEST(addTargetAndModeForProgramName, IgnoresExistingMode) {
  std::string Target = getAnyTarget();
  ASSERT_FALSE(Target.empty());

  std::vector<std::string> Args = {"clang", "-foo", "--driver-mode=abc"};
  addTargetAndModeForProgramName(Args, Target + "-g++");
  EXPECT_EQ((std::vector<std::string>{"clang", "-target", Target, "-foo",
                                      "--driver-mode=abc"}),
            Args);

  std::vector<std::string> ArgsAlt = {"clang", "-foo", "--driver-mode", "abc"};
  addTargetAndModeForProgramName(ArgsAlt, Target + "-g++");
  EXPECT_EQ((std::vector<std::string>{"clang", "-target", Target, "-foo",
                                      "--driver-mode", "abc"}),
            ArgsAlt);
}

#ifndef LLVM_ON_WIN32
TEST(ClangToolTest, BuildASTs) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());

  std::vector<std::string> Sources;
  Sources.push_back("/a.cc");
  Sources.push_back("/b.cc");
  ClangTool Tool(Compilations, Sources);

  Tool.mapVirtualFile("/a.cc", "void a() {}");
  Tool.mapVirtualFile("/b.cc", "void b() {}");

  std::vector<std::unique_ptr<ASTUnit>> ASTs;
  EXPECT_EQ(0, Tool.buildASTs(ASTs));
  EXPECT_EQ(2u, ASTs.size());
}

struct TestDiagnosticConsumer : public DiagnosticConsumer {
  TestDiagnosticConsumer() : NumDiagnosticsSeen(0) {}
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override {
    ++NumDiagnosticsSeen;
  }
  unsigned NumDiagnosticsSeen;
};

TEST(ClangToolTest, InjectDiagnosticConsumer) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "int x = undeclared;");
  TestDiagnosticConsumer Consumer;
  Tool.setDiagnosticConsumer(&Consumer);
  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory<SyntaxOnlyAction>());
  Tool.run(Action.get());
  EXPECT_EQ(1u, Consumer.NumDiagnosticsSeen);
}

TEST(ClangToolTest, InjectDiagnosticConsumerInBuildASTs) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "int x = undeclared;");
  TestDiagnosticConsumer Consumer;
  Tool.setDiagnosticConsumer(&Consumer);
  std::vector<std::unique_ptr<ASTUnit>> ASTs;
  Tool.buildASTs(ASTs);
  EXPECT_EQ(1u, ASTs.size());
  EXPECT_EQ(1u, Consumer.NumDiagnosticsSeen);
}
#endif

} // end namespace tooling
} // end namespace clang
