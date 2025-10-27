//===- unittest/Tooling/ToolingTest.cpp - Tooling unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Tooling.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Testing/CommandLineArgs.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>
#include <vector>

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
  EXPECT_TRUE(runToolOnCode(
      std::make_unique<TestAction>(
          std::make_unique<FindTopLevelDeclConsumer>(&FoundTopLevelDecl)),
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

struct TestDiagnosticConsumer : public DiagnosticConsumer {
  TestDiagnosticConsumer() : NumDiagnosticsSeen(0) {}
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override {
    ++NumDiagnosticsSeen;
  }
  unsigned NumDiagnosticsSeen;
};
} // end namespace

TEST(runToolOnCode, FindsClassDecl) {
  bool FoundClassDeclX = false;
  EXPECT_TRUE(runToolOnCode(
      std::make_unique<TestAction>(
          std::make_unique<FindClassDeclXConsumer>(&FoundClassDeclX)),
      "class X;"));
  EXPECT_TRUE(FoundClassDeclX);

  FoundClassDeclX = false;
  EXPECT_TRUE(runToolOnCode(
      std::make_unique<TestAction>(
          std::make_unique<FindClassDeclXConsumer>(&FoundClassDeclX)),
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

TEST(buildASTFromCode, ReportsErrors) {
  TestDiagnosticConsumer Consumer;
  std::unique_ptr<ASTUnit> AST = buildASTFromCodeWithArgs(
      "int x = \"A\";", {}, "input.cc", "clang-tool",
      std::make_shared<PCHContainerOperations>(),
      getClangStripDependencyFileAdjuster(), FileContentMappings(), &Consumer);
  EXPECT_TRUE(AST.get());
  EXPECT_EQ(1u, Consumer.NumDiagnosticsSeen);
}

TEST(buildASTFromCode, FileSystem) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  InMemoryFileSystem->addFile("included_file.h", 0,
                              llvm::MemoryBuffer::getMemBufferCopy("class X;"));
  std::unique_ptr<ASTUnit> AST = buildASTFromCodeWithArgs(
      R"(#include "included_file.h")", {}, "input.cc", "clang-tool",
      std::make_shared<PCHContainerOperations>(),
      getClangStripDependencyFileAdjuster(), FileContentMappings(), nullptr,
      InMemoryFileSystem);
  ASSERT_TRUE(AST.get());
  EXPECT_TRUE(FindClassDeclX(AST.get()));
}

TEST(newFrontendActionFactory, CreatesFrontendActionFactoryFromType) {
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory<SyntaxOnlyAction>());
  std::unique_ptr<FrontendAction> Action(Factory->create());
  EXPECT_TRUE(Action.get() != nullptr);
}

struct IndependentFrontendActionCreator {
  std::unique_ptr<ASTConsumer> newASTConsumer() {
    return std::make_unique<FindTopLevelDeclConsumer>(nullptr);
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
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  auto Files = llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions(),
                                                      OverlayFileSystem);
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  Args.push_back("-Idef");
  Args.push_back("-fsyntax-only");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(
      Args, std::make_unique<SyntaxOnlyAction>(), Files.get());
  InMemoryFileSystem->addFile(
      "test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("#include <abc>\n"));
  InMemoryFileSystem->addFile("def/abc", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  EXPECT_TRUE(Invocation.run());
}

TEST(ToolInvocation, TestVirtualModulesCompilation) {
  // FIXME: Currently, this only tests that we don't exit with an error if a
  // mapped module.modulemap is found on the include path. In the future, expand
  // this test to run a full modules enabled compilation, so we make sure we can
  // rerun modules compilations with a virtual file system.
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  auto Files = llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions(),
                                                      OverlayFileSystem);
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  Args.push_back("-Idef");
  Args.push_back("-fsyntax-only");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(
      Args, std::make_unique<SyntaxOnlyAction>(), Files.get());
  InMemoryFileSystem->addFile(
      "test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("#include <abc>\n"));
  InMemoryFileSystem->addFile("def/abc", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  // Add a module.modulemap file in the include directory of our header, so we
  // trigger the module.modulemap header search logic.
  InMemoryFileSystem->addFile("def/module.modulemap", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  EXPECT_TRUE(Invocation.run());
}

TEST(ToolInvocation, DiagnosticsEngineProperlyInitializedForCC1Construction) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  auto Files = llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions(),
                                                      OverlayFileSystem);

  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  // Unknown warning option will result in a warning.
  Args.push_back("-fexpensive-optimizations");
  // Argument that will suppress the warning above.
  Args.push_back("-Wno-ignored-optimization-argument");
  Args.push_back("-E");
  Args.push_back("test.cpp");

  clang::tooling::ToolInvocation Invocation(
      Args, std::make_unique<SyntaxOnlyAction>(), Files.get());
  InMemoryFileSystem->addFile("test.cpp", 0,
                              llvm::MemoryBuffer::getMemBuffer(""));
  TextDiagnosticBuffer Consumer;
  Invocation.setDiagnosticConsumer(&Consumer);
  EXPECT_TRUE(Invocation.run());
  // Check that the warning was ignored due to the '-Wno-xxx' argument.
  EXPECT_EQ(std::distance(Consumer.warn_begin(), Consumer.warn_end()), 0u);
}

TEST(ToolInvocation, CustomDiagnosticOptionsOverwriteParsedOnes) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  auto Files = llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions(),
                                                      OverlayFileSystem);

  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  // Unknown warning option will result in a warning.
  Args.push_back("-fexpensive-optimizations");
  // Argument that will suppress the warning above.
  Args.push_back("-Wno-ignored-optimization-argument");
  Args.push_back("-E");
  Args.push_back("test.cpp");

  clang::tooling::ToolInvocation Invocation(
      Args, std::make_unique<SyntaxOnlyAction>(), Files.get());
  InMemoryFileSystem->addFile("test.cpp", 0,
                              llvm::MemoryBuffer::getMemBuffer(""));
  TextDiagnosticBuffer Consumer;
  Invocation.setDiagnosticConsumer(&Consumer);

  // Inject custom `DiagnosticOptions` for command-line parsing.
  DiagnosticOptions DiagOpts;
  Invocation.setDiagnosticOptions(&DiagOpts);

  EXPECT_TRUE(Invocation.run());
  // Check that the warning was issued during command-line parsing due to the
  // custom `DiagnosticOptions` without '-Wno-xxx'.
  EXPECT_EQ(std::distance(Consumer.warn_begin(), Consumer.warn_end()), 1u);
}

struct DiagnosticConsumerExpectingSourceManager : public DiagnosticConsumer {
  bool SawSourceManager;

  DiagnosticConsumerExpectingSourceManager() : SawSourceManager(false) {}

  void HandleDiagnostic(clang::DiagnosticsEngine::Level,
                        const clang::Diagnostic &info) override {
    SawSourceManager = info.hasSourceManager();
  }
};

TEST(ToolInvocation, DiagConsumerExpectingSourceManager) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  auto Files = llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions(),
                                                      OverlayFileSystem);
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  // Note: intentional error; user probably meant -ferror-limit=0.
  Args.push_back("-ferror-limit=-1");
  Args.push_back("-fsyntax-only");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(
      Args, std::make_unique<SyntaxOnlyAction>(), Files.get());
  InMemoryFileSystem->addFile(
      "test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("int main() {}\n"));

  DiagnosticConsumerExpectingSourceManager Consumer;
  Invocation.setDiagnosticConsumer(&Consumer);

  EXPECT_TRUE(Invocation.run());
  EXPECT_TRUE(Consumer.SawSourceManager);
}

TEST(ToolInvocation, CC1Args) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  auto Files = llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions(),
                                                      OverlayFileSystem);
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  Args.push_back("-cc1");
  Args.push_back("-fsyntax-only");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(
      Args, std::make_unique<SyntaxOnlyAction>(), Files.get());
  InMemoryFileSystem->addFile(
      "test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("void foo(void);\n"));
  EXPECT_TRUE(Invocation.run());
}

TEST(ToolInvocation, CC1ArgsInvalid) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  auto Files = llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions(),
                                                      OverlayFileSystem);
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  Args.push_back("-cc1");
  Args.push_back("-invalid-arg");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(
      Args, std::make_unique<SyntaxOnlyAction>(), Files.get());
  InMemoryFileSystem->addFile(
      "test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("void foo(void);\n"));
  EXPECT_FALSE(Invocation.run());
}

namespace {
/// Overlays the real filesystem with the given VFS and returns the result.
llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem>
overlayRealFS(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  auto RFS = llvm::vfs::getRealFileSystem();
  auto OverlayFS = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(RFS);
  OverlayFS->pushOverlay(VFS);
  return OverlayFS;
}

struct CommandLineExtractorTest : public ::testing::Test {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFS;
  DiagnosticOptions DiagOpts;
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Diags;
  driver::Driver Driver;

public:
  CommandLineExtractorTest()
      : InMemoryFS(new llvm::vfs::InMemoryFileSystem),
        Diags(CompilerInstance::createDiagnostics(*InMemoryFS, DiagOpts)),
        Driver("clang", llvm::sys::getDefaultTargetTriple(), *Diags,
               "clang LLVM compiler", overlayRealFS(InMemoryFS)) {}

  void addFile(StringRef Name, StringRef Content) {
    InMemoryFS->addFile(Name, 0, llvm::MemoryBuffer::getMemBuffer(Content));
  }

  const llvm::opt::ArgStringList *
  extractCC1Arguments(llvm::ArrayRef<const char *> Argv) {
    const std::unique_ptr<driver::Compilation> Compilation(
        Driver.BuildCompilation(llvm::ArrayRef(Argv)));

    return getCC1Arguments(Diags.get(), Compilation.get());
  }
};
} // namespace

TEST_F(CommandLineExtractorTest, AcceptOffloading) {
  addFile("test.c", "int main() {}\n");
  const char *Args[] = {"clang",     "-target",  "arm64-apple-macosx11.0.0",
                        "-x",        "hip",      "test.c",
                        "-nogpulib", "-nogpuinc"};
  EXPECT_NE(extractCC1Arguments(Args), nullptr);
}

TEST_F(CommandLineExtractorTest, AcceptOffloadingCompile) {
  addFile("test.c", "int main() {}\n");
  const char *Args[] = {"clang",  "-target",   "arm64-apple-macosx11.0.0",
                        "-c",     "-x",        "hip",
                        "test.c", "-nogpulib", "-nogpuinc"};
  EXPECT_NE(extractCC1Arguments(Args), nullptr);
}

TEST_F(CommandLineExtractorTest, AcceptOffloadingSyntaxOnly) {
  addFile("test.c", "int main() {}\n");
  const char *Args[] = {
      "clang",         "-target",   "arm64-apple-macosx11.0.0",
      "-fsyntax-only", "-x",        "hip",
      "test.c",        "-nogpulib", "-nogpuinc"};
  EXPECT_NE(extractCC1Arguments(Args), nullptr);
}

TEST_F(CommandLineExtractorTest, AcceptExternalAssembler) {
  addFile("test.c", "int main() {}\n");
  const char *Args[] = {
      "clang", "-target", "arm64-apple-macosx11.0.0", "-fno-integrated-as",
      "-c",    "test.c"};
  EXPECT_NE(extractCC1Arguments(Args), nullptr);
}

TEST_F(CommandLineExtractorTest, AcceptEmbedBitcode) {
  addFile("test.c", "int main() {}\n");
  const char *Args[] = {"clang", "-target",         "arm64-apple-macosx11.0.0",
                        "-c",    "-fembed-bitcode", "test.c"};
  EXPECT_NE(extractCC1Arguments(Args), nullptr);
}

TEST_F(CommandLineExtractorTest, AcceptSaveTemps) {
  addFile("test.c", "int main() {}\n");
  const char *Args[] = {"clang", "-target",     "arm64-apple-macosx11.0.0",
                        "-c",    "-save-temps", "test.c"};
  EXPECT_NE(extractCC1Arguments(Args), nullptr);
}

TEST_F(CommandLineExtractorTest, AcceptPreprocessedInputFile) {
  addFile("test.i", "int main() {}\n");
  const char *Args[] = {"clang", "-target", "arm64-apple-macosx11.0.0", "-c",
                        "test.i"};
  EXPECT_NE(extractCC1Arguments(Args), nullptr);
}

TEST_F(CommandLineExtractorTest, RejectMultipleArchitectures) {
  addFile("test.c", "int main() {}\n");
  const char *Args[] = {"clang", "-target", "arm64-apple-macosx11.0.0",
                        "-arch", "x86_64",  "-arch",
                        "arm64", "-c",      "test.c"};
  EXPECT_EQ(extractCC1Arguments(Args), nullptr);
}

TEST_F(CommandLineExtractorTest, RejectMultipleInputFiles) {
  addFile("one.c", "void one() {}\n");
  addFile("two.c", "void two() {}\n");
  const char *Args[] = {"clang", "-target", "arm64-apple-macosx11.0.0",
                        "-c",    "one.c",   "two.c"};
  EXPECT_EQ(extractCC1Arguments(Args), nullptr);
}

struct VerifyEndCallback : public SourceFileCallbacks {
  VerifyEndCallback() : BeginCalled(0), EndCalled(0), Matched(false) {}
  bool handleBeginSource(CompilerInstance &CI) override {
    ++BeginCalled;
    return true;
  }
  void handleEndSource() override { ++EndCalled; }
  std::unique_ptr<ASTConsumer> newASTConsumer() {
    return std::make_unique<FindTopLevelDeclConsumer>(&Matched);
  }
  unsigned BeginCalled;
  unsigned EndCalled;
  bool Matched;
};

#if !defined(_WIN32)
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
    return std::make_unique<SkipBodyConsumer>();
  }
};

TEST(runToolOnCode, TestSkipFunctionBody) {
  std::vector<std::string> Args = {"-std=c++11"};
  std::vector<std::string> Args2 = {"-fno-delayed-template-parsing"};

  EXPECT_TRUE(runToolOnCode(std::make_unique<SkipBodyAction>(),
                            "int skipMe() { an_error_here }"));
  EXPECT_FALSE(runToolOnCode(std::make_unique<SkipBodyAction>(),
                             "int skipMeNot() { an_error_here }"));

  // Test constructors with initializers
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<SkipBodyAction>(),
      "struct skipMe { skipMe() : an_error() { more error } };", Args));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<SkipBodyAction>(), "struct skipMe { skipMe(); };"
                          "skipMe::skipMe() : an_error([](){;}) { more error }",
      Args));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<SkipBodyAction>(), "struct skipMe { skipMe(); };"
                          "skipMe::skipMe() : an_error{[](){;}} { more error }",
      Args));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<SkipBodyAction>(),
      "struct skipMe { skipMe(); };"
      "skipMe::skipMe() : a<b<c>(e)>>(), f{}, g() { error }",
      Args));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<SkipBodyAction>(), "struct skipMe { skipMe() : bases()... { error } };",
      Args));

  EXPECT_FALSE(runToolOnCodeWithArgs(
      std::make_unique<SkipBodyAction>(), "struct skipMeNot { skipMeNot() : an_error() { } };",
      Args));
  EXPECT_FALSE(runToolOnCodeWithArgs(std::make_unique<SkipBodyAction>(),
                                     "struct skipMeNot { skipMeNot(); };"
                                     "skipMeNot::skipMeNot() : an_error() { }",
                                     Args));

  // Try/catch
  EXPECT_TRUE(runToolOnCode(
      std::make_unique<SkipBodyAction>(),
      "void skipMe() try { an_error() } catch(error) { error };"));
  EXPECT_TRUE(runToolOnCode(
      std::make_unique<SkipBodyAction>(),
      "struct S { void skipMe() try { an_error() } catch(error) { error } };"));
  EXPECT_TRUE(
      runToolOnCode(std::make_unique<SkipBodyAction>(),
                    "void skipMe() try { an_error() } catch(error) { error; }"
                    "catch(error) { error } catch (error) { }"));
  EXPECT_FALSE(runToolOnCode(
      std::make_unique<SkipBodyAction>(),
      "void skipMe() try something;")); // don't crash while parsing

  // Template
  EXPECT_TRUE(runToolOnCode(
      std::make_unique<SkipBodyAction>(), "template<typename T> int skipMe() { an_error_here }"
                          "int x = skipMe<int>();"));
  EXPECT_FALSE(runToolOnCodeWithArgs(
      std::make_unique<SkipBodyAction>(),
      "template<typename T> int skipMeNot() { an_error_here }", Args2));

  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<SkipBodyAction>(),
      "__inline __attribute__((__gnu_inline__)) void skipMe() {}",
      {"--cuda-host-only", "-nocudainc", "-xcuda"}));
}

TEST(runToolOnCodeWithArgs, TestNoDepFile) {
  llvm::SmallString<32> DepFilePath;
  ASSERT_FALSE(llvm::sys::fs::getPotentiallyUniqueTempFileName("depfile", "d",
                                                               DepFilePath));
  std::vector<std::string> Args;
  Args.push_back("-MMD");
  Args.push_back("-MT");
  Args.push_back(std::string(DepFilePath.str()));
  Args.push_back("-MF");
  Args.push_back(std::string(DepFilePath.str()));
  EXPECT_TRUE(runToolOnCodeWithArgs(std::make_unique<SkipBodyAction>(), "", Args));
  EXPECT_FALSE(llvm::sys::fs::exists(DepFilePath.str()));
  EXPECT_FALSE(llvm::sys::fs::remove(DepFilePath.str()));
}

struct CheckColoredDiagnosticsAction : public clang::ASTFrontendAction {
  CheckColoredDiagnosticsAction(bool ShouldShowColor)
      : ShouldShowColor(ShouldShowColor) {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef) override {
    if (Compiler.getDiagnosticOpts().ShowColors != ShouldShowColor)
      Compiler.getDiagnostics().Report(
          Compiler.getDiagnostics().getCustomDiagID(
              DiagnosticsEngine::Fatal,
              "getDiagnosticOpts().ShowColors != ShouldShowColor"));
    return std::make_unique<ASTConsumer>();
  }

private:
  bool ShouldShowColor = true;
};

TEST(runToolOnCodeWithArgs, DiagnosticsColor) {
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<CheckColoredDiagnosticsAction>(true), "",
      {"-fcolor-diagnostics"}));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<CheckColoredDiagnosticsAction>(false), "",
      {"-fno-color-diagnostics"}));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<CheckColoredDiagnosticsAction>(true), "",
      {"-fno-color-diagnostics", "-fcolor-diagnostics"}));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<CheckColoredDiagnosticsAction>(false), "",
      {"-fcolor-diagnostics", "-fno-color-diagnostics"}));
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<CheckColoredDiagnosticsAction>(true), "",
      {"-fno-color-diagnostics", "-fdiagnostics-color=always"}));

  // Check that this test would fail if ShowColors is not what it should.
  EXPECT_FALSE(runToolOnCodeWithArgs(
      std::make_unique<CheckColoredDiagnosticsAction>(false), "",
      {"-fcolor-diagnostics"}));
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
    if (llvm::is_contained(Args, "-fsyntax-only"))
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

TEST(ClangToolTest, NoDoubleSyntaxOnly) {
  FixedCompilationDatabase Compilations("/", {"-fsyntax-only"});

  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "void a() {}");

  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory<SyntaxOnlyAction>());

  size_t SyntaxOnlyCount = 0;
  ArgumentsAdjuster CheckSyntaxOnlyAdjuster =
      [&SyntaxOnlyCount](const CommandLineArguments &Args,
                         StringRef /*unused*/) {
        for (llvm::StringRef Arg : Args) {
          if (Arg == "-fsyntax-only")
            ++SyntaxOnlyCount;
        }
        return Args;
      };

  Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(getClangSyntaxOnlyAdjuster());
  Tool.appendArgumentsAdjuster(CheckSyntaxOnlyAdjuster);
  Tool.run(Action.get());
  EXPECT_EQ(SyntaxOnlyCount, 1U);
}

TEST(ClangToolTest, NoOutputCommands) {
  FixedCompilationDatabase Compilations("/", {"-save-temps", "-save-temps=cwd",
                                              "--save-temps",
                                              "--save-temps=somedir"});

  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "void a() {}");

  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory<SyntaxOnlyAction>());

  const std::vector<llvm::StringRef> OutputCommands = {"-save-temps"};
  bool Ran = false;
  ArgumentsAdjuster CheckSyntaxOnlyAdjuster =
      [&OutputCommands, &Ran](const CommandLineArguments &Args,
                              StringRef /*unused*/) {
        for (llvm::StringRef Arg : Args) {
          for (llvm::StringRef OutputCommand : OutputCommands)
            EXPECT_FALSE(Arg.contains(OutputCommand));
        }
        Ran = true;
        return Args;
      };

  Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(getClangSyntaxOnlyAdjuster());
  Tool.appendArgumentsAdjuster(CheckSyntaxOnlyAdjuster);
  Tool.run(Action.get());
  EXPECT_TRUE(Ran);
}

TEST(ClangToolTest, BaseVirtualFileSystemUsage) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);

  InMemoryFileSystem->addFile(
      "a.cpp", 0, llvm::MemoryBuffer::getMemBuffer("int main() {}"));

  ClangTool Tool(Compilations, std::vector<std::string>(1, "a.cpp"),
                 std::make_shared<PCHContainerOperations>(), OverlayFileSystem);
  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory<SyntaxOnlyAction>());
  EXPECT_EQ(0, Tool.run(Action.get()));
}

// Check getClangStripDependencyFileAdjuster doesn't strip args after -MD/-MMD.
TEST(ClangToolTest, StripDependencyFileAdjuster) {
  FixedCompilationDatabase Compilations("/", {"-MD", "-c", "-MMD", "-w"});

  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "void a() {}");

  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory<SyntaxOnlyAction>());

  CommandLineArguments FinalArgs;
  ArgumentsAdjuster CheckFlagsAdjuster =
    [&FinalArgs](const CommandLineArguments &Args, StringRef /*unused*/) {
      FinalArgs = Args;
      return Args;
    };
  Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(getClangStripDependencyFileAdjuster());
  Tool.appendArgumentsAdjuster(CheckFlagsAdjuster);
  Tool.run(Action.get());

  auto HasFlag = [&FinalArgs](const std::string &Flag) {
    return llvm::is_contained(FinalArgs, Flag);
  };
  EXPECT_FALSE(HasFlag("-MD"));
  EXPECT_FALSE(HasFlag("-MMD"));
  EXPECT_TRUE(HasFlag("-c"));
  EXPECT_TRUE(HasFlag("-w"));
}

// Check getClangStripDependencyFileAdjuster strips /showIncludes and variants
TEST(ClangToolTest, StripDependencyFileAdjusterShowIncludes) {
  FixedCompilationDatabase Compilations(
      "/", {"/showIncludes", "/showIncludes:user", "-showIncludes",
            "-showIncludes:user", "-c"});

  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "void a() {}");

  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory<SyntaxOnlyAction>());

  CommandLineArguments FinalArgs;
  ArgumentsAdjuster CheckFlagsAdjuster =
      [&FinalArgs](const CommandLineArguments &Args, StringRef /*unused*/) {
        FinalArgs = Args;
        return Args;
      };
  Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(getClangStripDependencyFileAdjuster());
  Tool.appendArgumentsAdjuster(CheckFlagsAdjuster);
  Tool.run(Action.get());

  auto HasFlag = [&FinalArgs](const std::string &Flag) {
    return llvm::is_contained(FinalArgs, Flag);
  };
  EXPECT_FALSE(HasFlag("/showIncludes"));
  EXPECT_FALSE(HasFlag("/showIncludes:user"));
  EXPECT_FALSE(HasFlag("-showIncludes"));
  EXPECT_FALSE(HasFlag("-showIncludes:user"));
  EXPECT_TRUE(HasFlag("-c"));
}

// Check getClangStripDependencyFileAdjuster doesn't strip args when using the
// MSVC cl.exe driver
TEST(ClangToolTest, StripDependencyFileAdjusterMsvc) {
  FixedCompilationDatabase Compilations(
      "/", {"--driver-mode=cl", "-MD", "-MDd", "-MT", "-O1", "-MTd", "-MP"});

  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "void a() {}");

  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory<SyntaxOnlyAction>());

  CommandLineArguments FinalArgs;
  ArgumentsAdjuster CheckFlagsAdjuster =
      [&FinalArgs](const CommandLineArguments &Args, StringRef /*unused*/) {
        FinalArgs = Args;
        return Args;
      };
  Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(getClangStripDependencyFileAdjuster());
  Tool.appendArgumentsAdjuster(CheckFlagsAdjuster);
  Tool.run(Action.get());

  auto HasFlag = [&FinalArgs](const std::string &Flag) {
    return llvm::is_contained(FinalArgs, Flag);
  };
  EXPECT_TRUE(HasFlag("-MD"));
  EXPECT_TRUE(HasFlag("-MDd"));
  EXPECT_TRUE(HasFlag("-MT"));
  EXPECT_TRUE(HasFlag("-O1"));
  EXPECT_TRUE(HasFlag("-MTd"));
  EXPECT_TRUE(HasFlag("-MP"));
}

// Check getClangStripPluginsAdjuster strips plugin related args.
TEST(ClangToolTest, StripPluginsAdjuster) {
  FixedCompilationDatabase Compilations(
      "/", {"-Xclang", "-add-plugin", "-Xclang", "random-plugin"});

  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "void a() {}");

  std::unique_ptr<FrontendActionFactory> Action(
      newFrontendActionFactory<SyntaxOnlyAction>());

  CommandLineArguments FinalArgs;
  ArgumentsAdjuster CheckFlagsAdjuster =
      [&FinalArgs](const CommandLineArguments &Args, StringRef /*unused*/) {
        FinalArgs = Args;
        return Args;
      };
  Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(getStripPluginsAdjuster());
  Tool.appendArgumentsAdjuster(CheckFlagsAdjuster);
  Tool.run(Action.get());

  auto HasFlag = [&FinalArgs](const std::string &Flag) {
    return llvm::is_contained(FinalArgs, Flag);
  };
  EXPECT_FALSE(HasFlag("-Xclang"));
  EXPECT_FALSE(HasFlag("-add-plugin"));
  EXPECT_FALSE(HasFlag("-random-plugin"));
}

TEST(addTargetAndModeForProgramName, AddsTargetAndMode) {
  llvm::InitializeAllTargets();

  std::string Target = getAnyTargetForTesting();
  ASSERT_FALSE(Target.empty());

  std::vector<std::string> Args = {"clang", "-foo"};
  addTargetAndModeForProgramName(Args, "");
  EXPECT_EQ((std::vector<std::string>{"clang", "-foo"}), Args);
  addTargetAndModeForProgramName(Args, Target + "-g++");
  EXPECT_EQ((std::vector<std::string>{"clang", "--target=" + Target,
                                      "--driver-mode=g++", "-foo"}),
            Args);
}

TEST(addTargetAndModeForProgramName, PathIgnored) {
  llvm::InitializeAllTargets();
  std::string Target = getAnyTargetForTesting();
  ASSERT_FALSE(Target.empty());

  SmallString<32> ToolPath;
  llvm::sys::path::append(ToolPath, "foo", "bar", Target + "-g++");

  std::vector<std::string> Args = {"clang", "-foo"};
  addTargetAndModeForProgramName(Args, ToolPath);
  EXPECT_EQ((std::vector<std::string>{"clang", "--target=" + Target,
                                      "--driver-mode=g++", "-foo"}),
            Args);
}

TEST(addTargetAndModeForProgramName, IgnoresExistingTarget) {
  llvm::InitializeAllTargets();
  std::string Target = getAnyTargetForTesting();
  ASSERT_FALSE(Target.empty());

  std::vector<std::string> Args = {"clang", "-foo", "-target", "something"};
  addTargetAndModeForProgramName(Args, Target + "-g++");
  EXPECT_EQ((std::vector<std::string>{"clang", "--driver-mode=g++", "-foo",
                                      "-target", "something"}),
            Args);

  std::vector<std::string> ArgsAlt = {"clang", "-foo", "--target=something"};
  addTargetAndModeForProgramName(ArgsAlt, Target + "-g++");
  EXPECT_EQ((std::vector<std::string>{"clang", "--driver-mode=g++", "-foo",
                                      "--target=something"}),
            ArgsAlt);
}

TEST(addTargetAndModeForProgramName, IgnoresExistingMode) {
  llvm::InitializeAllTargets();
  std::string Target = getAnyTargetForTesting();
  ASSERT_FALSE(Target.empty());

  std::vector<std::string> Args = {"clang", "-foo", "--driver-mode=abc"};
  addTargetAndModeForProgramName(Args, Target + "-g++");
  EXPECT_EQ((std::vector<std::string>{"clang", "--target=" + Target, "-foo",
                                      "--driver-mode=abc"}),
            Args);
}

#ifndef _WIN32
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

TEST(runToolOnCode, TestResetDiagnostics) {
  // This is a tool that resets the diagnostic during the compilation.
  struct ResetDiagnosticAction : public clang::ASTFrontendAction {
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                   StringRef) override {
      struct Consumer : public clang::ASTConsumer {
        bool HandleTopLevelDecl(clang::DeclGroupRef D) override {
          auto &Diags = (*D.begin())->getASTContext().getDiagnostics();
          // Ignore any error
          Diags.Reset();
          // Disable warnings because computing the CFG might crash.
          Diags.setIgnoreAllWarnings(true);
          return true;
        }
      };
      return std::make_unique<Consumer>();
    }
  };

  // Should not crash
  EXPECT_FALSE(
      runToolOnCode(std::make_unique<ResetDiagnosticAction>(),
                    "struct Foo { Foo(int); ~Foo(); struct Fwd _fwd; };"
                    "void func() { long x; Foo f(x); }"));
}

} // end namespace tooling
} // end namespace clang
