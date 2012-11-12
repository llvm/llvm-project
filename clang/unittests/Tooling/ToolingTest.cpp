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
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <string>

namespace clang {
namespace tooling {

namespace {
/// Takes an ast consumer and returns it from CreateASTConsumer. This only
/// works with single translation unit compilations.
class TestAction : public clang::ASTFrontendAction {
 public:
  /// Takes ownership of TestConsumer.
  explicit TestAction(clang::ASTConsumer *TestConsumer)
      : TestConsumer(TestConsumer) {}

 protected:
  virtual clang::ASTConsumer* CreateASTConsumer(
      clang::CompilerInstance& compiler, StringRef dummy) {
    /// TestConsumer will be deleted by the framework calling us.
    return TestConsumer;
  }

 private:
  clang::ASTConsumer * const TestConsumer;
};

class FindTopLevelDeclConsumer : public clang::ASTConsumer {
 public:
  explicit FindTopLevelDeclConsumer(bool *FoundTopLevelDecl)
      : FoundTopLevelDecl(FoundTopLevelDecl) {}
  virtual bool HandleTopLevelDecl(clang::DeclGroupRef DeclGroup) {
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
      new TestAction(new FindTopLevelDeclConsumer(&FoundTopLevelDecl)), ""));
#if !defined(_MSC_VER)
  EXPECT_FALSE(FoundTopLevelDecl);
#else
  // FIXME: LangOpts.MicrosoftExt appends "class type_info;"
  EXPECT_TRUE(FoundTopLevelDecl);
#endif
}

namespace {
class FindClassDeclXConsumer : public clang::ASTConsumer {
 public:
  FindClassDeclXConsumer(bool *FoundClassDeclX)
      : FoundClassDeclX(FoundClassDeclX) {}
  virtual bool HandleTopLevelDecl(clang::DeclGroupRef GroupRef) {
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
} // end namespace

TEST(runToolOnCode, FindsClassDecl) {
  bool FoundClassDeclX = false;
  EXPECT_TRUE(runToolOnCode(new TestAction(
      new FindClassDeclXConsumer(&FoundClassDeclX)), "class X;"));
  EXPECT_TRUE(FoundClassDeclX);

  FoundClassDeclX = false;
  EXPECT_TRUE(runToolOnCode(new TestAction(
      new FindClassDeclXConsumer(&FoundClassDeclX)), "class Y;"));
  EXPECT_FALSE(FoundClassDeclX);
}

TEST(newFrontendActionFactory, CreatesFrontendActionFactoryFromType) {
  llvm::OwningPtr<FrontendActionFactory> Factory(
    newFrontendActionFactory<SyntaxOnlyAction>());
  llvm::OwningPtr<FrontendAction> Action(Factory->create());
  EXPECT_TRUE(Action.get() != NULL);
}

struct IndependentFrontendActionCreator {
  ASTConsumer *newASTConsumer() {
    return new FindTopLevelDeclConsumer(NULL);
  }
};

TEST(newFrontendActionFactory, CreatesFrontendActionFactoryFromFactoryType) {
  IndependentFrontendActionCreator Creator;
  llvm::OwningPtr<FrontendActionFactory> Factory(
    newFrontendActionFactory(&Creator));
  llvm::OwningPtr<FrontendAction> Action(Factory->create());
  EXPECT_TRUE(Action.get() != NULL);
}

TEST(ToolInvocation, TestMapVirtualFile) {
  clang::FileManager Files((clang::FileSystemOptions()));
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  Args.push_back("-Idef");
  Args.push_back("-fsyntax-only");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(Args, new SyntaxOnlyAction, &Files);
  Invocation.mapVirtualFile("test.cpp", "#include <abc>\n");
  Invocation.mapVirtualFile("def/abc", "\n");
  EXPECT_TRUE(Invocation.run());
}

struct VerifyEndCallback : public EndOfSourceFileCallback {
  VerifyEndCallback() : Called(0), Matched(false) {}
  virtual void run() {
    ++Called;
  }
  ASTConsumer *newASTConsumer() {
    return new FindTopLevelDeclConsumer(&Matched);
  }
  unsigned Called;
  bool Matched;
};

#if !defined(_WIN32)
TEST(newFrontendActionFactory, InjectsEndOfSourceFileCallback) {
  VerifyEndCallback EndCallback;

  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  std::vector<std::string> Sources;
  Sources.push_back("/a.cc");
  Sources.push_back("/b.cc");
  ClangTool Tool(Compilations, Sources);

  Tool.mapVirtualFile("/a.cc", "void a() {}");
  Tool.mapVirtualFile("/b.cc", "void b() {}");

  Tool.run(newFrontendActionFactory(&EndCallback, &EndCallback));

  EXPECT_TRUE(EndCallback.Matched);
  EXPECT_EQ(2u, EndCallback.Called);
}
#endif

} // end namespace tooling
} // end namespace clang
