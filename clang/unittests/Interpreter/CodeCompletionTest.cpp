//===- unittests/Interpreter/CodeCompletionTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InterpreterTestFixture.h"

#include "clang/Interpreter/CodeCompletion.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/Sema.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
namespace {
auto CB = clang::IncrementalCompilerBuilder();

class CodeCompletionTest : public InterpreterTestBase {
public:
  std::unique_ptr<clang::Interpreter> Interp;

  void SetUp() override {
    if (!HostSupportsJIT())
      GTEST_SKIP();
    std::unique_ptr<CompilerInstance> CI = cantFail(CB.CreateCpp());
    this->Interp = cantFail(clang::Interpreter::create(std::move(CI)));
  }

  std::vector<std::string> runComp(llvm::StringRef Input, llvm::Error &ErrR) {
    auto ComplCI = CB.CreateCpp();
    if (auto Err = ComplCI.takeError()) {
      ErrR = std::move(Err);
      return {};
    }

    auto ComplInterp = clang::Interpreter::create(std::move(*ComplCI));
    if (auto Err = ComplInterp.takeError()) {
      ErrR = std::move(Err);
      return {};
    }

    std::vector<std::string> Results;
    std::vector<std::string> Comps;
    auto *ParentCI = this->Interp->getCompilerInstance();
    auto *MainCI = (*ComplInterp)->getCompilerInstance();
    auto CC = ReplCodeCompleter();
    CC.codeComplete(MainCI, Input, /* Lines */ 1, Input.size() + 1, ParentCI,
                    Results);

    for (auto Res : Results)
      if (Res.find(CC.Prefix) == 0)
        Comps.push_back(Res);
    return Comps;
  }
};

TEST_F(CodeCompletionTest, Sanity) {
  cantFail(Interp->Parse("int foo = 12;"));
  auto Err = llvm::Error::success();
  auto comps = runComp("f", Err);
  EXPECT_EQ((size_t)2, comps.size()); // float and foo
  EXPECT_EQ(comps[0], std::string("float"));
  EXPECT_EQ(comps[1], std::string("foo"));
  EXPECT_EQ((bool)Err, false);
}

TEST_F(CodeCompletionTest, SanityNoneValid) {
  cantFail(Interp->Parse("int foo = 12;"));
  auto Err = llvm::Error::success();
  auto comps = runComp("babanana", Err);
  EXPECT_EQ((size_t)0, comps.size()); // foo and float
  EXPECT_EQ((bool)Err, false);
}

TEST_F(CodeCompletionTest, TwoDecls) {
  cantFail(Interp->Parse("int application = 12;"));
  cantFail(Interp->Parse("int apple = 12;"));
  auto Err = llvm::Error::success();
  auto comps = runComp("app", Err);
  EXPECT_EQ((size_t)2, comps.size());
  EXPECT_EQ((bool)Err, false);
}

TEST_F(CodeCompletionTest, CompFunDeclsNoError) {
  auto Err = llvm::Error::success();
  auto comps = runComp("void app(", Err);
  EXPECT_EQ((bool)Err, false);
}

TEST_F(CodeCompletionTest, TypedDirected) {
  cantFail(Interp->Parse("int application = 12;"));
  cantFail(Interp->Parse("char apple = '2';"));
  cantFail(Interp->Parse("void add(int &SomeInt){}"));
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(std::string("add("), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ((bool)Err, false);
  }

  cantFail(Interp->Parse("int banana = 42;"));

  {
    auto Err = llvm::Error::success();
    auto comps = runComp(std::string("add("), Err);
    EXPECT_EQ((size_t)2, comps.size());
    EXPECT_EQ(comps[0], "application");
    EXPECT_EQ(comps[1], "banana");
    EXPECT_EQ((bool)Err, false);
  }

  {
    auto Err = llvm::Error::success();
    auto comps = runComp(std::string("add(b"), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], "banana");
    EXPECT_EQ((bool)Err, false);
  }
}

TEST_F(CodeCompletionTest, SanityClasses) {
  cantFail(Interp->Parse("struct Apple{};"));
  cantFail(Interp->Parse("void takeApple(Apple &a1){}"));
  cantFail(Interp->Parse("Apple a1;"));
  cantFail(Interp->Parse("void takeAppleCopy(Apple a1){}"));

  {
    auto Err = llvm::Error::success();
    auto comps = runComp("takeApple(", Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("a1"));
    EXPECT_EQ((bool)Err, false);
  }
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(std::string("takeAppleCopy("), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("a1"));
    EXPECT_EQ((bool)Err, false);
  }
}

TEST_F(CodeCompletionTest, SubClassing) {
  cantFail(Interp->Parse("struct Fruit {};"));
  cantFail(Interp->Parse("struct Apple : Fruit{};"));
  cantFail(Interp->Parse("void takeFruit(Fruit &f){}"));
  cantFail(Interp->Parse("Apple a1;"));
  cantFail(Interp->Parse("Fruit f1;"));
  auto Err = llvm::Error::success();
  auto comps = runComp(std::string("takeFruit("), Err);
  EXPECT_EQ((size_t)2, comps.size());
  EXPECT_EQ(comps[0], std::string("a1"));
  EXPECT_EQ(comps[1], std::string("f1"));
  EXPECT_EQ((bool)Err, false);
}

TEST_F(CodeCompletionTest, MultipleArguments) {
  cantFail(Interp->Parse("int foo = 42;"));
  cantFail(Interp->Parse("char fowl = 'A';"));
  cantFail(Interp->Parse("void takeTwo(int &a, char b){}"));
  auto Err = llvm::Error::success();
  auto comps = runComp(std::string("takeTwo(foo,  "), Err);
  EXPECT_EQ((size_t)1, comps.size());
  EXPECT_EQ(comps[0], std::string("fowl"));
  EXPECT_EQ((bool)Err, false);
}

TEST_F(CodeCompletionTest, Methods) {
  cantFail(Interp->Parse(
      "struct Foo{int add(int a){return 42;} int par(int b){return 42;}};"));
  cantFail(Interp->Parse("Foo f1;"));

  auto Err = llvm::Error::success();
  auto comps = runComp(std::string("f1."), Err);
  EXPECT_EQ((size_t)2, comps.size());
  EXPECT_EQ(comps[0], std::string("add"));
  EXPECT_EQ(comps[1], std::string("par"));
  EXPECT_EQ((bool)Err, false);
}

TEST_F(CodeCompletionTest, MethodsInvocations) {
  cantFail(Interp->Parse(
      "struct Foo{int add(int a){return 42;} int par(int b){return 42;}};"));
  cantFail(Interp->Parse("Foo f1;"));
  cantFail(Interp->Parse("int a = 84;"));

  auto Err = llvm::Error::success();
  auto comps = runComp(std::string("f1.add("), Err);
  EXPECT_EQ((size_t)1, comps.size());
  EXPECT_EQ(comps[0], std::string("a"));
  EXPECT_EQ((bool)Err, false);
}

TEST_F(CodeCompletionTest, NestedInvocations) {
  cantFail(Interp->Parse(
      "struct Foo{int add(int a){return 42;} int par(int b){return 42;}};"));
  cantFail(Interp->Parse("Foo f1;"));
  cantFail(Interp->Parse("int a = 84;"));
  cantFail(Interp->Parse("int plus(int a, int b) { return a + b; }"));

  auto Err = llvm::Error::success();
  auto comps = runComp(std::string("plus(42, f1.add("), Err);
  EXPECT_EQ((size_t)1, comps.size());
  EXPECT_EQ(comps[0], std::string("a"));
  EXPECT_EQ((bool)Err, false);
}

TEST_F(CodeCompletionTest, TemplateFunctions) {
  cantFail(
      Interp->Parse("template <typename T> T id(T a) { return a;} "));
  cantFail(Interp->Parse("int apple = 84;"));
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(std::string("id<int>("), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("apple"));
    EXPECT_EQ((bool)Err, false);
  }

  cantFail(Interp->Parse(
      "template <typename T> T pickFirst(T a, T b) { return a;} "));
  cantFail(Interp->Parse("char pear = '4';"));
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(std::string("pickFirst(apple, "), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("apple"));
    EXPECT_EQ((bool)Err, false);
  }
}

} // anonymous namespace
