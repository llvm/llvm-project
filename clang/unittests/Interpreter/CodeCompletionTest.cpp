#include "clang/Interpreter/CodeCompletion.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/Sema.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#if defined(_AIX) || defined(__MVS__)
#define CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
#endif

using namespace clang;
namespace {
auto CB = clang::IncrementalCompilerBuilder();

static std::unique_ptr<Interpreter> createInterpreter() {
  auto CI = cantFail(CB.CreateCpp());
  return cantFail(clang::Interpreter::create(std::move(CI)));
}

static std::vector<std::string> runComp(clang::Interpreter &MainInterp,
                                        llvm::StringRef Input,
                                        llvm::Error &ErrR) {
  auto CI = CB.CreateCpp();
  if (auto Err = CI.takeError()) {
    ErrR = std::move(Err);
    return {};
  }

  auto Interp = clang::Interpreter::create(std::move(*CI));
  if (auto Err = Interp.takeError()) {
    // log the error and returns an empty vector;
    ErrR = std::move(Err);

    return {};
  }

  std::vector<std::string> Results;
  std::vector<std::string> Comps;
  auto *MainCI = (*Interp)->getCompilerInstance();
  auto CC = ReplCodeCompleter();
  CC.codeComplete(MainCI, Input, /* Lines */ 1, Input.size() + 1,
                  MainInterp.getCompilerInstance(), Results);

  for (auto Res : Results)
    if (Res.find(CC.Prefix) == 0)
      Comps.push_back(Res);
  return Comps;
}

static bool HostSupportsJit() {
  auto J = llvm::orc::LLJITBuilder().create();
  if (J)
    return true;
  LLVMConsumeError(llvm::wrap(J.takeError()));
  return false;
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_Sanity) {
#else
TEST(CodeCompletionTest, Sanity) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse("int foo = 12;"));
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "f", Err);
  EXPECT_EQ((size_t)2, comps.size()); // float and foo
  EXPECT_EQ(comps[0], std::string("float"));
  EXPECT_EQ(comps[1], std::string("foo"));
  EXPECT_EQ((bool)Err, false);
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_SanityNoneValid) {
#else
TEST(CodeCompletionTest, SanityNoneValid) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse("int foo = 12;"));
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "babanana", Err);
  EXPECT_EQ((size_t)0, comps.size()); // foo and float
  EXPECT_EQ((bool)Err, false);
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_TwoDecls) {
#else
TEST(CodeCompletionTest, TwoDecls) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse("int application = 12;"));
  cantFail(Interp->Parse("int apple = 12;"));
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "app", Err);
  EXPECT_EQ((size_t)2, comps.size());
  EXPECT_EQ((bool)Err, false);
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_CompFunDeclsNoError) {
#else
TEST(CodeCompletionTest, CompFunDeclsNoError) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "void app(", Err);
  EXPECT_EQ((bool)Err, false);
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_TypedDirected) {
#else
TEST(CodeCompletionTest, TypedDirected) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse("int application = 12;"));
  cantFail(Interp->Parse("char apple = '2';"));
  cantFail(Interp->Parse("void add(int &SomeInt){}"));
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("add("), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ((bool)Err, false);
  }

  cantFail(Interp->Parse("int banana = 42;"));

  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("add("), Err);
    EXPECT_EQ((size_t)2, comps.size());
    EXPECT_EQ(comps[0], "application");
    EXPECT_EQ(comps[1], "banana");
    EXPECT_EQ((bool)Err, false);
  }

  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("add(b"), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], "banana");
    EXPECT_EQ((bool)Err, false);
  }
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_SanityClasses) {
#else
TEST(CodeCompletionTest, SanityClasses) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse("struct Apple{};"));
  cantFail(Interp->Parse("void takeApple(Apple &a1){}"));
  cantFail(Interp->Parse("Apple a1;"));
  cantFail(Interp->Parse("void takeAppleCopy(Apple a1){}"));

  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, "takeApple(", Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("a1"));
    EXPECT_EQ((bool)Err, false);
  }
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("takeAppleCopy("), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("a1"));
    EXPECT_EQ((bool)Err, false);
  }
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_SubClassing) {
#else
TEST(CodeCompletionTest, SubClassing) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse("struct Fruit {};"));
  cantFail(Interp->Parse("struct Apple : Fruit{};"));
  cantFail(Interp->Parse("void takeFruit(Fruit &f){}"));
  cantFail(Interp->Parse("Apple a1;"));
  cantFail(Interp->Parse("Fruit f1;"));
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, std::string("takeFruit("), Err);
  EXPECT_EQ((size_t)2, comps.size());
  EXPECT_EQ(comps[0], std::string("a1"));
  EXPECT_EQ(comps[1], std::string("f1"));
  EXPECT_EQ((bool)Err, false);
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_MultipleArguments) {
#else
TEST(CodeCompletionTest, MultipleArguments) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse("int foo = 42;"));
  cantFail(Interp->Parse("char fowl = 'A';"));
  cantFail(Interp->Parse("void takeTwo(int &a, char b){}"));
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, std::string("takeTwo(foo,  "), Err);
  EXPECT_EQ((size_t)1, comps.size());
  EXPECT_EQ(comps[0], std::string("fowl"));
  EXPECT_EQ((bool)Err, false);
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_Methods) {
#else
TEST(CodeCompletionTest, Methods) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse(
      "struct Foo{int add(int a){return 42;} int par(int b){return 42;}};"));
  cantFail(Interp->Parse("Foo f1;"));

  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, std::string("f1."), Err);
  EXPECT_EQ((size_t)2, comps.size());
  EXPECT_EQ(comps[0], std::string("add"));
  EXPECT_EQ(comps[1], std::string("par"));
  EXPECT_EQ((bool)Err, false);
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_MethodsInvocations) {
#else
TEST(CodeCompletionTest, MethodsInvocations) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse(
      "struct Foo{int add(int a){return 42;} int par(int b){return 42;}};"));
  cantFail(Interp->Parse("Foo f1;"));
  cantFail(Interp->Parse("int a = 84;"));

  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, std::string("f1.add("), Err);
  EXPECT_EQ((size_t)1, comps.size());
  EXPECT_EQ(comps[0], std::string("a"));
  EXPECT_EQ((bool)Err, false);
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_NestedInvocations) {
#else
TEST(CodeCompletionTest, NestedInvocations) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(Interp->Parse(
      "struct Foo{int add(int a){return 42;} int par(int b){return 42;}};"));
  cantFail(Interp->Parse("Foo f1;"));
  cantFail(Interp->Parse("int a = 84;"));
  cantFail(Interp->Parse("int plus(int a, int b) { return a + b; }"));

  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, std::string("plus(42, f1.add("), Err);
  EXPECT_EQ((size_t)1, comps.size());
  EXPECT_EQ(comps[0], std::string("a"));
  EXPECT_EQ((bool)Err, false);
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(CodeCompletionTest, DISABLED_TemplateFunctions) {
#else
TEST(CodeCompletionTest, TemplateFunctions) {
#endif
  if (!HostSupportsJit())
    GTEST_SKIP();
  auto Interp = createInterpreter();
  cantFail(
      Interp->Parse("template <typename T> T id(T a) { return a;} "));
  cantFail(Interp->Parse("int apple = 84;"));
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("id<int>("), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("apple"));
    EXPECT_EQ((bool)Err, false);
  }

  cantFail(Interp->Parse(
      "template <typename T> T pickFirst(T a, T b) { return a;} "));
  cantFail(Interp->Parse("char pear = '4';"));
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("pickFirst(apple, "), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("apple"));
    EXPECT_EQ((bool)Err, false);
  }
}

} // anonymous namespace
