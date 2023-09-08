#include "clang/Interpreter/CodeCompletion.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
namespace {
auto CB = clang::IncrementalCompilerBuilder();

static std::unique_ptr<Interpreter> createInterpreter() {
  auto CI = cantFail(CB.CreateCpp());
  return cantFail(clang::Interpreter::create(std::move(CI)));
}

static std::vector<std::string> runComp(clang::Interpreter &MainInterp,
                                        llvm::StringRef Prefix,
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

  codeComplete(
      const_cast<clang::CompilerInstance *>((*Interp)->getCompilerInstance()),
      Prefix, /* Lines */ 1, Prefix.size(), MainInterp.getCompilerInstance(),
      Results);

  for (auto Res : Results)
    if (Res.find(Prefix) == 0)
      Comps.push_back(Res);

  return Comps;
}

#ifdef _AIX
TEST(CodeCompletionTest, DISABLED_Sanity) {
#else
TEST(CodeCompletionTest, Sanity) {
#endif
  auto Interp = createInterpreter();
  if (auto R = Interp->ParseAndExecute("int foo = 12;")) {
    consumeError(std::move(R));
    return;
  }
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "f", Err);
  EXPECT_EQ((size_t)2, comps.size()); // foo and float
  EXPECT_EQ(comps[0], std::string("foo"));
  EXPECT_EQ((bool)Err, false);
}

#ifdef _AIX
TEST(CodeCompletionTest, DISABLED_SanityNoneValid) {
#else
TEST(CodeCompletionTest, SanityNoneValid) {
#endif
  auto Interp = createInterpreter();
  if (auto R = Interp->ParseAndExecute("int foo = 12;")) {
    consumeError(std::move(R));
    return;
  }
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "babanana", Err);
  EXPECT_EQ((size_t)0, comps.size()); // foo and float
  EXPECT_EQ((bool)Err, false);
}

#ifdef _AIX
TEST(CodeCompletionTest, DISABLED_TwoDecls) {
#else
TEST(CodeCompletionTest, TwoDecls) {
#endif
  auto Interp = createInterpreter();
  if (auto R = Interp->ParseAndExecute("int application = 12;")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("int apple = 12;")) {
    consumeError(std::move(R));
    return;
  }
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "app", Err);
  EXPECT_EQ((size_t)2, comps.size());
  EXPECT_EQ((bool)Err, false);
}

TEST(CodeCompletionTest, CompFunDeclsNoError) {
  auto Interp = createInterpreter();
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "void app(", Err);
  EXPECT_EQ((bool)Err, false);
}

} // anonymous namespace
