//=== unittests/Interpreter/IncrementalCompilerBuilderTest.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InterpreterTestFixture.h"

#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class IncrementalCompilerBuilderInfoStreamTest : public InterpreterTestBase {};

// Usually FrontendAction takes the raw pointers and wraps them back into
// unique_ptrs in InitializeFileRemapping()
static void cleanupRemappedFileBuffers(CompilerInstance &CI) {
  for (const auto &RB : CI.getPreprocessorOpts().RemappedFileBuffers) {
    delete RB.second;
  }
  CI.getPreprocessorOpts().clearRemappedFiles();
}

TEST(IncrementalCompilerBuilder, SetCompilerArgs) {
  std::vector<const char *> ClangArgv = {"-Xclang", "-ast-dump-all"};
  auto CB = clang::IncrementalCompilerBuilder();
  CB.SetCompilerArgs(ClangArgv);
  auto CI = cantFail(CB.CreateCpp());
  EXPECT_TRUE(CI->getFrontendOpts().ASTDumpAll);
  cleanupRemappedFileBuffers(*CI);
}

TEST_F(IncrementalCompilerBuilderInfoStreamTest, RoutesVerboseOutputToInfos) {
  std::string Output;
  llvm::raw_string_ostream OS(Output);
  llvm::setInfoOutputStream(OS);
  llvm::scope_exit ResetInfoOutput([] { llvm::resetInfoOutputStream(); });

  std::vector<const char *> ClangArgv = {"-v", "-nostdinc", "-nostdinc++"};
  auto CB = clang::IncrementalCompilerBuilder();
  CB.SetCompilerArgs(ClangArgv);

  auto CI = cantFail(CB.CreateCpp());
  OS.flush();
  EXPECT_NE(Output.find("clang version"), std::string::npos);
  EXPECT_NE(Output.find("-cc1"), std::string::npos);

  auto Interp = cantFail(clang::Interpreter::create(std::move(CI)));
  OS.flush();
  EXPECT_NE(Output.find("clang -cc1 version"), std::string::npos);
  EXPECT_NE(Output.find("#include \"...\" search starts here:"),
            std::string::npos);
  EXPECT_NE(Output.find("End of search list."), std::string::npos);
}

TEST(IncrementalCompilerBuilder, SetTargetTriple) {
// FIXME : This test doesn't current work for Emscripten builds.
// It should be possible to make it work.For details on how it fails and
// the current progress to enable this test see
// the following Github issue https: //
// github.com/llvm/llvm-project/issues/153461
#ifdef __EMSCRIPTEN__
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
  auto CB = clang::IncrementalCompilerBuilder();
  CB.SetTargetTriple("armv6-none-eabi");
  auto CI = cantFail(CB.CreateCpp());
  EXPECT_EQ(CI->getTargetOpts().Triple, "armv6-unknown-none-eabi");
  cleanupRemappedFileBuffers(*CI);
}

} // end anonymous namespace
