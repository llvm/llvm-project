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
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

TEST(IncrementalCompilerBuilder, SetCompilerArgs) {
  std::vector<const char *> ClangArgv = {"-Xclang", "-ast-dump-all"};
  auto CB = clang::IncrementalCompilerBuilder();
  CB.SetCompilerArgs(ClangArgv);
  auto CI = cantFail(CB.CreateCpp());
  EXPECT_TRUE(CI->getFrontendOpts().ASTDumpAll);
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
}

} // end anonymous namespace
