//===-- CompilerTests.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Compiler.h"
#include "TestTU.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using testing::IsEmpty;

TEST(BuildCompilerInvocation, DropsPCH) {
  MockFS FS;
  IgnoreDiagnostics Diags;
  TestTU TU;
  TU.AdditionalFiles["test.h.pch"] = "";

  TU.ExtraArgs = {"-include-pch", "test.h.pch"};
  EXPECT_THAT(buildCompilerInvocation(TU.inputs(FS), Diags)
                  ->getPreprocessorOpts()
                  .ImplicitPCHInclude,
              IsEmpty());

  // Transparent include translation
  TU.ExtraArgs = {"-include", "test.h"};
  EXPECT_THAT(buildCompilerInvocation(TU.inputs(FS), Diags)
                  ->getPreprocessorOpts()
                  .ImplicitPCHInclude,
              IsEmpty());

  // CL mode parsing.
  TU.AdditionalFiles["test.pch"] = "";
  TU.ExtraArgs = {"--driver-mode=cl"};
  TU.ExtraArgs.push_back("/Yutest.h");
  EXPECT_THAT(buildCompilerInvocation(TU.inputs(FS), Diags)
                  ->getPreprocessorOpts()
                  .ImplicitPCHInclude,
              IsEmpty());
  EXPECT_THAT(buildCompilerInvocation(TU.inputs(FS), Diags)
                  ->getPreprocessorOpts()
                  .PCHThroughHeader,
              IsEmpty());
}

TEST(BuildCompilerInvocation, PragmaDebugCrash) {
  TestTU TU = TestTU::withCode("#pragma clang __debug parser_crash");
  TU.build(); // no-crash
}

} // namespace
} // namespace clangd
} // namespace clang
