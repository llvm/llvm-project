//===-- CompilerTests.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Compiler.h"
#include "TestTU.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/FrontendOptions.h"
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

TEST(BuildCompilerInvocation, DropsShowIncludes) {
  MockFS FS;
  IgnoreDiagnostics Diags;
  TestTU TU;

  TU.ExtraArgs = {"-Xclang", "--show-includes"};
  EXPECT_THAT(buildCompilerInvocation(TU.inputs(FS), Diags)
                  ->getDependencyOutputOpts()
                  .ShowIncludesDest,
              ShowIncludesDestination::None);

  TU.ExtraArgs = {"/showIncludes", "--driver-mode=cl"};
  EXPECT_THAT(buildCompilerInvocation(TU.inputs(FS), Diags)
                  ->getDependencyOutputOpts()
                  .ShowIncludesDest,
              ShowIncludesDestination::None);

  TU.ExtraArgs = {"/showIncludes:user", "--driver-mode=cl"};
  EXPECT_THAT(buildCompilerInvocation(TU.inputs(FS), Diags)
                  ->getDependencyOutputOpts()
                  .ShowIncludesDest,
              ShowIncludesDestination::None);
}

TEST(BuildCompilerInvocation, DropsPlugins) {
  MockFS FS;
  IgnoreDiagnostics Diags;
  TestTU TU;

  TU.ExtraArgs = {"-Xclang", "-load",
                  "-Xclang", "plugins.so",
                  "-Xclang", "-plugin",
                  "-Xclang", "my_plugin",
                  "-Xclang", "-plugin-arg-my_plugin",
                  "-Xclang", "foo=bar",
                  "-Xclang", "-add-plugin",
                  "-Xclang", "my_plugin2"};
  auto Opts = buildCompilerInvocation(TU.inputs(FS), Diags)->getFrontendOpts();
  EXPECT_THAT(Opts.Plugins, IsEmpty());
  EXPECT_THAT(Opts.PluginArgs, IsEmpty());
  EXPECT_THAT(Opts.AddPluginActions, IsEmpty());
  EXPECT_EQ(Opts.ProgramAction, frontend::ActionKind::ParseSyntaxOnly);
  EXPECT_TRUE(Opts.ActionName.empty());
}
} // namespace
} // namespace clangd
} // namespace clang
