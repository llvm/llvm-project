//===- unittests/Frontend/UtilsTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace {
using testing::ElementsAre;

TEST(BuildCompilerInvocationTest, RecoverMultipleJobs) {
  // This generates multiple jobs and we recover by using the first.
  std::vector<const char *> Args = {"clang", "--target=macho", "-arch",  "i386",
                                    "-arch", "x86_64",         "foo.cpp"};
  clang::IgnoringDiagConsumer D;
  CreateInvocationOptions Opts;
  Opts.RecoverOnError = true;
  Opts.VFS = new llvm::vfs::InMemoryFileSystem();
  Opts.Diags = clang::CompilerInstance::createDiagnostics(
      *Opts.VFS, new DiagnosticOptions, &D, false);
  std::unique_ptr<CompilerInvocation> CI = createInvocation(Args, Opts);
  ASSERT_TRUE(CI);
  EXPECT_THAT(CI->getTargetOpts().Triple, testing::StartsWith("i386-"));
}

// buildInvocationFromCommandLine should not translate -include to -include-pch,
// even if the PCH file exists.
TEST(BuildCompilerInvocationTest, ProbePrecompiled) {
  std::vector<const char *> Args = {"clang", "-include", "foo.h", "foo.cpp"};
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FS->addFile("foo.h", 0, llvm::MemoryBuffer::getMemBuffer(""));
  FS->addFile("foo.h.pch", 0, llvm::MemoryBuffer::getMemBuffer(""));

  clang::IgnoringDiagConsumer D;
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> CommandLineDiagsEngine =
      clang::CompilerInstance::createDiagnostics(*FS, new DiagnosticOptions, &D,
                                                 false);
  // Default: ProbePrecompiled=false
  CreateInvocationOptions CIOpts;
  CIOpts.Diags = CommandLineDiagsEngine;
  CIOpts.VFS = FS;
  std::unique_ptr<CompilerInvocation> CI = createInvocation(Args, CIOpts);
  ASSERT_TRUE(CI);
  EXPECT_THAT(CI->getPreprocessorOpts().Includes, ElementsAre("foo.h"));
  EXPECT_EQ(CI->getPreprocessorOpts().ImplicitPCHInclude, "");

  CIOpts.ProbePrecompiled = true;
  CI = createInvocation(Args, CIOpts);
  ASSERT_TRUE(CI);
  EXPECT_THAT(CI->getPreprocessorOpts().Includes, ElementsAre());
  EXPECT_EQ(CI->getPreprocessorOpts().ImplicitPCHInclude, "foo.h.pch");
}

} // namespace
} // namespace clang
