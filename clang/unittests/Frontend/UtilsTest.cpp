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
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace {

TEST(BuildCompilerInvocationTest, RecoverMultipleJobs) {
  // This generates multiple jobs and we recover by using the first.
  std::vector<const char *> Args = {"clang", "--target=macho", "-arch",  "i386",
                                    "-arch", "x86_64",         "foo.cpp"};
  clang::IgnoringDiagConsumer D;
  CreateInvocationOptions Opts;
  Opts.RecoverOnError = true;
  Opts.Diags = clang::CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                                          &D, false);
  Opts.VFS = new llvm::vfs::InMemoryFileSystem();
  std::unique_ptr<CompilerInvocation> CI = createInvocation(Args, Opts);
  ASSERT_TRUE(CI);
  EXPECT_THAT(CI->TargetOpts->Triple, testing::StartsWith("i386-"));
}

} // namespace
} // namespace clang
