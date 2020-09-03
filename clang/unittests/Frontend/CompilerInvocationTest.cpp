//===- unittests/Frontend/CompilerInvocationTest.cpp - CI tests //---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/Host.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

using ::testing::Contains;
using ::testing::Each;
using ::testing::StrEq;
using ::testing::StrNe;

namespace {

class CC1CommandLineGenerationTest : public ::testing::Test {
public:
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags;
  SmallVector<const char *, 32> GeneratedArgs;
  SmallVector<std::string, 32> GeneratedArgsStorage;

  const char *operator()(const Twine &Arg) {
    return GeneratedArgsStorage.emplace_back(Arg.str()).c_str();
  }

  CC1CommandLineGenerationTest()
      : Diags(CompilerInstance::createDiagnostics(new DiagnosticOptions())) {}
};

TEST_F(CC1CommandLineGenerationTest, CanGenerateCC1CommandLineFlag) {
  const char *Args[] = {"clang", "-xc++", "-fmodules-strict-context-hash", "-"};

  CompilerInvocation CInvok;
  CompilerInvocation::CreateFromArgs(CInvok, Args, *Diags);

  CInvok.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fmodules-strict-context-hash")));
}

TEST_F(CC1CommandLineGenerationTest, CanGenerateCC1CommandLineSeparate) {
  const char *TripleCStr = "i686-apple-darwin9";
  const char *Args[] = {"clang", "-xc++", "-triple", TripleCStr, "-"};

  CompilerInvocation CInvok;
  CompilerInvocation::CreateFromArgs(CInvok, Args, *Diags);

  CInvok.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq(TripleCStr)));
}

TEST_F(CC1CommandLineGenerationTest,
       CanGenerateCC1CommandLineSeparateRequiredPresent) {
  const std::string DefaultTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
  const char *Args[] = {"clang", "-xc++", "-triple", DefaultTriple.c_str(),
                        "-"};

  CompilerInvocation CInvok;
  CompilerInvocation::CreateFromArgs(CInvok, Args, *Diags);

  CInvok.generateCC1CommandLine(GeneratedArgs, *this);

  // Triple should always be emitted even if it is the default
  ASSERT_THAT(GeneratedArgs, Contains(StrEq(DefaultTriple.c_str())));
}

TEST_F(CC1CommandLineGenerationTest,
       CanGenerateCC1CommandLineSeparateRequiredAbsent) {
  const std::string DefaultTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
  const char *Args[] = {"clang", "-xc++", "-"};

  CompilerInvocation CInvok;
  CompilerInvocation::CreateFromArgs(CInvok, Args, *Diags);

  CInvok.generateCC1CommandLine(GeneratedArgs, *this);

  // Triple should always be emitted even if it is the default
  ASSERT_THAT(GeneratedArgs, Contains(StrEq(DefaultTriple.c_str())));
}

TEST_F(CC1CommandLineGenerationTest, CanGenerateCC1CommandLineSeparateEnum) {
  const char *RelocationModelCStr = "static";
  const char *Args[] = {"clang", "-xc++", "-mrelocation-model",
                        RelocationModelCStr, "-"};

  CompilerInvocation CInvok;
  CompilerInvocation::CreateFromArgs(CInvok, Args, *Diags);

  CInvok.generateCC1CommandLine(GeneratedArgs, *this);

  // Non default relocation model
  ASSERT_THAT(GeneratedArgs, Contains(StrEq(RelocationModelCStr)));
  GeneratedArgs.clear();

  RelocationModelCStr = "pic";
  Args[3] = RelocationModelCStr;

  CompilerInvocation CInvok1;
  CompilerInvocation::CreateFromArgs(CInvok1, Args, *Diags);

  CInvok1.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Each(StrNe(RelocationModelCStr)));
}

} // anonymous namespace
