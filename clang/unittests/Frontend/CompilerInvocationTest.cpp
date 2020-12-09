//===- unittests/Frontend/CompilerInvocationTest.cpp - CI tests //---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/Support/Host.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

using ::testing::Contains;
using ::testing::StrEq;

namespace {
class CommandLineTest : public ::testing::Test {
public:
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags;
  SmallVector<const char *, 32> GeneratedArgs;
  SmallVector<std::string, 32> GeneratedArgsStorage;
  CompilerInvocation Invocation;

  const char *operator()(const Twine &Arg) {
    return GeneratedArgsStorage.emplace_back(Arg.str()).c_str();
  }

  CommandLineTest()
      : Diags(CompilerInstance::createDiagnostics(new DiagnosticOptions(),
                                                  new TextDiagnosticBuffer())) {
  }
};

// Boolean option with a keypath that defaults to true.
// The only flag with a negative spelling can set the keypath to false.

TEST_F(CommandLineTest, BoolOptionDefaultTrueSingleFlagNotPresent) {
  const char *Args[] = {""};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_TRUE(Invocation.getFrontendOpts().UseTemporary);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fno-temp-file"))));
}

TEST_F(CommandLineTest, BoolOptionDefaultTrueSingleFlagPresent) {
  const char *Args[] = {"-fno-temp-file"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_FALSE(Invocation.getFrontendOpts().UseTemporary);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fno-temp-file")));
}

TEST_F(CommandLineTest, BoolOptionDefaultTrueSingleFlagUnknownPresent) {
  const char *Args[] = {"-ftemp-file"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  // Driver-only flag.
  ASSERT_TRUE(Diags->hasErrorOccurred());
  ASSERT_TRUE(Invocation.getFrontendOpts().UseTemporary);
}

TEST_F(CommandLineTest, CanGenerateCC1CommandLineFlag) {
  const char *Args[] = {"-fmodules-strict-context-hash"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fmodules-strict-context-hash")));
}

TEST_F(CommandLineTest, CanGenerateCC1CommandLineSeparate) {
  const char *TripleCStr = "i686-apple-darwin9";
  const char *Args[] = {"-triple", TripleCStr};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq(TripleCStr)));
}

TEST_F(CommandLineTest,  CanGenerateCC1CommandLineSeparateRequiredPresent) {
  const std::string DefaultTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
  const char *Args[] = {"-triple", DefaultTriple.c_str()};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Triple should always be emitted even if it is the default
  ASSERT_THAT(GeneratedArgs, Contains(StrEq(DefaultTriple.c_str())));
}

TEST_F(CommandLineTest, CanGenerateCC1CommandLineSeparateRequiredAbsent) {
  const std::string DefaultTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
  const char *Args[] = {""};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Triple should always be emitted even if it is the default
  ASSERT_THAT(GeneratedArgs, Contains(StrEq(DefaultTriple.c_str())));
}

TEST_F(CommandLineTest, CanGenerateCC1CommandLineSeparateEnumNonDefault) {
  const char *Args[] = {"-mrelocation-model", "static"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Non default relocation model.
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("static")));
}

TEST_F(CommandLineTest, CanGenerateCC1COmmandLineSeparateEnumDefault) {
  const char *Args[] = {"-mrelocation-model", "pic"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Default relocation model.
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("pic"))));
}

// Tree of boolean options that can be (directly or transitively) implied by
// their parent:
//
//   * -cl-unsafe-math-optimizations
//     * -cl-mad-enable
//     * -menable-unsafe-fp-math
//       * -freciprocal-math

TEST_F(CommandLineTest, ImpliedBoolOptionsNoFlagPresent) {
  const char *Args[] = {""};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_FALSE(Invocation.getLangOpts()->CLUnsafeMath);
  ASSERT_FALSE(Invocation.getCodeGenOpts().LessPreciseFPMAD);
  ASSERT_FALSE(Invocation.getLangOpts()->UnsafeFPMath);
  ASSERT_FALSE(Invocation.getLangOpts()->AllowRecip);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Not generated - missing.
  ASSERT_THAT(GeneratedArgs,
              Not(Contains(StrEq("-cl-unsafe-math-optimizations"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-cl-mad-enable"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-menable-unsafe-fp-math"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-freciprocal-math"))));
}

TEST_F(CommandLineTest, ImpliedBoolOptionsRootFlagPresent) {
  const char *Args[] = {"-cl-unsafe-math-optimizations"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  // Explicitly provided root flag.
  ASSERT_TRUE(Invocation.getLangOpts()->CLUnsafeMath);
  // Directly implied by explicitly provided root flag.
  ASSERT_TRUE(Invocation.getCodeGenOpts().LessPreciseFPMAD);
  ASSERT_TRUE(Invocation.getLangOpts()->UnsafeFPMath);
  // Transitively implied by explicitly provided root flag.
  ASSERT_TRUE(Invocation.getLangOpts()->AllowRecip);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Generated - explicitly provided.
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-cl-unsafe-math-optimizations")));
  // Not generated - implied by the generated root flag.
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-cl-mad-enable"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-menable-unsafe-fp-math"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-freciprocal-math"))));
}

TEST_F(CommandLineTest, ImpliedBoolOptionsAllFlagsPresent) {
  const char *Args[] = {"-cl-unsafe-math-optimizations", "-cl-mad-enable",
                        "-menable-unsafe-fp-math", "-freciprocal-math"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_TRUE(Invocation.getLangOpts()->CLUnsafeMath);
  ASSERT_TRUE(Invocation.getCodeGenOpts().LessPreciseFPMAD);
  ASSERT_TRUE(Invocation.getLangOpts()->UnsafeFPMath);
  ASSERT_TRUE(Invocation.getLangOpts()->AllowRecip);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Generated - explicitly provided.
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-cl-unsafe-math-optimizations")));
  // Not generated - implied by their generated parent.
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-cl-mad-enable"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-menable-unsafe-fp-math"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-freciprocal-math"))));
}

TEST_F(CommandLineTest, ImpliedBoolOptionsImpliedFlagsPresent) {
  const char *Args[] = {"-cl-mad-enable", "-menable-unsafe-fp-math",
                        "-freciprocal-math"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);
  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_FALSE(Invocation.getLangOpts()->CLUnsafeMath);
  ASSERT_TRUE(Invocation.getCodeGenOpts().LessPreciseFPMAD);
  ASSERT_TRUE(Invocation.getLangOpts()->UnsafeFPMath);
  ASSERT_TRUE(Invocation.getLangOpts()->AllowRecip);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  // Not generated - missing.
  ASSERT_THAT(GeneratedArgs,
              Not(Contains(StrEq("-cl-unsafe-math-optimizations"))));
  // Generated - explicitly provided.
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-cl-mad-enable")));
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-menable-unsafe-fp-math")));
  // Not generated - implied by its generated parent.
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-freciprocal-math"))));
}

TEST_F(CommandLineTest, PresentAndNotImpliedGenerated) {
  const char *Args[] = {"-cl-mad-enable", "-menable-unsafe-fp-math"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Present options that were not implied are generated.
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-cl-mad-enable")));
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-menable-unsafe-fp-math")));
}
} // anonymous namespace
