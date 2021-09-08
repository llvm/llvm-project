//===- unittests/Frontend/CompilerInvocationTest.cpp - CI tests //---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Serialization/ModuleFileExtension.h"
#include "llvm/Support/Host.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

using ::testing::Contains;
using ::testing::HasSubstr;
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

template <typename M>
std::string describeContainsN(M InnerMatcher, unsigned N, bool Negation) {
  StringRef Contains = Negation ? "doesn't contain" : "contains";
  StringRef Instance = N == 1 ? " instance " : " instances ";
  StringRef Element = "of element that ";

  std::ostringstream Inner;
  InnerMatcher.impl().DescribeTo(&Inner);

  return (Contains + " exactly " + Twine(N) + Instance + Element + Inner.str())
      .str();
}

MATCHER_P2(ContainsN, InnerMatcher, N,
           describeContainsN(InnerMatcher, N, negation)) {
  auto InnerMatches = [this](const auto &Element) {
    ::testing::internal::DummyMatchResultListener InnerListener;
    return InnerMatcher.impl().MatchAndExplain(Element, &InnerListener);
  };

  return count_if(arg, InnerMatches) == N;
}

TEST(ContainsN, Empty) {
  const char *Array[] = {""};

  ASSERT_THAT(Array, ContainsN(StrEq("x"), 0));
  ASSERT_THAT(Array, Not(ContainsN(StrEq("x"), 1)));
  ASSERT_THAT(Array, Not(ContainsN(StrEq("x"), 2)));
}

TEST(ContainsN, Zero) {
  const char *Array[] = {"y"};

  ASSERT_THAT(Array, ContainsN(StrEq("x"), 0));
  ASSERT_THAT(Array, Not(ContainsN(StrEq("x"), 1)));
  ASSERT_THAT(Array, Not(ContainsN(StrEq("x"), 2)));
}

TEST(ContainsN, One) {
  const char *Array[] = {"a", "b", "x", "z"};

  ASSERT_THAT(Array, Not(ContainsN(StrEq("x"), 0)));
  ASSERT_THAT(Array, ContainsN(StrEq("x"), 1));
  ASSERT_THAT(Array, Not(ContainsN(StrEq("x"), 2)));
}

TEST(ContainsN, Two) {
  const char *Array[] = {"x", "a", "b", "x"};

  ASSERT_THAT(Array, Not(ContainsN(StrEq("x"), 0)));
  ASSERT_THAT(Array, Not(ContainsN(StrEq("x"), 1)));
  ASSERT_THAT(Array, ContainsN(StrEq("x"), 2));
}

// Copy constructor/assignment perform deep copy of reference-counted pointers.

TEST(CompilerInvocationTest, DeepCopyConstructor) {
  CompilerInvocation A;
  A.getAnalyzerOpts()->Config["Key"] = "Old";

  CompilerInvocation B(A);
  B.getAnalyzerOpts()->Config["Key"] = "New";

  ASSERT_EQ(A.getAnalyzerOpts()->Config["Key"], "Old");
}

TEST(CompilerInvocationTest, DeepCopyAssignment) {
  CompilerInvocation A;
  A.getAnalyzerOpts()->Config["Key"] = "Old";

  CompilerInvocation B;
  B = A;
  B.getAnalyzerOpts()->Config["Key"] = "New";

  ASSERT_EQ(A.getAnalyzerOpts()->Config["Key"], "Old");
}

// Boolean option with a keypath that defaults to true.
// The only flag with a negative spelling can set the keypath to false.

TEST_F(CommandLineTest, BoolOptionDefaultTrueSingleFlagNotPresent) {
  const char *Args[] = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getFrontendOpts().UseTemporary);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fno-temp-file"))));
}

TEST_F(CommandLineTest, BoolOptionDefaultTrueSingleFlagPresent) {
  const char *Args[] = {"-fno-temp-file"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getFrontendOpts().UseTemporary);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fno-temp-file")));
}

TEST_F(CommandLineTest, BoolOptionDefaultTrueSingleFlagUnknownPresent) {
  const char *Args[] = {"-ftemp-file"};

  // Driver-only flag.
  ASSERT_FALSE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getFrontendOpts().UseTemporary);
}

// Boolean option with a keypath that defaults to true.
// The flag with negative spelling can set the keypath to false.
// The flag with positive spelling can reset the keypath to true.

TEST_F(CommandLineTest, BoolOptionDefaultTruePresentNone) {
  const char *Args[] = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getCodeGenOpts().Autolink);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fautolink"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fno-autolink"))));
}

TEST_F(CommandLineTest, BoolOptionDefaultTruePresentNegChange) {
  const char *Args[] = {"-fno-autolink"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getCodeGenOpts().Autolink);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fno-autolink")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fautolink"))));
}

TEST_F(CommandLineTest, BoolOptionDefaultTruePresentPosReset) {
  const char *Args[] = {"-fautolink"};

  // Driver-only flag.
  ASSERT_FALSE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getCodeGenOpts().Autolink);
}

// Boolean option with a keypath that defaults to false.
// The flag with negative spelling can set the keypath to true.
// The flag with positive spelling can reset the keypath to false.

TEST_F(CommandLineTest, BoolOptionDefaultFalsePresentNone) {
  const char *Args[] = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getCodeGenOpts().NoInlineLineTables);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-ginline-line-tables"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-gno-inline-line-tables"))));
}

TEST_F(CommandLineTest, BoolOptionDefaultFalsePresentNegChange) {
  const char *Args[] = {"-gno-inline-line-tables"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getCodeGenOpts().NoInlineLineTables);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-gno-inline-line-tables")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-ginline-line-tables"))));
}

TEST_F(CommandLineTest, BoolOptionDefaultFalsePresentPosReset) {
  const char *Args[] = {"-ginline-line-tables"};

  // Driver-only flag.
  ASSERT_FALSE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getCodeGenOpts().NoInlineLineTables);
}

// Boolean option with a keypath that defaults to false.
// The flag with positive spelling can set the keypath to true.
// The flag with negative spelling can reset the keypath to false.

TEST_F(CommandLineTest, BoolOptionDefaultFalsePresentNoneX) {
  const char *Args[] = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getCodeGenOpts().CodeViewGHash);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-gcodeview-ghash"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-gno-codeview-ghash"))));
}

TEST_F(CommandLineTest, BoolOptionDefaultFalsePresentPosChange) {
  const char *Args[] = {"-gcodeview-ghash"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getCodeGenOpts().CodeViewGHash);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-gcodeview-ghash")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-gno-codeview-ghash"))));
}

TEST_F(CommandLineTest, BoolOptionDefaultFalsePresentNegReset) {
  const char *Args[] = {"-gno-codeview-ghash"};

  // Driver-only flag.
  ASSERT_FALSE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getCodeGenOpts().CodeViewGHash);
}

// Boolean option with a keypath that defaults to an arbitrary expression.
// The flag with positive spelling can set the keypath to true.
// The flag with negative spelling can set the keypath to false.

static constexpr unsigned PassManagerDefault =
    !static_cast<unsigned>(LLVM_ENABLE_NEW_PASS_MANAGER);

static constexpr const char *PassManagerResetByFlag =
    LLVM_ENABLE_NEW_PASS_MANAGER ? "-fno-legacy-pass-manager"
                                 : "-flegacy-pass-manager";

static constexpr const char *PassManagerChangedByFlag =
    LLVM_ENABLE_NEW_PASS_MANAGER ? "-flegacy-pass-manager"
                                 : "-fno-legacy-pass-manager";

TEST_F(CommandLineTest, BoolOptionDefaultArbitraryTwoFlagsPresentNone) {
  const char *Args = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getCodeGenOpts().LegacyPassManager, PassManagerDefault);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq(PassManagerResetByFlag))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq(PassManagerChangedByFlag))));
}

TEST_F(CommandLineTest, BoolOptionDefaultArbitraryTwoFlagsPresentChange) {
  const char *Args[] = {PassManagerChangedByFlag};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getCodeGenOpts().LegacyPassManager, !PassManagerDefault);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Contains(StrEq(PassManagerChangedByFlag)));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq(PassManagerResetByFlag))));
}

TEST_F(CommandLineTest, BoolOptionDefaultArbitraryTwoFlagsPresentReset) {
  const char *Args[] = {PassManagerResetByFlag};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getCodeGenOpts().LegacyPassManager, PassManagerDefault);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq(PassManagerResetByFlag))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq(PassManagerChangedByFlag))));
}

// Boolean option that gets the CC1Option flag from a let statement (which
// is applied **after** the record is defined):
//
//   let Flags = [CC1Option] in {
//     defm option : BoolOption<...>;
//   }

TEST_F(CommandLineTest, BoolOptionCC1ViaLetPresentNone) {
  const char *Args[] = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getCodeGenOpts().DebugPassManager);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fdebug-pass-manager"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fno-debug-pass-manager"))));
}

TEST_F(CommandLineTest, BoolOptionCC1ViaLetPresentPos) {
  const char *Args[] = {"-fdebug-pass-manager"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getCodeGenOpts().DebugPassManager);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, ContainsN(StrEq("-fdebug-pass-manager"), 1));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fno-debug-pass-manager"))));
}

TEST_F(CommandLineTest, BoolOptionCC1ViaLetPresentNeg) {
  const char *Args[] = {"-fno-debug-pass-manager"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getCodeGenOpts().DebugPassManager);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fno-debug-pass-manager"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fdebug-pass-manager"))));
}

TEST_F(CommandLineTest, CanGenerateCC1CommandLineFlag) {
  const char *Args[] = {"-fmodules-strict-context-hash"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fmodules-strict-context-hash")));
}

TEST_F(CommandLineTest, CanGenerateCC1CommandLineSeparate) {
  const char *TripleCStr = "i686-apple-darwin9";
  const char *Args[] = {"-triple", TripleCStr};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq(TripleCStr)));
}

TEST_F(CommandLineTest,  CanGenerateCC1CommandLineSeparateRequiredPresent) {
  const std::string DefaultTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
  const char *Args[] = {"-triple", DefaultTriple.c_str()};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Triple should always be emitted even if it is the default
  ASSERT_THAT(GeneratedArgs, Contains(StrEq(DefaultTriple.c_str())));
}

TEST_F(CommandLineTest, CanGenerateCC1CommandLineSeparateRequiredAbsent) {
  const std::string DefaultTriple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
  const char *Args[] = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Triple should always be emitted even if it is the default
  ASSERT_THAT(GeneratedArgs, Contains(StrEq(DefaultTriple.c_str())));
}

TEST_F(CommandLineTest, SeparateEnumNonDefault) {
  const char *Args[] = {"-mrelocation-model", "static"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getCodeGenOpts().RelocationModel, Reloc::Model::Static);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Non default relocation model.
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-mrelocation-model")));
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("static")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-mrelocation-model=static"))));
}

TEST_F(CommandLineTest, SeparateEnumDefault) {
  const char *Args[] = {"-mrelocation-model", "pic"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getCodeGenOpts().RelocationModel, Reloc::Model::PIC_);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Default relocation model.
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-mrelocation-model"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("pic"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-mrelocation-model=pic"))));
}

TEST_F(CommandLineTest, JoinedEnumNonDefault) {
  const char *Args[] = {"-fobjc-dispatch-method=non-legacy"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getCodeGenOpts().getObjCDispatchMethod(),
            CodeGenOptions::NonLegacy);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs,
              Contains(StrEq("-fobjc-dispatch-method=non-legacy")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fobjc-dispatch-method="))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("non-legacy"))));
}

TEST_F(CommandLineTest, JoinedEnumDefault) {
  const char *Args[] = {"-fobjc-dispatch-method=legacy"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getCodeGenOpts().getObjCDispatchMethod(),
            CodeGenOptions::Legacy);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs,
              Not(Contains(StrEq("-fobjc-dispatch-method=legacy"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fobjc-dispatch-method="))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("legacy"))));
}

TEST_F(CommandLineTest, StringVectorEmpty) {
  const char *Args[] = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getFrontendOpts().ModuleMapFiles.empty());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Not(Contains(HasSubstr("-fmodule-map-file"))));
}

TEST_F(CommandLineTest, StringVectorSingle) {
  const char *Args[] = {"-fmodule-map-file=a"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getFrontendOpts().ModuleMapFiles,
            std::vector<std::string>({"a"}));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, ContainsN(StrEq("-fmodule-map-file=a"), 1));
  ASSERT_THAT(GeneratedArgs, ContainsN(HasSubstr("-fmodule-map-file"), 1));
}

TEST_F(CommandLineTest, StringVectorMultiple) {
  const char *Args[] = {"-fmodule-map-file=a", "-fmodule-map-file=b"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getFrontendOpts().ModuleMapFiles ==
              std::vector<std::string>({"a", "b"}));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, ContainsN(StrEq("-fmodule-map-file=a"), 1));
  ASSERT_THAT(GeneratedArgs, ContainsN(StrEq("-fmodule-map-file=b"), 1));
  ASSERT_THAT(GeneratedArgs, ContainsN(HasSubstr("-fmodule-map-file"), 2));
}

// CommaJoined option with MarshallingInfoStringVector.

TEST_F(CommandLineTest, StringVectorCommaJoinedNone) {
  const char *Args[] = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getLangOpts()->CommentOpts.BlockCommandNames.empty());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs,
              Not(Contains(HasSubstr("-fcomment-block-commands"))));
}

TEST_F(CommandLineTest, StringVectorCommaJoinedSingle) {
  const char *Args[] = {"-fcomment-block-commands=x,y"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getLangOpts()->CommentOpts.BlockCommandNames,
            std::vector<std::string>({"x", "y"}));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs,
              ContainsN(StrEq("-fcomment-block-commands=x,y"), 1));
}

TEST_F(CommandLineTest, StringVectorCommaJoinedMultiple) {
  const char *Args[] = {"-fcomment-block-commands=x,y",
                        "-fcomment-block-commands=a,b"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_EQ(Invocation.getLangOpts()->CommentOpts.BlockCommandNames,
            std::vector<std::string>({"x", "y", "a", "b"}));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs,
              ContainsN(StrEq("-fcomment-block-commands=x,y,a,b"), 1));
}

// A flag that should be parsed only if a condition is met.

TEST_F(CommandLineTest, ConditionalParsingIfFalseFlagNotPresent) {
  const char *Args[] = {""};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_FALSE(Invocation.getLangOpts()->SYCLIsDevice);
  ASSERT_FALSE(Invocation.getLangOpts()->SYCLIsHost);
  ASSERT_EQ(Invocation.getLangOpts()->getSYCLVersion(), LangOptions::SYCL_None);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fsycl"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(HasSubstr("-sycl-std="))));
}

TEST_F(CommandLineTest, ConditionalParsingIfFalseFlagPresent) {
  const char *Args[] = {"-sycl-std=2017"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_FALSE(Invocation.getLangOpts()->SYCLIsDevice);
  ASSERT_FALSE(Invocation.getLangOpts()->SYCLIsHost);
  ASSERT_EQ(Invocation.getLangOpts()->getSYCLVersion(), LangOptions::SYCL_None);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fsycl-is-device"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fsycl-is-host"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(HasSubstr("-sycl-std="))));
}

TEST_F(CommandLineTest, ConditionalParsingIfNonsenseSyclStdArg) {
  const char *Args[] = {"-fsycl-is-device", "-sycl-std=garbage"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_TRUE(Diags->hasErrorOccurred());
  ASSERT_TRUE(Invocation.getLangOpts()->SYCLIsDevice);
  ASSERT_FALSE(Invocation.getLangOpts()->SYCLIsHost);
  ASSERT_EQ(Invocation.getLangOpts()->getSYCLVersion(), LangOptions::SYCL_None);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fsycl-is-device")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fsycl-is-host"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(HasSubstr("-sycl-std="))));
}

TEST_F(CommandLineTest, ConditionalParsingIfOddSyclStdArg1) {
  const char *Args[] = {"-fsycl-is-device", "-sycl-std=121"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_TRUE(Invocation.getLangOpts()->SYCLIsDevice);
  ASSERT_FALSE(Invocation.getLangOpts()->SYCLIsHost);
  ASSERT_EQ(Invocation.getLangOpts()->getSYCLVersion(), LangOptions::SYCL_2017);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fsycl-is-device")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fsycl-is-host"))));
  ASSERT_THAT(GeneratedArgs, Contains(HasSubstr("-sycl-std=2017")));
}

TEST_F(CommandLineTest, ConditionalParsingIfOddSyclStdArg2) {
  const char *Args[] = {"-fsycl-is-device", "-sycl-std=1.2.1"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_TRUE(Invocation.getLangOpts()->SYCLIsDevice);
  ASSERT_FALSE(Invocation.getLangOpts()->SYCLIsHost);
  ASSERT_EQ(Invocation.getLangOpts()->getSYCLVersion(), LangOptions::SYCL_2017);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fsycl-is-device")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fsycl-is-host"))));
  ASSERT_THAT(GeneratedArgs, Contains(HasSubstr("-sycl-std=2017")));
}

TEST_F(CommandLineTest, ConditionalParsingIfOddSyclStdArg3) {
  const char *Args[] = {"-fsycl-is-device", "-sycl-std=sycl-1.2.1"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_TRUE(Invocation.getLangOpts()->SYCLIsDevice);
  ASSERT_FALSE(Invocation.getLangOpts()->SYCLIsHost);
  ASSERT_EQ(Invocation.getLangOpts()->getSYCLVersion(), LangOptions::SYCL_2017);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fsycl-is-device")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fsycl-is-host"))));
  ASSERT_THAT(GeneratedArgs, Contains(HasSubstr("-sycl-std=2017")));
}

TEST_F(CommandLineTest, ConditionalParsingIfTrueFlagNotPresentHost) {
  const char *Args[] = {"-fsycl-is-host"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_EQ(Invocation.getLangOpts()->getSYCLVersion(),
            LangOptions::SYCL_Default);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fsycl-is-host")));
  ASSERT_THAT(GeneratedArgs, Contains(HasSubstr("-sycl-std=")));
}

TEST_F(CommandLineTest, ConditionalParsingIfTrueFlagNotPresentDevice) {
  const char *Args[] = {"-fsycl-is-device"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_EQ(Invocation.getLangOpts()->getSYCLVersion(),
            LangOptions::SYCL_Default);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fsycl-is-device")));
  ASSERT_THAT(GeneratedArgs, Contains(HasSubstr("-sycl-std=")));
}

TEST_F(CommandLineTest, ConditionalParsingIfTrueFlagPresent) {
  const char *Args[] = {"-fsycl-is-device", "-sycl-std=2017"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_EQ(Invocation.getLangOpts()->getSYCLVersion(), LangOptions::SYCL_2017);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fsycl-is-device")));
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-sycl-std=2017")));
}

// Wide integer option.

TEST_F(CommandLineTest, WideIntegerHighValue) {
  const char *Args[] = {"-fbuild-session-timestamp=1609827494445723662"};

  CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags);

  ASSERT_FALSE(Diags->hasErrorOccurred());
  ASSERT_EQ(Invocation.getHeaderSearchOpts().BuildSessionTimestamp,
            1609827494445723662ull);
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

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
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

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
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

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
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

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
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

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  // Present options that were not implied are generated.
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-cl-mad-enable")));
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-menable-unsafe-fp-math")));
}

// Diagnostic option.

TEST_F(CommandLineTest, DiagnosticOptionPresent) {
  const char *Args[] = {"-verify=xyz"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));

  ASSERT_EQ(Invocation.getDiagnosticOpts().VerifyPrefixes,
            std::vector<std::string>({"xyz"}));

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs, ContainsN(StrEq("-verify=xyz"), 1));
}

// Option default depends on language standard.

TEST_F(CommandLineTest, DigraphsImplied) {
  const char *Args[] = {""};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getLangOpts()->Digraphs);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fno-digraphs"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fdigraphs"))));
}

TEST_F(CommandLineTest, DigraphsDisabled) {
  const char *Args[] = {"-fno-digraphs"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getLangOpts()->Digraphs);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fno-digraphs")));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fdigraphs"))));
}

TEST_F(CommandLineTest, DigraphsNotImplied) {
  const char *Args[] = {"-std=c89"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_FALSE(Invocation.getLangOpts()->Digraphs);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fno-digraphs"))));
  ASSERT_THAT(GeneratedArgs, Not(Contains(StrEq("-fdigraphs"))));
}

TEST_F(CommandLineTest, DigraphsEnabled) {
  const char *Args[] = {"-std=c89", "-fdigraphs"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_TRUE(Invocation.getLangOpts()->Digraphs);

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);
  ASSERT_THAT(GeneratedArgs, Contains(StrEq("-fdigraphs")));
}

struct DummyModuleFileExtension
    : public llvm::RTTIExtends<DummyModuleFileExtension, ModuleFileExtension> {
  static char ID;

  ModuleFileExtensionMetadata getExtensionMetadata() const override {
    return {};
  };

  void hashExtension(ExtensionHashBuilder &HBuilder) const override {}

  std::unique_ptr<ModuleFileExtensionWriter>
  createExtensionWriter(ASTWriter &Writer) override {
    return {};
  }

  std::unique_ptr<ModuleFileExtensionReader>
  createExtensionReader(const ModuleFileExtensionMetadata &Metadata,
                        ASTReader &Reader, serialization::ModuleFile &Mod,
                        const llvm::BitstreamCursor &Stream) override {
    return {};
  }
};

char DummyModuleFileExtension::ID = 0;

TEST_F(CommandLineTest, TestModuleFileExtension) {
  const char *Args[] = {"-ftest-module-file-extension=first:2:1:0:first",
                        "-ftest-module-file-extension=second:3:2:1:second"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
  ASSERT_THAT(Invocation.getFrontendOpts().ModuleFileExtensions.size(), 2);

  // Exercise the check that only serializes instances of
  // TestModuleFileExtension by providing an instance of another
  // ModuleFileExtension subclass.
  Invocation.getFrontendOpts().ModuleFileExtensions.push_back(
      std::make_shared<DummyModuleFileExtension>());

  Invocation.generateCC1CommandLine(GeneratedArgs, *this);

  ASSERT_THAT(GeneratedArgs,
              ContainsN(HasSubstr("-ftest-module-file-extension="), 2));
  ASSERT_THAT(
      GeneratedArgs,
      Contains(StrEq("-ftest-module-file-extension=first:2:1:0:first")));
  ASSERT_THAT(
      GeneratedArgs,
      Contains(StrEq("-ftest-module-file-extension=second:3:2:1:second")));
}

TEST_F(CommandLineTest, RoundTrip) {
  // Testing one marshalled and one manually generated option from each
  // CompilerInvocation member.
  const char *Args[] = {
      "-round-trip-args",
      // LanguageOptions
      "-std=c17",
      "-fmax-tokens=10",
      // TargetOptions
      "-target-sdk-version=1.2.3",
      "-meabi",
      "4",
      // DiagnosticOptions
      "-Wundef-prefix=XY",
      "-fdiagnostics-format",
      "clang",
      // HeaderSearchOptions
      "-stdlib=libc++",
      "-fimplicit-module-maps",
      // PreprocessorOptions
      "-DXY=AB",
      "-include-pch",
      "a.pch",
      // AnalyzerOptions
      "-analyzer-config",
      "ctu-import-threshold=42",
      "-unoptimized-cfg",
      // MigratorOptions (no manually handled arguments)
      "-no-ns-alloc-error",
      // CodeGenOptions
      "-debug-info-kind=limited",
      "-debug-info-macro",
      // DependencyOutputOptions
      "--show-includes",
      "-H",
      // FileSystemOptions (no manually handled arguments)
      "-working-directory",
      "folder",
      // FrontendOptions
      "-load",
      "plugin",
      "-ast-merge",
      // PreprocessorOutputOptions
      "-dD",
      "-CC",
  };

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));

  ASSERT_TRUE(Invocation.getLangOpts()->C17);
  ASSERT_EQ(Invocation.getLangOpts()->MaxTokens, 10u);

  ASSERT_EQ(Invocation.getTargetOpts().SDKVersion, llvm::VersionTuple(1, 2, 3));
  ASSERT_EQ(Invocation.getTargetOpts().EABIVersion, EABI::EABI4);

  ASSERT_THAT(Invocation.getDiagnosticOpts().UndefPrefixes,
              Contains(StrEq("XY")));
  ASSERT_EQ(Invocation.getDiagnosticOpts().getFormat(),
            TextDiagnosticFormat::Clang);

  ASSERT_TRUE(Invocation.getHeaderSearchOpts().UseLibcxx);
  ASSERT_TRUE(Invocation.getHeaderSearchOpts().ImplicitModuleMaps);

  ASSERT_THAT(Invocation.getPreprocessorOpts().Macros,
              Contains(std::make_pair(std::string("XY=AB"), false)));
  ASSERT_EQ(Invocation.getPreprocessorOpts().ImplicitPCHInclude, "a.pch");

  ASSERT_EQ(Invocation.getAnalyzerOpts()->Config["ctu-import-threshold"], "42");
  ASSERT_TRUE(Invocation.getAnalyzerOpts()->UnoptimizedCFG);

  ASSERT_TRUE(Invocation.getMigratorOpts().NoNSAllocReallocError);

  ASSERT_EQ(Invocation.getCodeGenOpts().getDebugInfo(),
            codegenoptions::DebugInfoKind::LimitedDebugInfo);
  ASSERT_TRUE(Invocation.getCodeGenOpts().MacroDebugInfo);

  ASSERT_EQ(Invocation.getDependencyOutputOpts().ShowIncludesDest,
            ShowIncludesDestination::Stdout);
  ASSERT_TRUE(Invocation.getDependencyOutputOpts().ShowHeaderIncludes);
}

TEST_F(CommandLineTest, PluginArgsRoundTripDeterminism) {
  const char *Args[] = {
      "-plugin-arg-blink-gc-plugin", "no-members-in-stack-allocated",
      "-plugin-arg-find-bad-constructs", "checked-ptr-as-trivial-member",
      "-plugin-arg-find-bad-constructs", "check-ipc",
      // Enable round-trip to ensure '-plugin-arg' generation is deterministic.
      "-round-trip-args"};

  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, Args, *Diags));
}
} // anonymous namespace
