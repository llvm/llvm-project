//===- ChromiumCheckModelTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// FIXME: Move this to clang/unittests/Analysis/FlowSensitive/Models.

#include "clang/Analysis/FlowSensitive/Models/ChromiumCheckModel.h"
#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

using namespace clang;
using namespace dataflow;
using namespace test;

namespace {
using ::testing::NotNull;
using ::testing::UnorderedElementsAre;

static constexpr char ChromiumCheckHeader[] = R"(
namespace std {
class ostream;
} // namespace std

namespace logging {
class VoidifyStream {
 public:
  VoidifyStream() = default;
  void operator&(std::ostream&) {}
};

class CheckError {
 public:
  static CheckError Check(const char* file, int line, const char* condition);
  static CheckError DCheck(const char* file, int line, const char* condition);
  static CheckError PCheck(const char* file, int line, const char* condition);
  static CheckError PCheck(const char* file, int line);
  static CheckError DPCheck(const char* file, int line, const char* condition);

  std::ostream& stream();

  ~CheckError();

  CheckError(const CheckError& other) = delete;
  CheckError& operator=(const CheckError& other) = delete;
  CheckError(CheckError&& other) = default;
  CheckError& operator=(CheckError&& other) = default;
};

} // namespace logging

#define LAZY_CHECK_STREAM(stream, condition) \
  !(condition) ? (void)0 : ::logging::VoidifyStream() & (stream)

#define CHECK(condition)                                                     \
  LAZY_CHECK_STREAM(                                                         \
      ::logging::CheckError::Check(__FILE__, __LINE__, #condition).stream(), \
      !(condition))

#define PCHECK(condition)                                                     \
  LAZY_CHECK_STREAM(                                                          \
      ::logging::CheckError::PCheck(__FILE__, __LINE__, #condition).stream(), \
      !(condition))

#define DCHECK(condition)                                                     \
  LAZY_CHECK_STREAM(                                                          \
      ::logging::CheckError::DCheck(__FILE__, __LINE__, #condition).stream(), \
      !(condition))

#define DPCHECK(condition)                                                     \
  LAZY_CHECK_STREAM(                                                           \
      ::logging::CheckError::DPCheck(__FILE__, __LINE__, #condition).stream(), \
      !(condition))
)";

// A definition of the `CheckError` class that looks like the Chromium one, but
// is actually something else.
static constexpr char OtherCheckHeader[] = R"(
namespace other {
namespace logging {
class CheckError {
 public:
  static CheckError Check(const char* file, int line, const char* condition);
};
} // namespace logging
} // namespace other
)";

/// Replaces all occurrences of `Pattern` in `S` with `Replacement`.
std::string ReplacePattern(std::string S, const std::string &Pattern,
                           const std::string &Replacement) {
  size_t Pos = 0;
  Pos = S.find(Pattern, Pos);
  if (Pos != std::string::npos)
    S.replace(Pos, Pattern.size(), Replacement);
  return S;
}

template <typename Model>
class ModelAdaptorAnalysis
    : public DataflowAnalysis<ModelAdaptorAnalysis<Model>, NoopLattice> {
public:
  explicit ModelAdaptorAnalysis(ASTContext &Context)
      : DataflowAnalysis<ModelAdaptorAnalysis, NoopLattice>(Context) {}

  static NoopLattice initialElement() { return NoopLattice(); }

  void transfer(const CFGElement &E, NoopLattice &, Environment &Env) {
    M.transfer(E, Env);
  }

private:
  Model M;
};

template <typename Matcher>
void runDataflow(llvm::StringRef Code, Matcher Match) {
  const tooling::FileContentMappings FileContents = {
      {"check.h", ChromiumCheckHeader}, {"othercheck.h", OtherCheckHeader}};

  ASSERT_THAT_ERROR(
      checkDataflow<ModelAdaptorAnalysis<ChromiumCheckModel>>(
          AnalysisInputs<ModelAdaptorAnalysis<ChromiumCheckModel>>(
              Code, ast_matchers::hasName("target"),
              [](ASTContext &C, Environment &) {
                return ModelAdaptorAnalysis<ChromiumCheckModel>(C);
              })
              .withASTBuildArgs({"-fsyntax-only",
                                 "-fno-delayed-template-parsing", "-std=c++17"})
              .withASTBuildVirtualMappedFiles(std::move(FileContents)),
          /*VerifyResults=*/
          [&Match](const llvm::StringMap<DataflowAnalysisState<NoopLattice>>
                       &Results,
                   const AnalysisOutputs &AO) { Match(Results, AO.ASTCtx); }),
      llvm::Succeeded());
}

TEST(ChromiumCheckModelTest, CheckSuccessImpliesConditionHolds) {
  auto Expectations =
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto *FooVal = cast<BoolValue>(Env.getValue(*FooDecl));

        EXPECT_TRUE(Env.flowConditionImplies(FooVal->formula()));
      };

  std::string Code = R"(
    #include "check.h"

    void target(bool Foo) {
      $check(Foo);
      bool X = true;
      (void)X;
      // [[p]]
    }
  )";
  runDataflow(ReplacePattern(Code, "$check", "CHECK"), Expectations);
  runDataflow(ReplacePattern(Code, "$check", "DCHECK"), Expectations);
  runDataflow(ReplacePattern(Code, "$check", "PCHECK"), Expectations);
  runDataflow(ReplacePattern(Code, "$check", "DPCHECK"), Expectations);
}

TEST(ChromiumCheckModelTest, UnrelatedCheckIgnored) {
  auto Expectations =
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto *FooVal = cast<BoolValue>(Env.getValue(*FooDecl));

        EXPECT_FALSE(Env.flowConditionImplies(FooVal->formula()));
      };

  std::string Code = R"(
    #include "othercheck.h"

    void target(bool Foo) {
      if (!Foo) {
        (void)other::logging::CheckError::Check(__FILE__, __LINE__, "Foo");
      }
      bool X = true;
      (void)X;
      // [[p]]
    }
  )";
  runDataflow(Code, Expectations);
}
} // namespace
