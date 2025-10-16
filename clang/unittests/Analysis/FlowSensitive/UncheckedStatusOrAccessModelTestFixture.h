//===- UncheckedStatusOrAccessModelTestFixture.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_UNCHECKEDSTATUSORACCESSMODELTESTFIXTURE_H_
#define LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_UNCHECKEDSTATUSORACCESSMODELTESTFIXTURE_H_

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedStatusOrAccessModel.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::dataflow::statusor_model {

enum class UncheckedStatusOrAccessModelTestAliasKind {
  kUnaliased = 0,        // no alias
  kPartiallyAliased = 1, // template<typename T> using Alias = absl::StatusOr;
  kFullyAliased = 2,     // using Alias = absl::StatusOr<int>;
};

// Base class for the test executors. This is needed to abstract away the
// template parameter from the UncheckedStatusOrAccessModelTestExecutor. This
// allows us to use UncheckedStatusOrAccessModelTestExecutorBase* in the
// UncheckedStatusOrAccessModelTest.
class UncheckedStatusOrAccessModelTestExecutorBase {
public:
  virtual void
  ExpectDiagnosticsFor(std::string SourceCode,
                       UncheckedStatusOrAccessModelTestAliasKind) const = 0;
  virtual void ExpectDiagnosticsForLambda(
      std::string SourceCode,
      UncheckedStatusOrAccessModelTestAliasKind) const = 0;
  virtual ~UncheckedStatusOrAccessModelTestExecutorBase() = default;
};

// Returns these macros according to the alias kind:
//  - STATUS
//  - STATUSOR_INT
//  - STATUSOR_BOOL
//  - STATUSOR_VOIDPTR
// Tests should use these macros instead of e.g. absl::StatusOr<int> to ensure
// the model is insensitive to whether the StatusOr<> is aliased or not.
std::string GetAliasMacros(UncheckedStatusOrAccessModelTestAliasKind AliasKind);

std::vector<std::pair<std::string, std::string>>
GetHeaders(UncheckedStatusOrAccessModelTestAliasKind AliasKind);

// This allows us to run the same test suite for multiple models. This allows
// vendors to model internal APIs in an extension of the base model, and make
// sure that these tests still pass.
template <typename Model>
class UncheckedStatusOrAccessModelTestExecutor
    : public UncheckedStatusOrAccessModelTestExecutorBase {
public:
  void ExpectDiagnosticsFor(
      std::string SourceCode,
      UncheckedStatusOrAccessModelTestAliasKind AliasKind) const override {
    using namespace ::clang::ast_matchers; // NOLINT: Too many names
    ExpectDiagnosticsFor(SourceCode, hasName("target"), AliasKind);
  }

  void ExpectDiagnosticsForLambda(
      std::string SourceCode,
      UncheckedStatusOrAccessModelTestAliasKind AliasKind) const override {
    using namespace ::clang::ast_matchers; // NOLINT: Too many names
    ExpectDiagnosticsFor(SourceCode,
                         allOf(hasOverloadedOperatorName("()"),
                               hasDeclContext(cxxRecordDecl(isLambda()))),
                         AliasKind);
  }

  template <typename FuncDeclMatcher>
  void ExpectDiagnosticsFor(
      std::string SourceCode, FuncDeclMatcher FuncMatcher,
      UncheckedStatusOrAccessModelTestAliasKind AliasKind) const {
    std::vector<std::pair<std::string, std::string>> Headers =
        GetHeaders(AliasKind);

    UncheckedStatusOrAccessModelOptions Options{};
    std::vector<SourceLocation> Diagnostics;
    llvm::Error Error = test::checkDataflow<Model>(
        test::AnalysisInputs<Model>(
            SourceCode, std::move(FuncMatcher),
            [](ASTContext &Ctx, Environment &Env) { return Model(Ctx, Env); })
            .withPostVisitCFG(
                [&Diagnostics,
                 Diagnoser = UncheckedStatusOrAccessDiagnoser(Options)](
                    ASTContext &Ctx, const CFGElement &Elt,
                    const TransferStateForDiagnostics<
                        UncheckedStatusOrAccessModel::Lattice> &State) mutable {
                  auto EltDiagnostics = Diagnoser(Elt, Ctx, State);
                  llvm::move(EltDiagnostics, std::back_inserter(Diagnostics));
                })
            .withASTBuildArgs(
                {"-fsyntax-only", "-std=c++17", "-Wno-undefined-inline"})
            .withASTBuildVirtualMappedFiles(
                tooling::FileContentMappings(Headers.begin(), Headers.end())),
        /*VerifyResults=*/[&Diagnostics, SourceCode](
                              const llvm::DenseMap<unsigned, std::string>
                                  &Annotations,
                              const test::AnalysisOutputs &AO) {
          llvm::DenseSet<unsigned> AnnotationLines;
          for (const auto &[Line, _] : Annotations)
            AnnotationLines.insert(Line);
          auto &SrcMgr = AO.ASTCtx.getSourceManager();
          llvm::DenseSet<unsigned> DiagnosticLines;
          for (SourceLocation &Loc : Diagnostics)
            DiagnosticLines.insert(SrcMgr.getPresumedLineNumber(Loc));

          EXPECT_THAT(DiagnosticLines, testing::ContainerEq(AnnotationLines))
              << "\nFailing code:\n"
              << SourceCode;
        });
    if (Error)
      FAIL() << llvm::toString(std::move(Error));
  }
};

class UncheckedStatusOrAccessModelTest
    : public ::testing::TestWithParam<
          std::pair<UncheckedStatusOrAccessModelTestExecutorBase *,
                    UncheckedStatusOrAccessModelTestAliasKind>> {
protected:
  void ExpectDiagnosticsFor(std::string SourceCode) {
    GetParam().first->ExpectDiagnosticsFor(SourceCode, GetParam().second);
  }

  void ExpectDiagnosticsForLambda(std::string SourceCode) {
    GetParam().first->ExpectDiagnosticsForLambda(SourceCode, GetParam().second);
  }
};

} // namespace clang::dataflow::statusor_model

#endif // LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_UNCHECKEDSTATUSORACCESSMODELTESTFIXTURE_H_
