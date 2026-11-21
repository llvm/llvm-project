//===- UncheckedExpectedAccessModelTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/UncheckedExpectedAccessModel.h"
#include "MockHeaders.h"
#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <utility>
#include <vector>

namespace clang::dataflow::test {

using ::testing::ContainerEq;

/// Replaces all occurrences of `Pattern` in `S` with `Replacement`.
static void ReplaceAllOccurrences(std::string &S, const std::string &Pattern,
                                  const std::string &Replacement) {
  size_t Pos = 0;
  while (true) {
    Pos = S.find(Pattern, Pos);
    if (Pos == std::string::npos)
      break;
    S.replace(Pos, Pattern.size(), Replacement);
  }
}

struct ExpectedTypeIdentifier {
  std::string NamespaceName;
};

static raw_ostream &operator<<(raw_ostream &OS,
                               const ExpectedTypeIdentifier &TypeId) {
  OS << TypeId.NamespaceName << "::expected";
  return OS;
}

class UncheckedExpectedAccessTest
    : public ::testing::TestWithParam<ExpectedTypeIdentifier> {
protected:
  // Check that after running the analysis on SourceCode, it produces the
  // expected diagnostics according to [[unsafe]] annotations.
  // - No annotations => no diagnostics.
  // - Given "// [[unsafe]]" annotations on a line, we expect a diagnostic for
  //   an unchecked value access on that line.
  // - Given "// [[unsafe:error]]" annotations on a line, we expect a diagnostic
  //   for an unchecked error access on that line.
  void ExpectDiagnosticsFor(std::string SourceCode) {
    // `tl::expected` is usable since C++17, and `std::expected` since C++23.
    for (const char *CxxMode : {"-std=c++17", "-std=c++23"})
      ExpectDiagnosticsFor(SourceCode, CxxMode);
  }

  void ExpectDiagnosticsFor(std::string SourceCode, const char *CxxMode) {
    ReplaceAllOccurrences(SourceCode, "$ns", GetParam().NamespaceName);

    auto Headers = getMockHeaders();
    Headers.emplace_back("unchecked_expected_access_test.h", R"(
      #include "std_expected.h"
      #include "std_utility.h"
      #include "tl_expected.h"

      template <typename T>
      T Make();
    )");
    std::vector<UncheckedExpectedAccessDiagnostic> Diagnostics;
    llvm::Error Error = checkDataflow<UncheckedExpectedAccessModel>(
        AnalysisInputs<UncheckedExpectedAccessModel>(
            SourceCode, ast_matchers::hasName("target"),
            [](ASTContext &Ctx, Environment &Env) {
              return UncheckedExpectedAccessModel(Ctx, Env);
            })
            .withDiagnosisCallbacks(
                {/*Before=*/[&Diagnostics,
                             Diagnoser = UncheckedExpectedAccessDiagnoser()](
                                ASTContext &Ctx, const CFGElement &Elt,
                                const TransferStateForDiagnostics<NoopLattice>
                                    &State) mutable {
                   auto EltDiagnostics = Diagnoser(Elt, Ctx, State);
                   llvm::move(EltDiagnostics, std::back_inserter(Diagnostics));
                 },
                 /*After=*/nullptr})
            .withASTBuildArgs(
                {"-fsyntax-only", CxxMode, "-Wno-undefined-inline"})
            .withASTBuildVirtualMappedFiles(
                tooling::FileContentMappings(Headers.begin(), Headers.end())),
        /*VerifyResults=*/[&Diagnostics,
                           TypeName = GetParam().NamespaceName + "::expected"](
                              const llvm::DenseMap<unsigned, std::string>
                                  &Annotations,
                              const AnalysisOutputs &AO) {
          llvm::DenseSet<unsigned> AnnotationLines;
          for (const auto &[Line, _] : Annotations)
            AnnotationLines.insert(Line);
          auto &SrcMgr = AO.ASTCtx.getSourceManager();
          llvm::DenseSet<unsigned> DiagnosticLines;
          for (const UncheckedExpectedAccessDiagnostic &Diag : Diagnostics) {
            unsigned Line = SrcMgr.getPresumedLineNumber(Diag.Loc);
            DiagnosticLines.insert(Line);
            auto It = Annotations.find(Line);
            if (It == Annotations.end()) {
              DiagnosticOptions DiagOpts;
              TextDiagnostic TD(llvm::errs(), AO.ASTCtx.getLangOpts(),
                                DiagOpts);
              TD.emitDiagnostic(FullSourceLoc(Diag.Loc, SrcMgr),
                                DiagnosticsEngine::Error,
                                "unexpected diagnostic", {}, {});
            } else {
              UncheckedExpectedAccessKind ExpectedKind =
                  It->second == "unsafe:error"
                      ? UncheckedExpectedAccessKind::Error
                      : UncheckedExpectedAccessKind::Value;
              EXPECT_EQ(Diag.Kind, ExpectedKind) << "on line " << Line;
              EXPECT_EQ(Diag.TypeName, TypeName) << "on line " << Line;
            }
          }

          EXPECT_THAT(DiagnosticLines, ContainerEq(AnnotationLines));
        });
    if (Error)
      FAIL() << llvm::toString(std::move(Error));
  }
};

INSTANTIATE_TEST_SUITE_P(
    UncheckedExpectedUseTestInst, UncheckedExpectedAccessTest,
    ::testing::Values(ExpectedTypeIdentifier{"std"},
                      ExpectedTypeIdentifier{"tl"}),
    [](const ::testing::TestParamInfo<ExpectedTypeIdentifier> &Info) {
      return Info.param.NamespaceName;
    });

// Verifies that similarly-named types are ignored.
TEST_P(UncheckedExpectedAccessTest, NonTrackedExpectedType) {
  ExpectDiagnosticsFor(
      R"(
    namespace other {
    namespace $ns {
    template <typename T, typename E>
    struct expected {
      T value();
      E error();
    };
    }

    void target($ns::expected<int, int> e) {
      e.value();
      e.error();
    }
    }
  )");
}

// Classes derived from an expected are deliberately not analyzed.
TEST_P(UncheckedExpectedAccessTest, DerivedClassIsNotAnalyzed) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    struct Derived : $ns::expected<int, int> {};

    void target(Derived d) {
      d.value();
      *d;
      d.error();
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, TypeAlias) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    using MyExpected = $ns::expected<int, int>;
    typedef $ns::expected<int, int> MyExpectedTypedef;
    template <typename T>
    using MyExpectedT = $ns::expected<T, int>;

    void target(MyExpected e1, MyExpectedTypedef e2, MyExpectedT<int> e3) {
      e1.value(); // [[unsafe]]
      *e2; // [[unsafe]]
      e3.error(); // [[unsafe:error]]
      if (e1)
        e1.value();
      if (e2.has_value())
        *e2;
      if (!e3)
        e3.error();
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, EmptyFunctionBody) {
  ExpectDiagnosticsFor(R"(
    void target() {
      (void)0;
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapUsingValueNoCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      e.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target(const $ns::expected<int, int> &e) {
      e.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapUsingOperatorStarNoCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      *e; // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target(const $ns::expected<int, int> &e) {
      *e; // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapUsingOperatorArrowNoCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    struct Foo {
      void foo() const {}
    };

    void target($ns::expected<Foo, int> e) {
      e->foo(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    struct Foo {
      void foo() const {}
    };

    void target(const $ns::expected<Foo, int> &e) {
      e->foo(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapFunctionCallResultNoCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target() {
      auto e = Make<$ns::expected<int, int>>();
      e.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapVoidUsingValueNoCheck) {
  // `tl::expected<void, E>` has no `value()`.
  if (GetParam().NamespaceName != "std")
    GTEST_SKIP();

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<void, int> e) {
      e.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapInElseBranch) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (e.has_value()) {
      } else {
        e.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapInNegatedCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (!e) {
        *e; // [[unsafe]]
      }
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    int target($ns::expected<int, int> e) {
      return !e ? //
             *e : 0; // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, HasValueCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    struct Foo {
      void foo() const {}
    };

    void target($ns::expected<Foo, int> e) {
      if (e.has_value()) {
        e.value();
        *e;
        e->foo();
      }
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, OperatorBoolCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (e) {
        e.value();
        *e;
      }
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, CheckedAfterEarlyReturn) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (!e)
        return;
      e.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    int target($ns::expected<int, int> e) {
      if (!e.has_value())
        return 0;
      return *e;
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, CheckedInTernary) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    int target($ns::expected<int, int> e) {
      return e ? *e : 0;
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    int target($ns::expected<int, int> e) {
      return e.has_value() ? e.value() : 0;
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, ValueOrIsNotAnAccess) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      e.value_or(0);
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapErrorNoCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      e.error(); // [[unsafe:error]]
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target(const $ns::expected<int, int> &e) {
      e.error(); // [[unsafe:error]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapErrorOfFunctionCallResultNoCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target() {
      auto e = Make<$ns::expected<int, int>>();
      e.error(); // [[unsafe:error]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapVoidErrorNoCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<void, int> e) {
      e.error(); // [[unsafe:error]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnwrapErrorWhenHoldingValue) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (e) {
        e.error(); // [[unsafe:error]]
      }
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (e.has_value()) {
        e.error(); // [[unsafe:error]]
      }
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (!e)
        return;
      e.error(); // [[unsafe:error]]
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    int target($ns::expected<int, int> e) {
      return e ? //
             e.error() : 0; // [[unsafe:error]]
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, ErrorCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (!e) {
        e.error();
      }
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (!e.has_value()) {
        e.error();
      }
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (e) {
      } else {
        e.error();
      }
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<void, int> e) {
      if (!e)
        e.error();
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, ErrorCheckedAfterEarlyReturn) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    int target($ns::expected<int, int> e) {
      if (e)
        return 0;
      return e.error();
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, ErrorCheckedInTernary) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    int target($ns::expected<int, int> e) {
      return e ? 0 : e.error();
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, ValueAndErrorChecked) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      if (e) {
        *e;
      } else {
        e.error();
      }
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, ErrorOrIsNotAnAccess) {
  // `tl::expected` has no `error_or()`.
  if (GetParam().NamespaceName != "std")
    GTEST_SKIP();

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target($ns::expected<int, int> e) {
      e.error_or(0);
    }
  )");
}

TEST_P(UncheckedExpectedAccessTest, UnexpectedErrorIsNotAnAccess) {
  // `tl::unexpected` names its accessor `value()`.
  if (GetParam().NamespaceName == "tl") {
    ExpectDiagnosticsFor(R"(
      #include "unchecked_expected_access_test.h"

      void target() {
        $ns::unexpected<int> u(1);
        u.value();
      }
    )");
    return;
  }

  ExpectDiagnosticsFor(R"(
    #include "unchecked_expected_access_test.h"

    void target() {
      $ns::unexpected<int> u(1);
      u.error();
    }
  )");
}

} // namespace clang::dataflow::test
