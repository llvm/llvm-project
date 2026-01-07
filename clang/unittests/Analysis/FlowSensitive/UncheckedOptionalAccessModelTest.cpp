//===- UncheckedOptionalAccessModelTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// FIXME: Move this to clang/unittests/Analysis/FlowSensitive/Models.

#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
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
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace dataflow;
using namespace test;

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

struct OptionalTypeIdentifier {
  std::string NamespaceName;
  std::string TypeName;
};

static raw_ostream &operator<<(raw_ostream &OS,
                               const OptionalTypeIdentifier &TypeId) {
  OS << TypeId.NamespaceName << "::" << TypeId.TypeName;
  return OS;
}

class UncheckedOptionalAccessTest
    : public ::testing::TestWithParam<OptionalTypeIdentifier> {
protected:
  // Check that after running the analysis on SourceCode, it produces the
  // expected diagnostics according to [[unsafe]] annotations.
  // - No annotations => no diagnostics.
  // - Given "// [[unsafe]]" annotations on a line, we expect a diagnostic on
  //   that line.
  // - Given "// [[unsafe:range_text]]" annotations on a line, we expect a
  //   diagnostic on that line, and we expect the diagnostic Range (printed as
  //   a string) to match the "range_text".
  void ExpectDiagnosticsFor(std::string SourceCode,
                            bool IgnoreSmartPointerDereference = true) {
    ExpectDiagnosticsFor(SourceCode, ast_matchers::hasName("target"),
                         IgnoreSmartPointerDereference);
  }

  void ExpectDiagnosticsForLambda(std::string SourceCode,
                                  bool IgnoreSmartPointerDereference = true) {
    ExpectDiagnosticsFor(
        SourceCode,
        ast_matchers::hasDeclContext(
            ast_matchers::cxxRecordDecl(ast_matchers::isLambda())),
        IgnoreSmartPointerDereference);
  }

  template <typename FuncDeclMatcher>
  void ExpectDiagnosticsFor(std::string SourceCode, FuncDeclMatcher FuncMatcher,
                            bool IgnoreSmartPointerDereference = true) {
    // Run in C++17 and C++20 mode to cover differences in the AST between modes
    // (e.g. C++20 can contain `CXXRewrittenBinaryOperator`).
    for (const char *CxxMode : {"-std=c++17", "-std=c++20"})
      ExpectDiagnosticsFor(SourceCode, FuncMatcher, CxxMode,
                           IgnoreSmartPointerDereference);
  }

  template <typename FuncDeclMatcher>
  void ExpectDiagnosticsFor(std::string SourceCode, FuncDeclMatcher FuncMatcher,
                            const char *CxxMode,
                            bool IgnoreSmartPointerDereference) {
    ReplaceAllOccurrences(SourceCode, "$ns", GetParam().NamespaceName);
    ReplaceAllOccurrences(SourceCode, "$optional", GetParam().TypeName);

    auto Headers = getMockHeaders();
    Headers.emplace_back("unchecked_optional_access_test.h", R"(
      #include "absl_optional.h"
      #include "base_optional.h"
      #include "std_initializer_list.h"
      #include "std_optional.h"
      #include "std_string.h"
      #include "std_utility.h"

      template <typename T>
      T Make();
    )");
    UncheckedOptionalAccessModelOptions Options{IgnoreSmartPointerDereference};
    std::vector<UncheckedOptionalAccessDiagnostic> Diagnostics;
    llvm::Error Error = checkDataflow<UncheckedOptionalAccessModel>(
        AnalysisInputs<UncheckedOptionalAccessModel>(
            SourceCode, std::move(FuncMatcher),
            [](ASTContext &Ctx, Environment &Env) {
              return UncheckedOptionalAccessModel(Ctx, Env);
            })
            .withDiagnosisCallbacks(
                {/*Before=*/[&Diagnostics,
                             Diagnoser =
                                 UncheckedOptionalAccessDiagnoser(Options)](
                                ASTContext &Ctx, const CFGElement &Elt,
                                const TransferStateForDiagnostics<
                                    UncheckedOptionalAccessLattice>
                                    &State) mutable {
                   auto EltDiagnostics = Diagnoser(Elt, Ctx, State);
                   llvm::move(EltDiagnostics, std::back_inserter(Diagnostics));
                 },
                 /*After=*/nullptr})
            .withASTBuildArgs(
                {"-fsyntax-only", CxxMode, "-Wno-undefined-inline"})
            .withASTBuildVirtualMappedFiles(
                tooling::FileContentMappings(Headers.begin(), Headers.end())),
        /*VerifyResults=*/[&Diagnostics](
                              const llvm::DenseMap<unsigned, std::string>
                                  &Annotations,
                              const AnalysisOutputs &AO) {
          llvm::DenseSet<unsigned> AnnotationLines;
          llvm::DenseMap<unsigned, std::string> AnnotationRangesInLines;
          for (const auto &[Line, AnnotationWithMaybeRange] : Annotations) {
            AnnotationLines.insert(Line);
            auto it = AnnotationWithMaybeRange.find(':');
            if (it != std::string::npos) {
              AnnotationRangesInLines[Line] =
                  AnnotationWithMaybeRange.substr(it + 1);
            }
          }
          auto &SrcMgr = AO.ASTCtx.getSourceManager();
          llvm::DenseSet<unsigned> DiagnosticLines;
          for (const UncheckedOptionalAccessDiagnostic &Diag : Diagnostics) {
            unsigned Line = SrcMgr.getPresumedLineNumber(Diag.Range.getBegin());
            DiagnosticLines.insert(Line);
            if (!AnnotationLines.contains(Line)) {
              DiagnosticOptions DiagOpts;
              TextDiagnostic TD(llvm::errs(), AO.ASTCtx.getLangOpts(),
                                DiagOpts);
              TD.emitDiagnostic(FullSourceLoc(Diag.Range.getBegin(), SrcMgr),
                                DiagnosticsEngine::Error,
                                "unexpected diagnostic", {Diag.Range}, {});
            } else {
              auto it = AnnotationRangesInLines.find(Line);
              if (it != AnnotationRangesInLines.end()) {
                EXPECT_EQ(Diag.Range.getAsRange().printToString(SrcMgr),
                          it->second);
              }
            }
          }

          EXPECT_THAT(DiagnosticLines, ContainerEq(AnnotationLines));
        });
    if (Error)
      FAIL() << llvm::toString(std::move(Error));
  }
};

INSTANTIATE_TEST_SUITE_P(
    UncheckedOptionalUseTestInst, UncheckedOptionalAccessTest,
    ::testing::Values(OptionalTypeIdentifier{"std", "optional"},
                      OptionalTypeIdentifier{"absl", "optional"},
                      OptionalTypeIdentifier{"base", "Optional"}),
    [](const ::testing::TestParamInfo<OptionalTypeIdentifier> &Info) {
      return Info.param.NamespaceName;
    });

// Verifies that similarly-named types are ignored.
TEST_P(UncheckedOptionalAccessTest, NonTrackedOptionalType) {
  ExpectDiagnosticsFor(
      R"(
    namespace other {
    namespace $ns {
    template <typename T>
    struct $optional {
      T value();
    };
    }

    void target($ns::$optional<int> opt) {
      opt.value();
    }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, EmptyFunctionBody) {
  ExpectDiagnosticsFor(R"(
    void target() {
      (void)0;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, UnwrapUsingValueNoCheck) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      opt.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      std::move(opt).value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, UnwrapUsingOperatorStarNoCheck) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      *opt; // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      *std::move(opt); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, UnwrapUsingOperatorArrowNoCheck) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      void foo();
    };

    void target($ns::$optional<Foo> opt) {
      opt->foo(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      void foo();
    };

    void target($ns::$optional<Foo> opt) {
      std::move(opt)->foo(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, HasValueCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      if (opt.has_value()) {
        opt.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, OperatorBoolCheck) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      if (opt) {
        opt.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, UnwrapFunctionCallResultNoCheck) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      Make<$ns::$optional<int>>().value(); // [[unsafe]]
      (void)0;
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      std::move(opt).value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, DefaultConstructor) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt;
      opt.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, NulloptConstructor) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt($ns::nullopt);
      opt.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, NulloptConstructorWithSugaredType) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"
    template <typename T>
    using wrapper = T;

    template <typename T>
    wrapper<T> wrap(T);

    void target() {
      $ns::$optional<int> opt(wrap($ns::nullopt));
      opt.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InPlaceConstructor) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt($ns::in_place, 3);
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    void target() {
      $ns::$optional<Foo> opt($ns::in_place);
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      explicit Foo(int, bool);
    };

    void target() {
      $ns::$optional<Foo> opt($ns::in_place, 3, false);
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      explicit Foo(std::initializer_list<int>);
    };

    void target() {
      $ns::$optional<Foo> opt($ns::in_place, {3});
      opt.value();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ValueConstructor) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt(21);
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::$optional<int>(21);
      opt.value();
    }
  )");
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<$ns::$optional<int>> opt(Make<$ns::$optional<int>>());
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct MyString {
      MyString(const char*);
    };

    void target() {
      $ns::$optional<MyString> opt("foo");
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Bar> opt(Make<Foo>());
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      explicit Foo(int);
    };

    void target() {
      $ns::$optional<Foo> opt(3);
      opt.value();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ConvertibleOptionalConstructor) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Bar> opt(Make<$ns::$optional<Foo>>());
      opt.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      explicit Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Bar> opt(Make<$ns::$optional<Foo>>());
      opt.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1 = $ns::nullopt;
      $ns::$optional<Bar> opt2(opt1);
      opt2.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1(Make<Foo>());
      $ns::$optional<Bar> opt2(opt1);
      opt2.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      explicit Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1(Make<Foo>());
      $ns::$optional<Bar> opt2(opt1);
      opt2.value();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, MakeOptional) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::make_optional(0);
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      Foo(int, int);
    };

    void target() {
      $ns::$optional<Foo> opt = $ns::make_optional<Foo>(21, 22);
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      constexpr Foo(std::initializer_list<char>);
    };

    void target() {
      char a = 'a';
      $ns::$optional<Foo> opt = $ns::make_optional<Foo>({a});
      opt.value();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ValueOr) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt;
      opt.value_or(0);
      (void)0;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ValueOrComparisonPointers) {
  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int*> opt) {
      if (opt.value_or(nullptr) != nullptr) {
        opt.value();
      } else {
        opt.value(); // [[unsafe]]
      }
    }
  )code");
}

TEST_P(UncheckedOptionalAccessTest, ValueOrComparisonIntegers) {
  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      if (opt.value_or(0) != 0) {
        opt.value();
      } else {
        opt.value(); // [[unsafe]]
      }
    }
  )code");
}

TEST_P(UncheckedOptionalAccessTest, ValueOrComparisonStrings) {
  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<std::string> opt) {
      if (!opt.value_or("").empty()) {
        opt.value();
      } else {
        opt.value(); // [[unsafe]]
      }
    }
  )code");

  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<std::string> opt) {
      if (opt.value_or("") != "") {
        opt.value();
      } else {
        opt.value(); // [[unsafe]]
      }
    }
  )code");
}

TEST_P(UncheckedOptionalAccessTest, ValueOrComparisonPointerToOptional) {
  // FIXME: make `opt` a parameter directly, once we ensure that all `optional`
  // values have a `has_value` property.
  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> p) {
      $ns::$optional<int> *opt = &p;
      if (opt->value_or(0) != 0) {
        opt->value();
      } else {
        opt->value(); // [[unsafe]]
      }
    }
  )code");
}

TEST_P(UncheckedOptionalAccessTest, Emplace) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt;
      opt.emplace(0);
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> *opt) {
      opt->emplace(0);
      opt->value();
    }
  )");

  // FIXME: Add tests that call `emplace` in conditional branches:
  //  ExpectDiagnosticsFor(
  //      R"(
  //    #include "unchecked_optional_access_test.h"
  //
  //    void target($ns::$optional<int> opt, bool b) {
  //      if (b) {
  //        opt.emplace(0);
  //      }
  //      if (b) {
  //        opt.value();
  //      } else {
  //        opt.value(); // [[unsafe]]
  //      }
  //    }
  //  )");
}

TEST_P(UncheckedOptionalAccessTest, Reset) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::make_optional(0);
      opt.reset();
      opt.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> &opt) {
      if (opt.has_value()) {
        opt.reset();
        opt.value(); // [[unsafe]]
      }
    }
  )");

  // FIXME: Add tests that call `reset` in conditional branches:
  //  ExpectDiagnosticsFor(
  //      R"(
  //    #include "unchecked_optional_access_test.h"
  //
  //    void target(bool b) {
  //      $ns::$optional<int> opt = $ns::make_optional(0);
  //      if (b) {
  //        opt.reset();
  //      }
  //      if (b) {
  //        opt.value(); // [[unsafe]]
  //      } else {
  //        opt.value();
  //      }
  //    }
  //  )");
}

TEST_P(UncheckedOptionalAccessTest, ValueAssignment) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    void target() {
      $ns::$optional<Foo> opt;
      opt = Foo();
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    void target() {
      $ns::$optional<Foo> opt;
      (opt = Foo()).value();
      (void)0;
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct MyString {
      MyString(const char*);
    };

    void target() {
      $ns::$optional<MyString> opt;
      opt = "foo";
      opt.value();
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct MyString {
      MyString(const char*);
    };

    void target() {
      $ns::$optional<MyString> opt;
      (opt = "foo").value();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, OptionalConversionAssignment) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1 = Foo();
      $ns::$optional<Bar> opt2;
      opt2 = opt1;
      opt2.value();
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1;
      $ns::$optional<Bar> opt2;
      if (opt2.has_value()) {
        opt2 = opt1;
        opt2.value(); // [[unsafe]]
      }
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1 = Foo();
      $ns::$optional<Bar> opt2;
      (opt2 = opt1).value();
      (void)0;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, NulloptAssignment) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = 3;
      opt = $ns::nullopt;
      opt.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = 3;
      (opt = $ns::nullopt).value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, OptionalSwap) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = 3;

      opt1.swap(opt2);

      opt1.value();

      opt2.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = 3;

      opt2.swap(opt1);

      opt1.value();

      opt2.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, OptionalReturnedFromFuntionCall) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"
    
    struct S {
      $ns::$optional<float> x;
    } s;
    S getOptional() {
      return s;
    }

    void target() {
      getOptional().x = 0;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, NonConstMethodMayClearOptionalField) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      $ns::$optional<std::string> opt;
      void clear();  // assume this may modify the opt field's state
    };

    void target(Foo& foo) {
      if (foo.opt) {
        foo.opt.value();
        foo.clear();
        foo.opt.value();  // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest,
       NonConstMethodMayNotClearConstOptionalField) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      const $ns::$optional<std::string> opt;
      void clear();
    };

    void target(Foo& foo) {
      if (foo.opt) {
        foo.opt.value();
        foo.clear();
        foo.opt.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, StdSwap) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = 3;

      std::swap(opt1, opt2);

      opt1.value();

      opt2.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = 3;

      std::swap(opt2, opt1);

      opt1.value();

      opt2.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, SwapUnmodeledLocLeft) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct L { $ns::$optional<int> hd; L* tl; };

    void target() {
      $ns::$optional<int> foo = 3;
      L bar;

      // Any `tl` beyond the first is not modeled.
      bar.tl->tl->hd.swap(foo);

      bar.tl->tl->hd.value(); // [[unsafe]]
      foo.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, SwapUnmodeledLocRight) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct L { $ns::$optional<int> hd; L* tl; };

    void target() {
      $ns::$optional<int> foo = 3;
      L bar;

      // Any `tl` beyond the first is not modeled.
      foo.swap(bar.tl->tl->hd);

      bar.tl->tl->hd.value(); // [[unsafe]]
      foo.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, SwapUnmodeledValueLeftSet) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct S { int x; };
    struct A { $ns::$optional<S> late; };
    struct B { A f3; };
    struct C { B f2; };
    struct D { C f1; };

    void target() {
      $ns::$optional<S> foo = S{3};
      D bar;

      bar.f1.f2.f3.late.swap(foo);

      bar.f1.f2.f3.late.value();
      foo.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, SwapUnmodeledValueLeftUnset) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct S { int x; };
    struct A { $ns::$optional<S> late; };
    struct B { A f3; };
    struct C { B f2; };
    struct D { C f1; };

    void target() {
      $ns::$optional<S> foo;
      D bar;

      bar.f1.f2.f3.late.swap(foo);

      bar.f1.f2.f3.late.value(); // [[unsafe]]
      foo.value(); // [[unsafe]]
    }
  )");
}

// fixme: use recursion instead of depth.
TEST_P(UncheckedOptionalAccessTest, SwapUnmodeledValueRightSet) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct S { int x; };
    struct A { $ns::$optional<S> late; };
    struct B { A f3; };
    struct C { B f2; };
    struct D { C f1; };

    void target() {
      $ns::$optional<S> foo = S{3};
      D bar;

      foo.swap(bar.f1.f2.f3.late);

      bar.f1.f2.f3.late.value();
      foo.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, SwapUnmodeledValueRightUnset) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct S { int x; };
    struct A { $ns::$optional<S> late; };
    struct B { A f3; };
    struct C { B f2; };
    struct D { C f1; };

    void target() {
      $ns::$optional<S> foo;
      D bar;

      foo.swap(bar.f1.f2.f3.late);

      bar.f1.f2.f3.late.value(); // [[unsafe]]
      foo.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, UniquePtrToOptional) {
  // We suppress diagnostics for optionals in smart pointers (other than
  // `optional` itself).
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    template <typename T>
    struct smart_ptr {
      T& operator*() &;
      T* operator->();
    };

    void target() {
      smart_ptr<$ns::$optional<bool>> foo;
      foo->value();
      (*foo).value();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, UniquePtrToStructWithOptionalField) {
  // We suppress diagnostics for optional fields reachable from smart pointers
  // (other than `optional` itself) through (exactly) one member access.
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    template <typename T>
    struct smart_ptr {
      T& operator*() &;
      T* operator->();
    };

    struct Foo {
      $ns::$optional<int> opt;
    };

    void target() {
      smart_ptr<Foo> foo;
      *foo->opt;
      *(*foo).opt;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, CallReturningOptional) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    $ns::$optional<int> MakeOpt();

    void target() {
      $ns::$optional<int> opt = 0;
      opt = MakeOpt();
      opt.value(); // [[unsafe]]
    }
  )");
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    const $ns::$optional<int>& MakeOpt();

    void target() {
      $ns::$optional<int> opt = 0;
      opt = MakeOpt();
      opt.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    using IntOpt = $ns::$optional<int>;
    IntOpt MakeOpt();

    void target() {
      IntOpt opt = 0;
      opt = MakeOpt();
      opt.value(); // [[unsafe]]
    }
  )");

  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    using IntOpt = $ns::$optional<int>;
    const IntOpt& MakeOpt();

    void target() {
      IntOpt opt = 0;
      opt = MakeOpt();
      opt.value(); // [[unsafe]]
    }
  )");
}


TEST_P(UncheckedOptionalAccessTest, EqualityCheckLeftSet) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = 3;
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt1 == opt2) {
        opt2.value();
      } else {
        opt2.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, EqualityCheckRightSet) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = 3;
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt2 == opt1) {
        opt2.value();
      } else {
        opt2.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, EqualityCheckVerifySetAfterEq) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = Make<$ns::$optional<int>>();
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt1 == opt2) {
        if (opt1.has_value())
          opt2.value();
        if (opt2.has_value())
          opt1.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, EqualityCheckLeftUnset) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt1 == opt2) {
        opt2.value(); // [[unsafe]]
      } else {
        opt2.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, EqualityCheckRightUnset) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt2 == opt1) {
        opt2.value(); // [[unsafe]]
      } else {
        opt2.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, EqualityCheckRightNullopt) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = Make<$ns::$optional<int>>();

      if (opt == $ns::nullopt) {
        opt.value(); // [[unsafe]]
      } else {
        opt.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, EqualityCheckLeftNullopt) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = Make<$ns::$optional<int>>();

      if ($ns::nullopt == opt) {
        opt.value(); // [[unsafe]]
      } else {
        opt.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, EqualityCheckRightValue) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = Make<$ns::$optional<int>>();

      if (opt == 3) {
        opt.value();
      } else {
        opt.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, EqualityCheckLeftValue) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = Make<$ns::$optional<int>>();

      if (3 == opt) {
        opt.value();
      } else {
        opt.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InequalityCheckLeftSet) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = 3;
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt1 != opt2) {
        opt2.value(); // [[unsafe]]
      } else {
        opt2.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InequalityCheckRightSet) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = 3;
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt2 != opt1) {
        opt2.value(); // [[unsafe]]
      } else {
        opt2.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InequalityCheckVerifySetAfterEq) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = Make<$ns::$optional<int>>();
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt1 != opt2) {
        if (opt1.has_value())
          opt2.value(); // [[unsafe]]
        if (opt2.has_value())
          opt1.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InequalityCheckLeftUnset) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt1 != opt2) {
        opt2.value();
      } else {
        opt2.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InequalityCheckRightUnset) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = Make<$ns::$optional<int>>();

      if (opt2 != opt1) {
        opt2.value();
      } else {
        opt2.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InequalityCheckRightNullopt) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = Make<$ns::$optional<int>>();

      if (opt != $ns::nullopt) {
        opt.value();
      } else {
        opt.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InequalityCheckLeftNullopt) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = Make<$ns::$optional<int>>();

      if ($ns::nullopt != opt) {
        opt.value();
      } else {
        opt.value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InequalityCheckRightValue) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = Make<$ns::$optional<int>>();

      if (opt != 3) {
        opt.value(); // [[unsafe]]
      } else {
        opt.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, InequalityCheckLeftValue) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = Make<$ns::$optional<int>>();

      if (3 != opt) {
        opt.value(); // [[unsafe]]
      } else {
        opt.value();
      }
    }
  )");
}

// Verifies that the model sees through aliases.
TEST_P(UncheckedOptionalAccessTest, WithAlias) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    template <typename T>
    using MyOptional = $ns::$optional<T>;

    void target(MyOptional<int> opt) {
      opt.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, OptionalValueOptional) {
  // Basic test that nested values are populated.  We nest an optional because
  // its easy to use in a test, but the type of the nested value shouldn't
  // matter.
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    using Foo = $ns::$optional<std::string>;

    void target($ns::$optional<Foo> foo) {
      if (foo && *foo) {
        foo->value();
      }
    }
  )");

  // Mutation is supported for nested values.
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    using Foo = $ns::$optional<std::string>;

    void target($ns::$optional<Foo> foo) {
      if (foo && *foo) {
        foo->reset();
        foo->value(); // [[unsafe]]
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, NestedOptionalAssignValue) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    using OptionalInt = $ns::$optional<int>;

    void target($ns::$optional<OptionalInt> opt) {
      if (!opt) return;

      // Accessing the outer optional is OK now.
      *opt;

      // But accessing the nested optional is still unsafe because we haven't
      // checked it.
      **opt;  // [[unsafe]]

      *opt = 1;

      // Accessing the nested optional is safe after assigning a value to it.
      **opt;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, NestedOptionalAssignOptional) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    using OptionalInt = $ns::$optional<int>;

    void target($ns::$optional<OptionalInt> opt) {
      if (!opt) return;

      // Accessing the outer optional is OK now.
      *opt;

      // But accessing the nested optional is still unsafe because we haven't
      // checked it.
      **opt;  // [[unsafe]]

      // Assign from `optional<short>` so that we trigger conversion assignment
      // instead of move assignment.
      *opt = $ns::$optional<short>();

      // Accessing the nested optional is still unsafe after assigning an empty
      // optional to it.
      **opt;  // [[unsafe]]
    }
  )");
}

// Tests that structs can be nested. We use an optional field because its easy
// to use in a test, but the type of the field shouldn't matter.
TEST_P(UncheckedOptionalAccessTest, OptionalValueStruct) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      $ns::$optional<std::string> opt;
    };

    void target($ns::$optional<Foo> foo) {
      if (foo && foo->opt) {
        foo->opt.value();
      }
    }
  )");
}

// FIXME: A case that we should handle but currently don't.
// When there is a field of type reference to non-optional, we may
// stop recursively creating storage locations.
// E.g., the field `second` below in `pair` should eventually lead to
// the optional `x` in `A`.
TEST_P(UncheckedOptionalAccessTest, NestedOptionalThroughNonOptionalRefField) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct A {
      $ns::$optional<int> x;
    };

    struct pair {
      int first;
      const A &second;
    };

    struct B {
      $ns::$optional<pair>& nonConstGetRef();
    };

    void target(B b) {
      const auto& maybe_pair = b.nonConstGetRef();
      if (!maybe_pair.has_value())
        return;

      if(!maybe_pair->second.x.has_value())
        return;
      maybe_pair->second.x.value();  // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, OptionalValueInitialization) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    using Foo = $ns::$optional<std::string>;

    void target($ns::$optional<Foo> foo, bool b) {
      if (!foo.has_value()) return;
      if (b) {
        if (!foo->has_value()) return;
        // We have created `foo.value()`.
        foo->value();
      } else {
        if (!foo->has_value()) return;
        // We have created `foo.value()` again, in a different environment.
        foo->value();
      }
      // Now we merge the two values. UncheckedOptionalAccessModel::merge() will
      // throw away the "value" property.
      foo->value();
    }
  )");
}

// This test is aimed at the core model, not the diagnostic. It is a regression
// test against a crash when using non-trivial smart pointers, like
// `std::unique_ptr`. As such, it doesn't test the access itself, which would be
// ignored regardless because of `IgnoreSmartPointerDereference = true`, above.
TEST_P(UncheckedOptionalAccessTest, AssignThroughLvalueReferencePtr) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    template <typename T>
    struct smart_ptr {
      typename std::add_lvalue_reference<T>::type operator*() &;
    };

    void target() {
      smart_ptr<$ns::$optional<int>> x;
      // Verify that this assignment does not crash.
      *x = 3;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, CorrelatedBranches) {
  ExpectDiagnosticsFor(R"code(
    #include "unchecked_optional_access_test.h"

    void target(bool b, $ns::$optional<int> opt) {
      if (b || opt.has_value()) {
        if (!b) {
          opt.value();
        }
      }
    }
  )code");

  ExpectDiagnosticsFor(R"code(
    #include "unchecked_optional_access_test.h"

    void target(bool b, $ns::$optional<int> opt) {
      if (b && !opt.has_value()) return;
      if (b) {
        opt.value();
      }
    }
  )code");

  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target(bool b, $ns::$optional<int> opt) {
      if (opt.has_value()) b = true;
      if (b) {
        opt.value(); // [[unsafe]]
      }
    }
  )code");

  ExpectDiagnosticsFor(R"code(
    #include "unchecked_optional_access_test.h"

    void target(bool b, $ns::$optional<int> opt) {
      if (b) return;
      if (opt.has_value()) b = true;
      if (b) {
        opt.value();
      }
    }
  )code");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target(bool b, $ns::$optional<int> opt) {
      if (opt.has_value() == b) {
        if (b) {
          opt.value();
        }
      }
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target(bool b, $ns::$optional<int> opt) {
      if (opt.has_value() != b) {
        if (!b) {
          opt.value();
        }
      }
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target(bool b) {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2;
      if (b) {
        opt2 = $ns::nullopt;
      } else {
        opt2 = $ns::nullopt;
      }
      if (opt2.has_value()) {
        opt1.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, JoinDistinctValues) {
  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target(bool b) {
      $ns::$optional<int> opt;
      if (b) {
        opt = Make<$ns::$optional<int>>();
      } else {
        opt = Make<$ns::$optional<int>>();
      }
      if (opt.has_value()) {
        opt.value();
      } else {
        opt.value(); // [[unsafe]]
      }
    }
  )code");

  ExpectDiagnosticsFor(R"code(
    #include "unchecked_optional_access_test.h"

    void target(bool b) {
      $ns::$optional<int> opt;
      if (b) {
        opt = Make<$ns::$optional<int>>();
        if (!opt.has_value()) return;
      } else {
        opt = Make<$ns::$optional<int>>();
        if (!opt.has_value()) return;
      }
      opt.value();
    }
  )code");

  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target(bool b) {
      $ns::$optional<int> opt;
      if (b) {
        opt = Make<$ns::$optional<int>>();
        if (!opt.has_value()) return;
      } else {
        opt = Make<$ns::$optional<int>>();
      }
      opt.value(); // [[unsafe]]
    }
  )code");

  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target(bool b) {
      $ns::$optional<int> opt;
      if (b) {
        opt = 1;
      } else {
        opt = 2;
      }
      opt.value();
    }
  )code");

  ExpectDiagnosticsFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target(bool b) {
      $ns::$optional<int> opt;
      if (b) {
        opt = 1;
      } else {
        opt = Make<$ns::$optional<int>>();
      }
      opt.value(); // [[unsafe]]
    }
  )code");
}

TEST_P(UncheckedOptionalAccessTest, AccessValueInLoop) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = 3;
      while (Make<bool>()) {
        opt.value();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ReassignValueInLoopWithCheckSafe) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = 3;
      while (Make<bool>()) {
        opt.value();

        opt = Make<$ns::$optional<int>>();
        if (!opt.has_value()) return;
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ReassignValueInLoopNoCheckUnsafe) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = 3;
      while (Make<bool>()) {
        opt.value(); // [[unsafe]]

        opt = Make<$ns::$optional<int>>();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ReassignValueInLoopToUnsetUnsafe) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = 3;
      while (Make<bool>())
        opt = $ns::nullopt;
      $ns::$optional<int> opt2 = $ns::nullopt;
      if (opt.has_value())
        opt2 = $ns::$optional<int>(3);
      opt2.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ReassignValueInLoopToSetUnsafe) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::nullopt;
      while (Make<bool>())
        opt = $ns::$optional<int>(3);
      $ns::$optional<int> opt2 = $ns::nullopt;
      if (!opt.has_value())
        opt2 = $ns::$optional<int>(3);
      opt2.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ReassignValueInLoopToUnknownUnsafe) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::nullopt;
      while (Make<bool>())
        opt = Make<$ns::$optional<int>>();
      $ns::$optional<int> opt2 = $ns::nullopt;
      if (!opt.has_value())
        opt2 = $ns::$optional<int>(3);
      opt2.value(); // [[unsafe]]
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ReassignValueInLoopBadConditionUnsafe) {
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = 3;
      while (Make<bool>()) {
        opt.value(); // [[unsafe]]

        opt = Make<$ns::$optional<int>>();
        if (!opt.has_value()) continue;
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, StructuredBindingsFromStruct) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct kv { $ns::$optional<int> opt; int x; };
    int target() {
      auto [contents, x] = Make<kv>();
      return contents ? *contents : x;
    }
  )");

  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    template <typename T1, typename T2>
    struct pair { T1 fst;  T2 snd; };
    int target() {
      auto [contents, x] = Make<pair<$ns::$optional<int>, int>>();
      return contents ? *contents : x;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, StructuredBindingsFromTupleLikeType) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    namespace std {
    template <class> struct tuple_size;
    template <size_t, class> struct tuple_element;
    template <class...> class tuple;

    template <class... T>
    struct tuple_size<tuple<T...>> : integral_constant<size_t, sizeof...(T)> {};

    template <size_t I, class... T>
    struct tuple_element<I, tuple<T...>> {
      using type =  __type_pack_element<I, T...>;
    };

    template <class...> class tuple {};
    template <size_t I, class... T>
    typename tuple_element<I, tuple<T...>>::type get(tuple<T...>);
    } // namespace std

    std::tuple<$ns::$optional<const char *>, int> get_opt();
    void target() {
      auto [content, ck] = get_opt();
      content ? *content : "";
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, CtorInitializerNullopt) {
  using namespace ast_matchers;
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Target {
      Target(): opt($ns::nullopt) {
        opt.value(); // [[unsafe]]
      }
      $ns::$optional<int> opt;
    };
  )",
      cxxConstructorDecl(ofClass(hasName("Target"))));
}

TEST_P(UncheckedOptionalAccessTest, CtorInitializerValue) {
  using namespace ast_matchers;
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Target {
      Target(): opt(3) {
        opt.value();
      }
      $ns::$optional<int> opt;
    };
  )",
      cxxConstructorDecl(ofClass(hasName("Target"))));
}

// This is regression test, it shouldn't crash.
TEST_P(UncheckedOptionalAccessTest, Bitfield) {
  using namespace ast_matchers;
  ExpectDiagnosticsFor(
      R"(
    #include "unchecked_optional_access_test.h"
    struct Dst {
      unsigned int n : 1;
    };
    void target() {
      $ns::$optional<bool> v;
      Dst d;
      if (v.has_value())
        d.n = v.value();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, LambdaParam) {
  ExpectDiagnosticsForLambda(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      []($ns::$optional<int> opt) {
        if (opt.has_value()) {
          opt.value();
        } else {
          opt.value(); // [[unsafe]]
        }
      }(Make<$ns::$optional<int>>());
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, LambdaCaptureByCopy) {
  ExpectDiagnosticsForLambda(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      [opt]() {
        if (opt.has_value()) {
          opt.value();
        } else {
          opt.value(); // [[unsafe]]
        }
      }();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, LambdaCaptureByReference) {
  ExpectDiagnosticsForLambda(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      [&opt]() {
        if (opt.has_value()) {
          opt.value();
        } else {
          opt.value(); // [[unsafe]]
        }
      }();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, LambdaCaptureWithInitializer) {
  ExpectDiagnosticsForLambda(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      [opt2=opt]() {
        if (opt2.has_value()) {
          opt2.value();
        } else {
          opt2.value(); // [[unsafe]]
        }
      }();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, LambdaCaptureByCopyImplicit) {
  ExpectDiagnosticsForLambda(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      [=]() {
        if (opt.has_value()) {
          opt.value();
        } else {
          opt.value(); // [[unsafe]]
        }
      }();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, LambdaCaptureByReferenceImplicit) {
  ExpectDiagnosticsForLambda(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      [&]() {
        if (opt.has_value()) {
          opt.value();
        } else {
          opt.value(); // [[unsafe]]
        }
      }();
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, LambdaCaptureThis) {
  ExpectDiagnosticsForLambda(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      $ns::$optional<int> opt;

      void target() {
        [this]() {
          if (opt.has_value()) {
            opt.value();
          } else {
            opt.value(); // [[unsafe]]
          }
        }();
      }
    };
  )");
}

TEST_P(UncheckedOptionalAccessTest, LambdaCaptureStateNotPropagated) {
  // We can't propagate information from the surrounding context.
  ExpectDiagnosticsForLambda(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      if (opt.has_value()) {
        [&opt]() {
          opt.value(); // [[unsafe]]
        }();
      }
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ClassDerivedFromOptional) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Derived : public $ns::$optional<int> {};

    void target(Derived opt) {
      *opt;  // [[unsafe]]
      if (opt.has_value())
        *opt;

      // The same thing, but with a pointer receiver.
      Derived *popt = &opt;
      **popt;  // [[unsafe]]
      if (popt->has_value())
        **popt;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ClassTemplateDerivedFromOptional) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    template <class T>
    struct Derived : public $ns::$optional<T> {};

    void target(Derived<int> opt) {
      *opt;  // [[unsafe]]
      if (opt.has_value())
        *opt;

      // The same thing, but with a pointer receiver.
      Derived<int> *popt = &opt;
      **popt;  // [[unsafe]]
      if (popt->has_value())
        **popt;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ClassDerivedPrivatelyFromOptional) {
  // Classes that derive privately from optional can themselves still call
  // member functions of optional. Check that we model the optional correctly
  // in this situation.
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Derived : private $ns::$optional<int> {
      void Method() {
        **this;  // [[unsafe]]
        if (this->has_value())
          **this;
      }
    };
  )",
                       ast_matchers::hasName("Method"));
}

TEST_P(UncheckedOptionalAccessTest, ClassDerivedFromOptionalValueConstructor) {
  ExpectDiagnosticsFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Derived : public $ns::$optional<int> {
      Derived(int);
    };

    void target(Derived opt) {
      *opt;  // [[unsafe]]
      opt = 1;
      *opt;
    }
  )");
}

TEST_P(UncheckedOptionalAccessTest, ConstRefAccessor) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }
      $ns::$optional<int> x;
    };

    void target(A& a) {
      if (a.get().has_value()) {
        a.get().value();
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest, ConstRefAccessorWithModInBetween) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }
      void clear();
      $ns::$optional<int> x;
    };

    void target(A& a) {
      if (a.get().has_value()) {
        a.clear();
        a.get().value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest, ConstRefAccessorWithModReturningOptional) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }
      $ns::$optional<int> take();
      $ns::$optional<int> x;
    };

    void target(A& a) {
      if (a.get().has_value()) {
        $ns::$optional<int> other = a.take();
        a.get().value();  // [[unsafe]]
        if (other.has_value()) {
          other.value();
        }
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest, ConstRefAccessorDifferentObjects) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }
      $ns::$optional<int> x;
    };

    void target(A& a1, A& a2) {
      if (a1.get().has_value()) {
        a2.get().value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest, ConstRefAccessorLoop) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }
      $ns::$optional<int> x;
    };

    void target(A& a, int N) {
      for (int i = 0; i < N; ++i) {
        if (a.get().has_value()) {
          a.get().value();
        }
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest, ConstByValueAccessor) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      $ns::$optional<int> get() const { return x; }
      $ns::$optional<int> x;
    };

    void target(A& a) {
      if (a.get().has_value()) {
        a.get().value();
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest, ConstByValueAccessorWithModInBetween) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      $ns::$optional<int> get() const { return x; }
      void clear();
      $ns::$optional<int> x;
    };

    void target(A& a) {
      if (a.get().has_value()) {
        a.clear();
        a.get().value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest, ConstPointerAccessor) {
  ExpectDiagnosticsFor(R"cc(
     #include "unchecked_optional_access_test.h"

    struct A {
      $ns::$optional<int> x;
    };

    struct MyUniquePtr {
      A* operator->() const;
    };

    void target(MyUniquePtr p) {
      if (p->x) {
        *p->x;
      }
    }
  )cc",
                       /*IgnoreSmartPointerDereference=*/false);
}

TEST_P(UncheckedOptionalAccessTest, ConstPointerAccessorWithModInBetween) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      $ns::$optional<int> x;
    };

    struct MyUniquePtr {
      A* operator->() const;
      void reset(A*);
    };

    void target(MyUniquePtr p) {
      if (p->x) {
        p.reset(nullptr);
        *p->x;  // [[unsafe]]
      }
    }
  )cc",
                       /*IgnoreSmartPointerDereference=*/false);
}

TEST_P(UncheckedOptionalAccessTest, SmartPointerAccessorMixed) {
  ExpectDiagnosticsFor(R"cc(
     #include "unchecked_optional_access_test.h"

    struct A {
      $ns::$optional<int> x;
    };

    namespace absl {
    template<typename T>
    class StatusOr {
      public:
      bool ok() const;

      const T& operator*() const&;
      T& operator*() &;

      const T* operator->() const;
      T* operator->();

      const T& value() const;
      T& value();
    };
    }

    void target(absl::StatusOr<A> &mut, const absl::StatusOr<A> &imm) {
      if (!mut.ok() || !imm.ok())
        return;

      if (mut->x.has_value()) {
        mut->x.value();
        ((*mut).x).value();
        (mut.value().x).value();

        // check flagged after modifying
        mut = imm;
        mut->x.value();  // [[unsafe]]
      }
      if (imm->x.has_value()) {
        imm->x.value();
        ((*imm).x).value();
        (imm.value().x).value();
      }
    }
  )cc",
                       /*IgnoreSmartPointerDereference=*/false);
}

TEST_P(UncheckedOptionalAccessTest, ConstBoolAccessor) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      bool isFoo() const { return f; }
      bool f;
    };

    void target(A& a) {
      $ns::$optional<int> opt;
      if (a.isFoo()) {
        opt = 1;
      }
      if (a.isFoo()) {
        opt.value();
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest, ConstBoolAccessorWithModInBetween) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      bool isFoo() const { return f; }
      void clear();
      bool f;
    };

    void target(A& a) {
      $ns::$optional<int> opt;
      if (a.isFoo()) {
        opt = 1;
      }
      a.clear();
      if (a.isFoo()) {
        opt.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest,
       ConstRefAccessorToOptionalViaConstRefAccessorToHoldingObject) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }

      $ns::$optional<int> x;
    };

    struct B {
      const A& getA() const { return a; }

      A a;
    };

    void target(B& b) {
      if (b.getA().get().has_value()) {
        b.getA().get().value();
      }
    }
  )cc");
}

TEST_P(
    UncheckedOptionalAccessTest,
    ConstRefAccessorToOptionalViaConstRefAccessorToHoldingObjectWithoutValueCheck) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }

      $ns::$optional<int> x;
    };

    struct B {
      const A& getA() const { return a; }

      A a;
    };

    void target(B& b) {
      b.getA().get().value(); // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest,
       ConstRefToOptionalSavedAsTemporaryVariable) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }

      $ns::$optional<int> x;
    };

    struct B {
      const A& getA() const { return a; }

      A a;
    };

    void target(B& b) {
      const auto& opt = b.getA().get();
      if (opt.has_value()) {
        opt.value();
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest,
       ConstRefAccessorToOptionalViaAccessorToHoldingObjectByValue) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }

      $ns::$optional<int> x;
    };

    struct B {
      const A copyA() const { return a; }

      A a;
    };

    void target(B& b) {
      if (b.copyA().get().has_value()) {
        b.copyA().get().value(); // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedOptionalAccessTest,
       ConstRefAccessorToOptionalViaNonConstRefAccessorToHoldingObject) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }

      $ns::$optional<int> x;
    };

    struct B {
      A& getA() { return a; }

      A a;
    };

    void target(B& b) {
      if (b.getA().get().has_value()) {
        b.getA().get().value(); // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(
    UncheckedOptionalAccessTest,
    ConstRefAccessorToOptionalViaConstRefAccessorToHoldingObjectWithModAfterCheck) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      const $ns::$optional<int>& get() const { return x; }

      $ns::$optional<int> x;
    };

    struct B {
      const A& getA() const { return a; }

      A& getA() { return a; }

      void clear() { a = A{}; }

      A a;
    };

    void target(B& b) {
      // changing field A via non-const getter after const getter check
      if (b.getA().get().has_value()) {
        b.getA() = A{};
        b.getA().get().value(); // [[unsafe]]
      }

      // calling non-const method which might change field A
      if (b.getA().get().has_value()) {
        b.clear();
        b.getA().get().value(); // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(
    UncheckedOptionalAccessTest,
    ConstRefAccessorToOptionalViaConstRefAccessorToHoldingObjectWithAnotherConstCallAfterCheck) {
  ExpectDiagnosticsFor(R"cc(
      #include "unchecked_optional_access_test.h"

      struct A {
        const $ns::$optional<int>& get() const { return x; }

        $ns::$optional<int> x;
      };

      struct B {
        const A& getA() const { return a; }

        void callWithoutChanges() const { 
          // no-op 
        }

        A a;
      };

      void target(B& b) {  
        if (b.getA().get().has_value()) {
          b.callWithoutChanges(); // calling const method which cannot change A
          b.getA().get().value();
        }
      }
    )cc");
}

TEST_P(UncheckedOptionalAccessTest, ConstPointerRefAccessor) {
  // A crash reproducer for https://github.com/llvm/llvm-project/issues/125589
  // NOTE: we currently cache const ref accessors's locations.
  // If we want to support const ref to pointers or bools, we should initialize
  // their values.
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      $ns::$optional<int> x;
    };

    struct PtrWrapper {
      A*& getPtrRef() const;
      void reset(A*);
    };

    void target(PtrWrapper p) {
      if (p.getPtrRef()->x) {
        *p.getPtrRef()->x;  // [[unsafe]]
        p.reset(nullptr);
        *p.getPtrRef()->x;  // [[unsafe]]
      }
    }
  )cc",
                       /*IgnoreSmartPointerDereference=*/false);
}

TEST_P(UncheckedOptionalAccessTest, DiagnosticsHaveRanges) {
  ExpectDiagnosticsFor(R"cc(
    #include "unchecked_optional_access_test.h"

    struct A {
      $ns::$optional<int> fi;
    };
    struct B {
      $ns::$optional<A> fa;
    };

    void target($ns::$optional<B> opt) {
      opt.value();  // [[unsafe:<input.cc:12:7>]]
      if (opt) {
        opt  // [[unsafe:<input.cc:14:9, line:16:13>]]
          ->
            fa.value();
        if (opt->fa) {
          opt->fa->fi.value();  // [[unsafe:<input.cc:18:11, col:20>]]
        }
      }
    }
  )cc");
}

// FIXME: Add support for:
// - constructors (copy, move)
// - assignment operators (default, copy, move)
// - invalidation (passing optional by non-const reference/pointer)
