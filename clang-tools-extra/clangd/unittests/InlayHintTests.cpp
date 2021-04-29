//===-- InlayHintTests.cpp  -------------------------------*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "InlayHints.h"
#include "Protocol.h"
#include "TestTU.h"
#include "XRefs.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::UnorderedElementsAre;

std::vector<InlayHint> parameterHints(ParsedAST &AST) {
  std::vector<InlayHint> Result;
  for (auto &Hint : inlayHints(AST)) {
    if (Hint.kind == InlayHintKind::ParameterHint)
      Result.push_back(Hint);
  }
  return Result;
}

struct ExpectedHint {
  std::string Label;
  std::string RangeName;
};

MATCHER_P2(HintMatcher, Expected, Code, "") {
  return arg.label == Expected.Label &&
         arg.range == Code.range(Expected.RangeName);
}

template <typename... ExpectedHints>
void assertParameterHints(llvm::StringRef AnnotatedSource,
                          ExpectedHints... Expected) {
  Annotations Source(AnnotatedSource);
  TestTU TU = TestTU::withCode(Source.code());
  TU.ExtraArgs.push_back("-std=c++11");
  auto AST = TU.build();

  EXPECT_THAT(parameterHints(AST),
              UnorderedElementsAre(HintMatcher(Expected, Source)...));
}

TEST(ParameterHints, Smoke) {
  assertParameterHints(R"cpp(
    void foo(int param);
    void bar() {
      foo($param[[42]]);
    }
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, NoName) {
  // No hint for anonymous parameter.
  assertParameterHints(R"cpp(
    void foo(int);
    void bar() {
      foo(42);
    }
  )cpp");
}

TEST(ParameterHints, NameInDefinition) {
  // Parameter name picked up from definition if necessary.
  assertParameterHints(R"cpp(
    void foo(int);
    void bar() {
      foo($param[[42]]);
    }
    void foo(int param) {};
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, NameMismatch) {
  // Prefer name from declaration.
  assertParameterHints(R"cpp(
    void foo(int good);
    void bar() {
      foo($good[[42]]);
    }
    void foo(int bad) {};
  )cpp",
                       ExpectedHint{"good: ", "good"});
}

TEST(ParameterHints, Operator) {
  // No hint for operator call with operator syntax.
  assertParameterHints(R"cpp(
    struct S {};
    void operator+(S lhs, S rhs);
    void bar() {
      S a, b;
      a + b;
    }
  )cpp");
}

TEST(ParameterHints, Macros) {
  // Handling of macros depends on where the call's argument list comes from.

  // If it comes from a macro definition, there's nothing to hint
  // at the invocation site.
  assertParameterHints(R"cpp(
    void foo(int param);
    #define ExpandsToCall() foo(42)
    void bar() {
      ExpandsToCall();
    }
  )cpp");

  // The argument expression being a macro invocation shouldn't interfere
  // with hinting.
  assertParameterHints(R"cpp(
    #define PI 3.14
    void foo(double param);
    void bar() {
      foo($param[[PI]]);
    }
  )cpp",
                       ExpectedHint{"param: ", "param"});

  // If the whole argument list comes from a macro parameter, hint it.
  assertParameterHints(R"cpp(
    void abort();
    #define ASSERT(expr) if (!expr) abort()
    int foo(int param);
    void bar() {
      ASSERT(foo($param[[42]]) == 0);
    }
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, ConstructorParens) {
  assertParameterHints(R"cpp(
    struct S {
      S(int param);
    };
    void bar() {
      S obj($param[[42]]);
    }
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, ConstructorBraces) {
  assertParameterHints(R"cpp(
    struct S {
      S(int param);
    };
    void bar() {
      S obj{$param[[42]]};
    }
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, ConstructorStdInitList) {
  // Do not show hints for std::initializer_list constructors.
  assertParameterHints(R"cpp(
    namespace std {
      template <typename> class initializer_list {};
    }
    struct S {
      S(std::initializer_list<int> param);
    };
    void bar() {
      S obj{42, 43};
    }
  )cpp");
}

TEST(ParameterHints, MemberInit) {
  assertParameterHints(R"cpp(
    struct S {
      S(int param);
    };
    struct T {
      S member;
      T() : member($param[[42]]) {}
    };
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, ImplicitConstructor) {
  assertParameterHints(R"cpp(
    struct S {
      S(int param);
    };
    void bar(S);
    S foo() {
      // Do not show hint for implicit constructor call in argument.
      bar(42);
      // Do not show hint for implicit constructor call in return.
      return 42;
    }
  )cpp");
}

TEST(ParameterHints, ArgMatchesParam) {
  assertParameterHints(R"cpp(
    void foo(int param);
    struct S {
      static const int param = 42;
    };
    void bar() {
      int param = 42;
      // Do not show redundant "param: param".
      foo(param);
      // But show it if the argument is qualified.
      foo($param[[S::param]]);
    }
    struct A {
      int param;
      void bar() {
        // Do not show "param: param" for member-expr.
        foo(param);
      }
    };
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, LeadingUnderscore) {
  assertParameterHints(R"cpp(
    void foo(int p1, int _p2, int __p3);
    void bar() {
      foo($p1[[41]], $p2[[42]], $p3[[43]]);
    }
  )cpp",
                       ExpectedHint{"p1: ", "p1"}, ExpectedHint{"p2: ", "p2"},
                       ExpectedHint{"p3: ", "p3"});
}

TEST(ParameterHints, DependentCall) {
  // FIXME: This doesn't currently produce a hint but should.
  assertParameterHints(R"cpp(
    template <typename T>
    void foo(T param);

    template <typename T>
    struct S {
      void bar(T par) {
        foo($param[[par]]);
      }
    };
  )cpp");
}

TEST(ParameterHints, VariadicFunction) {
  assertParameterHints(R"cpp(
    template <typename... T>
    void foo(int fixed, T... variadic);

    void bar() {
      foo($fixed[[41]], 42, 43);
    }
  )cpp",
                       ExpectedHint{"fixed: ", "fixed"});
}

TEST(ParameterHints, VarargsFunction) {
  assertParameterHints(R"cpp(
    void foo(int fixed, ...);

    void bar() { 
      foo($fixed[[41]], 42, 43);
    }
  )cpp",
                       ExpectedHint{"fixed: ", "fixed"});
}

TEST(ParameterHints, CopyOrMoveConstructor) {
  // Do not show hint for parameter of copy or move constructor.
  assertParameterHints(R"cpp(
    struct S {
      S();
      S(const S& other);
      S(S&& other);
    };
    void bar() {
      S a;
      S b(a);    // copy
      S c(S());  // move
    }
  )cpp");
}

TEST(ParameterHints, AggregateInit) {
  // FIXME: This is not implemented yet, but it would be a natural
  // extension to show member names as hints here.
  assertParameterHints(R"cpp(
    struct Point {
      int x;
      int y;
    };
    void bar() {
      Point p{41, 42};
    }
  )cpp");
}

TEST(ParameterHints, UserDefinedLiteral) {
  // Do not hint call to user-defined literal operator.
  assertParameterHints(R"cpp(
    long double operator"" _w(long double param);
    void bar() {
      1.2_w;
    }
  )cpp");
}

TEST(ParameterHints, ParamNameComment) {
  // Do not hint an argument which already has a comment
  // with the parameter name preceding it.
  assertParameterHints(R"cpp(
    void foo(int param);
    void bar() {
      foo(/*param*/42);
      foo( /* param = */ 42);
      foo(/* the answer */$param[[42]]);
    }
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, SetterFunctions) {
  assertParameterHints(R"cpp(
    struct S {
      void setParent(S* parent);
      void set_parent(S* parent);
      void setTimeout(int timeoutMillis);
      void setTimeoutMillis(int timeout_millis);
    };
    void bar() {
      S s;
      // Parameter name matches setter name - omit hint.
      s.setParent(nullptr);
      // Support snake_case
      s.set_parent(nullptr);
      // Parameter name may contain extra info - show hint.
      s.setTimeout($timeoutMillis[[120]]);
      // FIXME: Ideally we'd want to omit this.
      s.setTimeoutMillis($timeout_millis[[120]]);
    }
  )cpp",
                       ExpectedHint{"timeoutMillis: ", "timeoutMillis"},
                       ExpectedHint{"timeout_millis: ", "timeout_millis"});
}

} // namespace
} // namespace clangd
} // namespace clang
