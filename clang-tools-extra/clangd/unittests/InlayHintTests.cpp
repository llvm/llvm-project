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

std::ostream &operator<<(std::ostream &Stream, const InlayHint &Hint) {
  return Stream << Hint.label;
}

namespace {

using ::testing::UnorderedElementsAre;

std::vector<InlayHint> hintsOfKind(ParsedAST &AST, InlayHintKind Kind) {
  std::vector<InlayHint> Result;
  for (auto &Hint : inlayHints(AST)) {
    if (Hint.kind == Kind)
      Result.push_back(Hint);
  }
  return Result;
}

struct ExpectedHint {
  std::string Label;
  std::string RangeName;

  friend std::ostream &operator<<(std::ostream &Stream,
                                  const ExpectedHint &Hint) {
    return Stream << Hint.RangeName << ": " << Hint.Label;
  }
};

MATCHER_P2(HintMatcher, Expected, Code, "") {
  return arg.label == Expected.Label &&
         arg.range == Code.range(Expected.RangeName);
}

template <typename... ExpectedHints>
void assertHints(InlayHintKind Kind, llvm::StringRef AnnotatedSource,
                 ExpectedHints... Expected) {
  Annotations Source(AnnotatedSource);
  TestTU TU = TestTU::withCode(Source.code());
  TU.ExtraArgs.push_back("-std=c++14");
  auto AST = TU.build();

  EXPECT_THAT(hintsOfKind(AST, Kind),
              UnorderedElementsAre(HintMatcher(Expected, Source)...));
}

template <typename... ExpectedHints>
void assertParameterHints(llvm::StringRef AnnotatedSource,
                          ExpectedHints... Expected) {
  assertHints(InlayHintKind::ParameterHint, AnnotatedSource, Expected...);
}

template <typename... ExpectedHints>
void assertTypeHints(llvm::StringRef AnnotatedSource,
                     ExpectedHints... Expected) {
  assertHints(InlayHintKind::TypeHint, AnnotatedSource, Expected...);
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

TEST(ParameterHints, DependentCalls) {
  assertParameterHints(R"cpp(
    template <typename T>
    void nonmember(T par1);

    template <typename T>
    struct A {
      void member(T par2);
      static void static_member(T par3);
    };

    void overload(int anInt);
    void overload(double aDouble);

    template <typename T>
    struct S {
      void bar(A<T> a, T t) {
        nonmember($par1[[t]]);
        a.member($par2[[t]]);
        A<T>::static_member($par3[[t]]);
        // We don't want to arbitrarily pick between
        // "anInt" or "aDouble", so just show no hint.
        overload(T{});
      }
    };
  )cpp",
                       ExpectedHint{"par1: ", "par1"},
                       ExpectedHint{"par2: ", "par2"},
                       ExpectedHint{"par3: ", "par3"});
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

TEST(TypeHints, Smoke) {
  assertTypeHints(R"cpp(
    auto $waldo[[waldo]] = 42;
  )cpp",
                  ExpectedHint{": int", "waldo"});
}

TEST(TypeHints, Decorations) {
  assertTypeHints(R"cpp(
    int x = 42;
    auto* $var1[[var1]] = &x;
    auto&& $var2[[var2]] = x;
    const auto& $var3[[var3]] = x;
  )cpp",
                  ExpectedHint{": int *", "var1"},
                  ExpectedHint{": int &", "var2"},
                  ExpectedHint{": const int &", "var3"});
}

TEST(TypeHints, DecltypeAuto) {
  assertTypeHints(R"cpp(
    int x = 42;
    int& y = x;
    decltype(auto) $z[[z]] = y;
  )cpp",
                  ExpectedHint{": int &", "z"});
}

TEST(TypeHints, NoQualifiers) {
  assertTypeHints(R"cpp(
    namespace A {
      namespace B {
        struct S1 {};
        S1 foo();
        auto $x[[x]] = foo();

        struct S2 {
          template <typename T>
          struct Inner {};
        };
        S2::Inner<int> bar();
        auto $y[[y]] = bar();
      }
    }
  )cpp",
                  ExpectedHint{": S1", "x"}, ExpectedHint{": Inner<int>", "y"});
}

TEST(TypeHints, Lambda) {
  // Do not print something overly verbose like the lambda's location.
  // Show hints for init-captures (but not regular captures).
  assertTypeHints(R"cpp(
    void f() {
      int cap = 42;
      auto $L[[L]] = [cap, $init[[init]] = 1 + 1](int a) { 
        return a + cap + init; 
      };
    }
  )cpp",
                  ExpectedHint{": (lambda)", "L"},
                  ExpectedHint{": int", "init"});
}

TEST(TypeHints, StructuredBindings) {
  // FIXME: Not handled yet.
  // To handle it, we could print:
  //  - the aggregate type next to the 'auto', or
  //  - the individual types inside the brackets
  // The latter is probably more useful.
  assertTypeHints(R"cpp(
    struct Point {
      int x;
      int y;
    };
    Point foo();
    auto [x, y] = foo();
  )cpp");
}

TEST(TypeHints, ReturnTypeDeduction) {
  // FIXME: Not handled yet.
  // This test is currently here mostly because a naive implementation
  // might have us print something not super helpful like the function type.
  assertTypeHints(R"cpp(
    auto func(int x) {
      return x + 1;
    }
  )cpp");
}

TEST(TypeHints, DependentType) {
  assertTypeHints(R"cpp(
    template <typename T>
    void foo(T arg) {
      // The hint would just be "auto" and we can't do any better.
      auto var1 = arg.method();
      // FIXME: It would be nice to show "T" as the hint.
      auto $var2[[var2]] = arg;
    }
  )cpp");
}

// FIXME: Low-hanging fruit where we could omit a type hint:
//  - auto x = TypeName(...);
//  - auto x = (TypeName) (...);
//  - auto x = static_cast<TypeName>(...);  // and other built-in casts

// Annoyances for which a heuristic is not obvious:
//  - auto x = llvm::dyn_cast<LongTypeName>(y);  // and similar
//  - stdlib algos return unwieldy __normal_iterator<X*, ...> type
//    (For this one, perhaps we should omit type hints that start
//     with a double underscore.)

} // namespace
} // namespace clangd
} // namespace clang