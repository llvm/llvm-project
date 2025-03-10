//===-- InlayHintTests.cpp  -------------------------------*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "Config.h"
#include "InlayHints.h"
#include "Protocol.h"
#include "TestTU.h"
#include "TestWorkspace.h"
#include "XRefs.h"
#include "support/Context.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {

llvm::raw_ostream &operator<<(llvm::raw_ostream &Stream,
                              const InlayHint &Hint) {
  return Stream << Hint.joinLabels() << "@" << Hint.range;
}

namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

std::vector<InlayHint> hintsOfKind(ParsedAST &AST, InlayHintKind Kind) {
  std::vector<InlayHint> Result;
  for (auto &Hint : inlayHints(AST, /*RestrictRange=*/std::nullopt)) {
    if (Hint.kind == Kind)
      Result.push_back(Hint);
  }
  return Result;
}

enum HintSide { Left, Right };

struct ExpectedHint {
  std::string Label;
  std::string RangeName;
  HintSide Side = Left;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &Stream,
                                       const ExpectedHint &Hint) {
    return Stream << Hint.Label << "@$" << Hint.RangeName;
  }
};

MATCHER_P2(HintMatcher, Expected, Code, llvm::to_string(Expected)) {
  llvm::StringRef ExpectedView(Expected.Label);
  std::string ResultLabel = arg.joinLabels();
  if (ResultLabel != ExpectedView.trim(" ") ||
      arg.paddingLeft != ExpectedView.starts_with(" ") ||
      arg.paddingRight != ExpectedView.ends_with(" ")) {
    *result_listener << "label is '" << ResultLabel << "'";
    return false;
  }
  if (arg.range != Code.range(Expected.RangeName)) {
    *result_listener << "range is " << llvm::to_string(arg.range) << " but $"
                     << Expected.RangeName << " is "
                     << llvm::to_string(Code.range(Expected.RangeName));
    return false;
  }
  return true;
}

MATCHER_P(labelIs, Label, "") { return arg.joinLabels() == Label; }

Config noHintsConfig() {
  Config C;
  C.InlayHints.Parameters = false;
  C.InlayHints.DeducedTypes = false;
  C.InlayHints.Designators = false;
  C.InlayHints.BlockEnd = false;
  C.InlayHints.DefaultArguments = false;
  return C;
}

template <typename... ExpectedHints>
void assertHintsWithHeader(InlayHintKind Kind, llvm::StringRef AnnotatedSource,
                           llvm::StringRef HeaderContent,
                           ExpectedHints... Expected) {
  Annotations Source(AnnotatedSource);
  TestTU TU = TestTU::withCode(Source.code());
  TU.ExtraArgs.push_back("-std=c++23");
  TU.HeaderCode = HeaderContent;
  auto AST = TU.build();

  EXPECT_THAT(hintsOfKind(AST, Kind),
              ElementsAre(HintMatcher(Expected, Source)...));
  // Sneak in a cross-cutting check that hints are disabled by config.
  // We'll hit an assertion failure if addInlayHint still gets called.
  WithContextValue WithCfg(Config::Key, noHintsConfig());
  EXPECT_THAT(inlayHints(AST, std::nullopt), IsEmpty());
}

template <typename... ExpectedHints>
void assertHints(InlayHintKind Kind, llvm::StringRef AnnotatedSource,
                 ExpectedHints... Expected) {
  return assertHintsWithHeader(Kind, AnnotatedSource, "",
                               std::move(Expected)...);
}

// Hack to allow expression-statements operating on parameter packs in C++14.
template <typename... T> void ignore(T &&...) {}

template <typename... ExpectedHints>
void assertParameterHints(llvm::StringRef AnnotatedSource,
                          ExpectedHints... Expected) {
  ignore(Expected.Side = Left...);
  assertHints(InlayHintKind::Parameter, AnnotatedSource, Expected...);
}

template <typename... ExpectedHints>
void assertTypeHints(llvm::StringRef AnnotatedSource,
                     ExpectedHints... Expected) {
  ignore(Expected.Side = Right...);
  assertHints(InlayHintKind::Type, AnnotatedSource, Expected...);
}

template <typename... ExpectedHints>
void assertDesignatorHints(llvm::StringRef AnnotatedSource,
                           ExpectedHints... Expected) {
  Config Cfg;
  Cfg.InlayHints.Designators = true;
  WithContextValue WithCfg(Config::Key, std::move(Cfg));
  assertHints(InlayHintKind::Designator, AnnotatedSource, Expected...);
}

template <typename... ExpectedHints>
void assertBlockEndHints(llvm::StringRef AnnotatedSource,
                         ExpectedHints... Expected) {
  Config Cfg;
  Cfg.InlayHints.BlockEnd = true;
  WithContextValue WithCfg(Config::Key, std::move(Cfg));
  assertHints(InlayHintKind::BlockEnd, AnnotatedSource, Expected...);
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

TEST(ParameterHints, NoNameConstReference) {
  // No hint for anonymous const l-value ref parameter.
  assertParameterHints(R"cpp(
    void foo(const int&);
    void bar() {
      foo(42);
    }
  )cpp");
}

TEST(ParameterHints, NoNameReference) {
  // Reference hint for anonymous l-value ref parameter.
  assertParameterHints(R"cpp(
    void foo(int&);
    void bar() {
      int i;
      foo($param[[i]]);
    }
  )cpp",
                       ExpectedHint{"&: ", "param"});
}

TEST(ParameterHints, NoNameRValueReference) {
  // No reference hint for anonymous r-value ref parameter.
  assertParameterHints(R"cpp(
    void foo(int&&);
    void bar() {
      foo(42);
    }
  )cpp");
}

TEST(ParameterHints, NoNameVariadicDeclaration) {
  // No hint for anonymous variadic parameter
  assertParameterHints(R"cpp(
    template <typename... Args>
    void foo(Args&& ...);
    void bar() {
      foo(42);
    }
  )cpp");
}

TEST(ParameterHints, NoNameVariadicForwarded) {
  // No hint for anonymous variadic parameter
  // This prototype of std::forward is sufficient for clang to recognize it
  assertParameterHints(R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    void foo(int);
    template <typename... Args>
    void bar(Args&&... args) { return foo(std::forward<Args>(args)...); }
    void baz() {
      bar(42);
    }
  )cpp");
}

TEST(ParameterHints, NoNameVariadicPlain) {
  // No hint for anonymous variadic parameter
  assertParameterHints(R"cpp(
    void foo(int);
    template <typename... Args>
    void bar(Args&&... args) { return foo(args...); }
    void baz() {
      bar(42);
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

TEST(ParameterHints, NamePartiallyInDefinition) {
  // Parameter name picked up from definition if necessary.
  assertParameterHints(R"cpp(
    void foo(int, int b);
    void bar() {
      foo($param1[[42]], $param2[[42]]);
    }
    void foo(int a, int) {};
  )cpp",
                       ExpectedHint{"a: ", "param1"},
                       ExpectedHint{"b: ", "param2"});
}

TEST(ParameterHints, NameInDefinitionVariadic) {
  // Parameter name picked up from definition in a resolved forwarded parameter.
  assertParameterHints(R"cpp(
    void foo(int, int);
    template <typename... Args>
    void bar(Args... args) {
      foo(args...);
    }
    void baz() {
      bar($param1[[42]], $param2[[42]]);
    }
    void foo(int a, int b) {};
  )cpp",
                       ExpectedHint{"a: ", "param1"},
                       ExpectedHint{"b: ", "param2"});
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

TEST(ParameterHints, NameConstReference) {
  // Only name hint for const l-value ref parameter.
  assertParameterHints(R"cpp(
    void foo(const int& param);
    void bar() {
      foo($param[[42]]);
    }
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, NameTypeAliasConstReference) {
  // Only name hint for const l-value ref parameter via type alias.
  assertParameterHints(R"cpp(
    using alias = const int&;
    void foo(alias param);
    void bar() {
      int i;
      foo($param[[i]]);
    }
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, NameReference) {
  // Reference and name hint for l-value ref parameter.
  assertParameterHints(R"cpp(
    void foo(int& param);
    void bar() {
      int i;
      foo($param[[i]]);
    }
  )cpp",
                       ExpectedHint{"&param: ", "param"});
}

TEST(ParameterHints, NameTypeAliasReference) {
  // Reference and name hint for l-value ref parameter via type alias.
  assertParameterHints(R"cpp(
    using alias = int&;
    void foo(alias param);
    void bar() {
      int i;
      foo($param[[i]]);
    }
  )cpp",
                       ExpectedHint{"&param: ", "param"});
}

TEST(ParameterHints, NameRValueReference) {
  // Only name hint for r-value ref parameter.
  assertParameterHints(R"cpp(
    void foo(int&& param);
    void bar() {
      foo($param[[42]]);
    }
  )cpp",
                       ExpectedHint{"param: ", "param"});
}

TEST(ParameterHints, VariadicForwardedConstructor) {
  // Name hint for variadic parameter using std::forward in a constructor call
  // This prototype of std::forward is sufficient for clang to recognize it
  assertParameterHints(R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    struct S { S(int a); };
    template <typename T, typename... Args>
    T bar(Args&&... args) { return T{std::forward<Args>(args)...}; }
    void baz() {
      int b;
      bar<S>($param[[b]]);
    }
  )cpp",
                       ExpectedHint{"a: ", "param"});
}

TEST(ParameterHints, VariadicPlainConstructor) {
  // Name hint for variadic parameter in a constructor call
  assertParameterHints(R"cpp(
    struct S { S(int a); };
    template <typename T, typename... Args>
    T bar(Args&&... args) { return T{args...}; }
    void baz() {
      int b;
      bar<S>($param[[b]]);
    }
  )cpp",
                       ExpectedHint{"a: ", "param"});
}

TEST(ParameterHints, VariadicForwardedNewConstructor) {
  // Name hint for variadic parameter using std::forward in a new expression
  // This prototype of std::forward is sufficient for clang to recognize it
  assertParameterHints(R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    struct S { S(int a); };
    template <typename T, typename... Args>
    T* bar(Args&&... args) { return new T{std::forward<Args>(args)...}; }
    void baz() {
      int b;
      bar<S>($param[[b]]);
    }
  )cpp",
                       ExpectedHint{"a: ", "param"});
}

TEST(ParameterHints, VariadicPlainNewConstructor) {
  // Name hint for variadic parameter in a new expression
  assertParameterHints(R"cpp(
    struct S { S(int a); };
    template <typename T, typename... Args>
    T* bar(Args&&... args) { return new T{args...}; }
    void baz() {
      int b;
      bar<S>($param[[b]]);
    }
  )cpp",
                       ExpectedHint{"a: ", "param"});
}

TEST(ParameterHints, VariadicForwarded) {
  // Name for variadic parameter using std::forward
  // This prototype of std::forward is sufficient for clang to recognize it
  assertParameterHints(R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    void foo(int a);
    template <typename... Args>
    void bar(Args&&... args) { return foo(std::forward<Args>(args)...); }
    void baz() {
      int b;
      bar($param[[b]]);
    }
  )cpp",
                       ExpectedHint{"a: ", "param"});
}

TEST(ParameterHints, VariadicPlain) {
  // Name hint for variadic parameter
  assertParameterHints(R"cpp(
    void foo(int a);
    template <typename... Args>
    void bar(Args&&... args) { return foo(args...); }
    void baz() {
      bar($param[[42]]);
    }
  )cpp",
                       ExpectedHint{"a: ", "param"});
}

TEST(ParameterHints, VariadicPlainWithPackFirst) {
  // Name hint for variadic parameter when the parameter pack is not the last
  // template parameter
  assertParameterHints(R"cpp(
    void foo(int a);
    template <typename... Args, typename Arg>
    void bar(Arg, Args&&... args) { return foo(args...); }
    void baz() {
      bar(1, $param[[42]]);
    }
  )cpp",
                       ExpectedHint{"a: ", "param"});
}

TEST(ParameterHints, VariadicSplitTwolevel) {
  // Name for variadic parameter that involves both head and tail parameters to
  // deal with.
  // This prototype of std::forward is sufficient for clang to recognize it
  assertParameterHints(R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    void baz(int, int b, double);
    template <typename... Args>
    void foo(int a, Args&&... args) {
      return baz(1, std::forward<Args>(args)..., 1.0);
    }
    template <typename... Args>
    void bar(Args&&... args) { return foo(std::forward<Args>(args)...); }
    void bazz() {
      bar($param1[[32]], $param2[[42]]);
    }
  )cpp",
                       ExpectedHint{"a: ", "param1"},
                       ExpectedHint{"b: ", "param2"});
}

TEST(ParameterHints, VariadicNameFromSpecialization) {
  // We don't try to resolve forwarding parameters if the function call uses a
  // specialization.
  assertParameterHints(R"cpp(
    void foo(int a);
    template <typename... Args>
    void bar(Args... args) {
      foo(args...);
    }
    template <>
    void bar<int>(int b);
    void baz() {
      bar($param[[42]]);
    }
  )cpp",
                       ExpectedHint{"b: ", "param"});
}

TEST(ParameterHints, VariadicNameFromSpecializationRecursive) {
  // We don't try to resolve forwarding parameters inside a forwarding function
  // call if that function call uses a specialization.
  assertParameterHints(R"cpp(
    void foo2(int a);
    template <typename... Args>
    void foo(Args... args) {
      foo2(args...);
    }
    template <typename... Args>
    void bar(Args... args) {
      foo(args...);
    }
    template <>
    void foo<int>(int b);
    void baz() {
      bar($param[[42]]);
    }
  )cpp",
                       ExpectedHint{"b: ", "param"});
}

TEST(ParameterHints, VariadicOverloaded) {
  // Name for variadic parameter for an overloaded function with unique number
  // of parameters.
  // This prototype of std::forward is sufficient for clang to recognize it
  assertParameterHints(
      R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    void baz(int b, int c);
    void baz(int bb, int cc, int dd);
    template <typename... Args>
    void foo(int a, Args&&... args) {
      return baz(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void bar(Args&&... args) { return foo(std::forward<Args>(args)...); }
    void bazz() {
      bar($param1[[32]], $param2[[42]], $param3[[52]]);
      bar($param4[[1]], $param5[[2]], $param6[[3]], $param7[[4]]);
    }
  )cpp",
      ExpectedHint{"a: ", "param1"}, ExpectedHint{"b: ", "param2"},
      ExpectedHint{"c: ", "param3"}, ExpectedHint{"a: ", "param4"},
      ExpectedHint{"bb: ", "param5"}, ExpectedHint{"cc: ", "param6"},
      ExpectedHint{"dd: ", "param7"});
}

TEST(ParameterHints, VariadicRecursive) {
  // make_tuple-like recursive variadic call
  assertParameterHints(
      R"cpp(
    void foo();

    template <typename Head, typename... Tail>
    void foo(Head head, Tail... tail) {
      foo(tail...);
    }

    template <typename... Args>
    void bar(Args... args) {
      foo(args...);
    }

    int main() {
      bar(1, 2, 3);
    }
  )cpp");
}

TEST(ParameterHints, VariadicVarargs) {
  // variadic call involving varargs (to make sure we don't crash)
  assertParameterHints(R"cpp(
    void foo(int fixed, ...);
    template <typename... Args>
    void bar(Args&&... args) {
      foo(args...);
    }

    void baz() {
      bar($fixed[[41]], 42, 43);
    }
  )cpp");
}

TEST(ParameterHints, VariadicTwolevelUnresolved) {
  // the same setting as VariadicVarargs, only with parameter pack
  assertParameterHints(R"cpp(
    template <typename... Args>
    void foo(int fixed, Args&& ... args);
    template <typename... Args>
    void bar(Args&&... args) {
      foo(args...);
    }

    void baz() {
      bar($fixed[[41]], 42, 43);
    }
  )cpp",
                       ExpectedHint{"fixed: ", "fixed"});
}

TEST(ParameterHints, VariadicTwoCalls) {
  // only the first call using the parameter pack should be picked up
  assertParameterHints(
      R"cpp(
    void f1(int a, int b);
    void f2(int c, int d);

    bool cond;

    template <typename... Args>
    void foo(Args... args) {
      if (cond) {
        f1(args...);
      } else {
        f2(args...);
      }
    }

    int main() {
      foo($param1[[1]], $param2[[2]]);
    }
  )cpp",
      ExpectedHint{"a: ", "param1"}, ExpectedHint{"b: ", "param2"});
}

TEST(ParameterHints, VariadicInfinite) {
  // infinite recursion should not break clangd
  assertParameterHints(
      R"cpp(
    template <typename... Args>
    void foo(Args...);

    template <typename... Args>
    void bar(Args... args) {
      foo(args...);
    }

    template <typename... Args>
    void foo(Args... args) {
      bar(args...);
    }

    int main() {
      foo(1, 2);
    }
  )cpp");
}

TEST(ParameterHints, VariadicDuplicatePack) {
  // edge cases with multiple adjacent packs should work
  assertParameterHints(
      R"cpp(
    void foo(int a, int b, int c, int);

    template <typename... Args>
    void bar(int, Args... args, int d) {
      foo(args..., d);
    }

    template <typename... Args>
    void baz(Args... args, Args... args2) {
      bar<Args..., int>(1, args..., args2...);
    }

    int main() {
      baz<int, int>($p1[[1]], $p2[[2]], $p3[[3]], $p4[[4]]);
    }
  )cpp",
      ExpectedHint{"a: ", "p1"}, ExpectedHint{"b: ", "p2"},
      ExpectedHint{"c: ", "p3"}, ExpectedHint{"d: ", "p4"});
}

TEST(ParameterHints, VariadicEmplace) {
  // emplace-like calls should forward constructor parameters
  // This prototype of std::forward is sufficient for clang to recognize it
  assertParameterHints(
      R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    using size_t = decltype(sizeof(0));
    void *operator new(size_t, void *);
    struct S {
      S(int A);
      S(int B, int C);
    };
    struct alloc {
      template <typename T>
      T* allocate();
      template <typename T, typename... Args>
      void construct(T* ptr, Args&&... args) {
        ::new ((void*)ptr) T{std::forward<Args>(args)...};
      }
    };
    template <typename T>
    struct container {
      template <typename... Args>
      void emplace(Args&&... args) {
        alloc a;
        auto ptr = a.template allocate<T>();
        a.construct(ptr, std::forward<Args>(args)...);
      }
    };
    void foo() {
      container<S> c;
      c.emplace($param1[[1]]);
      c.emplace($param2[[2]], $param3[[3]]);
    }
  )cpp",
      ExpectedHint{"A: ", "param1"}, ExpectedHint{"B: ", "param2"},
      ExpectedHint{"C: ", "param3"});
}

TEST(ParameterHints, VariadicReferenceHint) {
  assertParameterHints(R"cpp(
    void foo(int&);
    template <typename... Args>
    void bar(Args... args) { return foo(args...); }
    void baz() {
      int a;
      bar(a);
      bar(1);
    }
  )cpp");
}

TEST(ParameterHints, VariadicReferenceHintForwardingRef) {
  assertParameterHints(R"cpp(
    void foo(int&);
    template <typename... Args>
    void bar(Args&&... args) { return foo(args...); }
    void baz() {
      int a;
      bar($param[[a]]);
      bar(1);
    }
  )cpp",
                       ExpectedHint{"&: ", "param"});
}

TEST(ParameterHints, VariadicReferenceHintForwardingRefStdForward) {
  assertParameterHints(R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    void foo(int&);
    template <typename... Args>
    void bar(Args&&... args) { return foo(std::forward<Args>(args)...); }
    void baz() {
      int a;
      bar($param[[a]]);
    }
  )cpp",
                       ExpectedHint{"&: ", "param"});
}

TEST(ParameterHints, VariadicNoReferenceHintForwardingRefStdForward) {
  assertParameterHints(R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    void foo(int);
    template <typename... Args>
    void bar(Args&&... args) { return foo(std::forward<Args>(args)...); }
    void baz() {
      int a;
      bar(a);
      bar(1);
    }
  )cpp");
}

TEST(ParameterHints, VariadicNoReferenceHintUnresolvedForward) {
  assertParameterHints(R"cpp(
    template <typename... Args>
    void foo(Args&&... args);
    void bar() {
      int a;
      foo(a);
    }
  )cpp");
}

TEST(ParameterHints, MatchingNameVariadicForwarded) {
  // No name hint for variadic parameter with matching name
  // This prototype of std::forward is sufficient for clang to recognize it
  assertParameterHints(R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    void foo(int a);
    template <typename... Args>
    void bar(Args&&... args) { return foo(std::forward<Args>(args)...); }
    void baz() {
      int a;
      bar(a);
    }
  )cpp");
}

TEST(ParameterHints, MatchingNameVariadicPlain) {
  // No name hint for variadic parameter with matching name
  assertParameterHints(R"cpp(
    void foo(int a);
    template <typename... Args>
    void bar(Args&&... args) { return foo(args...); }
    void baz() {
      int a;
      bar(a);
    }
  )cpp");
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

TEST(ParameterHints, FunctionCallOperator) {
  assertParameterHints(R"cpp(
    struct W {
      void operator()(int x);
    };
    struct S : W {
      using W::operator();
      static void operator()(int x, int y);
    };
    void bar() {
      auto l1 = [](int x) {};
      auto l2 = [](int x) static {};

      S s;
      s($1[[1]]);
      s.operator()($2[[1]]);
      s.operator()($3[[1]], $4[[2]]);
      S::operator()($5[[1]], $6[[2]]);

      l1($7[[1]]);
      l1.operator()($8[[1]]);
      l2($9[[1]]);
      l2.operator()($10[[1]]);

      void (*ptr)(int a, int b) = &S::operator();
      ptr($11[[1]], $12[[2]]);
    }
  )cpp",
                       ExpectedHint{"x: ", "1"}, ExpectedHint{"x: ", "2"},
                       ExpectedHint{"x: ", "3"}, ExpectedHint{"y: ", "4"},
                       ExpectedHint{"x: ", "5"}, ExpectedHint{"y: ", "6"},
                       ExpectedHint{"x: ", "7"}, ExpectedHint{"x: ", "8"},
                       ExpectedHint{"x: ", "9"}, ExpectedHint{"x: ", "10"},
                       ExpectedHint{"a: ", "11"}, ExpectedHint{"b: ", "12"});
}

TEST(ParameterHints, DeducingThis) {
  assertParameterHints(R"cpp(
    struct S {
      template <typename This>
      auto operator()(this This &&Self, int Param) {
        return 42;
      }

      auto function(this auto &Self, int Param) {
        return Param;
      }
    };
    void work() {
      S s;
      s($1[[42]]);
      s.function($2[[42]]);
      S()($3[[42]]);
      auto lambda = [](this auto &Self, char C) -> void {
        return Self(C);
      };
      lambda($4[['A']]);
    }
  )cpp",
                       ExpectedHint{"Param: ", "1"},
                       ExpectedHint{"Param: ", "2"},
                       ExpectedHint{"Param: ", "3"}, ExpectedHint{"C: ", "4"});
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

  // If the macro expands to multiple arguments, don't hint it.
  assertParameterHints(R"cpp(
    void foo(double x, double y);
    #define CONSTANTS 3.14, 2.72
    void bar() {
      foo(CONSTANTS);
    }
  )cpp");
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
      template <typename E> class initializer_list { const E *a, *b; };
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

TEST(ParameterHints, FunctionPointer) {
  assertParameterHints(
      R"cpp(
    void (*f1)(int param);
    void (__stdcall *f2)(int param);
    using f3_t = void(*)(int param);
    f3_t f3;
    using f4_t = void(__stdcall *)(int param);
    f4_t f4;
    void bar() {
      f1($f1[[42]]);
      f2($f2[[42]]);
      f3($f3[[42]]);
      f4($f4[[42]]);
    }
  )cpp",
      ExpectedHint{"param: ", "f1"}, ExpectedHint{"param: ", "f2"},
      ExpectedHint{"param: ", "f3"}, ExpectedHint{"param: ", "f4"});
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

TEST(ParameterHints, ArgMatchesParamReference) {
  assertParameterHints(R"cpp(
    void foo(int& param);
    void foo2(const int& param);
    void bar() {
      int param;
      // show reference hint on mutable reference
      foo($param[[param]]);
      // but not on const reference
      foo2(param);
    }
  )cpp",
                       ExpectedHint{"&: ", "param"});
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
#define X 42
#define Y X
#define Z(...) Y
      foo(/*param=*/Z(a));
      foo($macro[[Z(a)]]);
      foo(/* the answer */$param[[42]]);
    }
  )cpp",
                       ExpectedHint{"param: ", "macro"},
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

TEST(ParameterHints, BuiltinFunctions) {
  // This prototype of std::forward is sufficient for clang to recognize it
  assertParameterHints(R"cpp(
    namespace std { template <typename T> T&& forward(T&); }
    void foo() {
      int i;
      std::forward(i);
    }
  )cpp");
}

TEST(ParameterHints, IncludeAtNonGlobalScope) {
  Annotations FooInc(R"cpp(
    void bar() { foo(42); }
  )cpp");
  Annotations FooCC(R"cpp(
    struct S {
      void foo(int param);
      #include "foo.inc"
    };
  )cpp");

  TestWorkspace Workspace;
  Workspace.addSource("foo.inc", FooInc.code());
  Workspace.addMainFile("foo.cc", FooCC.code());

  auto AST = Workspace.openFile("foo.cc");
  ASSERT_TRUE(bool(AST));

  // Ensure the hint for the call in foo.inc is NOT materialized in foo.cc.
  EXPECT_EQ(hintsOfKind(*AST, InlayHintKind::Parameter).size(), 0u);
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
                  ExpectedHint{": S1", "x"},
                  // FIXME: We want to suppress scope specifiers
                  //        here because we are into the whole
                  //        brevity thing, but the ElaboratedType
                  //        printer does not honor the SuppressScope
                  //        flag by design, so we need to extend the
                  //        PrintingPolicy to support this use case.
                  ExpectedHint{": S2::Inner<int>", "y"});
}

TEST(TypeHints, Lambda) {
  // Do not print something overly verbose like the lambda's location.
  // Show hints for init-captures (but not regular captures).
  assertTypeHints(R"cpp(
    void f() {
      int cap = 42;
      auto $L[[L]] = [cap, $init[[init]] = 1 + 1](int a$ret[[)]] { 
        return a + cap + init; 
      };
    }
  )cpp",
                  ExpectedHint{": (lambda)", "L"},
                  ExpectedHint{": int", "init"}, ExpectedHint{"-> int", "ret"});

  // Lambda return hint shown even if no param list.
  // (The digraph :> is just a ] that doesn't conflict with the annotations).
  assertTypeHints("auto $L[[x]] = <:$ret[[:>]]{return 42;};",
                  ExpectedHint{": (lambda)", "L"},
                  ExpectedHint{"-> int", "ret"});
}

// Structured bindings tests.
// Note, we hint the individual bindings, not the aggregate.

TEST(TypeHints, StructuredBindings_PublicStruct) {
  assertTypeHints(R"cpp(
    // Struct with public fields.
    struct Point {
      int x;
      int y;
    };
    Point foo();
    auto [$x[[x]], $y[[y]]] = foo();
  )cpp",
                  ExpectedHint{": int", "x"}, ExpectedHint{": int", "y"});
}

TEST(TypeHints, StructuredBindings_Array) {
  assertTypeHints(R"cpp(
    int arr[2];
    auto [$x[[x]], $y[[y]]] = arr;
  )cpp",
                  ExpectedHint{": int", "x"}, ExpectedHint{": int", "y"});
}

TEST(TypeHints, StructuredBindings_TupleLike) {
  assertTypeHints(R"cpp(
    // Tuple-like type.
    struct IntPair {
      int a;
      int b;
    };
    namespace std {
      template <typename T>
      struct tuple_size {};
      template <>
      struct tuple_size<IntPair> {
        constexpr static unsigned value = 2;
      };
      template <unsigned I, typename T>
      struct tuple_element {};
      template <unsigned I>
      struct tuple_element<I, IntPair> {
        using type = int;
      };
    }
    template <unsigned I>
    int get(const IntPair& p) {
      if constexpr (I == 0) {
        return p.a;
      } else if constexpr (I == 1) {
        return p.b;
      }
    }
    IntPair bar();
    auto [$x[[x]], $y[[y]]] = bar();
  )cpp",
                  ExpectedHint{": int", "x"}, ExpectedHint{": int", "y"});
}

TEST(TypeHints, StructuredBindings_NoInitializer) {
  assertTypeHints(R"cpp(
    // No initializer (ill-formed).
    // Do not show useless "NULL TYPE" hint.    
    auto [x, y];  /*error-ok*/
  )cpp");
}

TEST(TypeHints, InvalidType) {
  assertTypeHints(R"cpp(
    auto x = (unknown_type)42; /*error-ok*/
    auto *y = (unknown_ptr)nullptr;
  )cpp");
}

TEST(TypeHints, ReturnTypeDeduction) {
  assertTypeHints(
      R"cpp(
    auto f1(int x$ret1a[[)]];  // Hint forward declaration too
    auto f1(int x$ret1b[[)]] { return x + 1; }

    // Include pointer operators in hint
    int s;
    auto& f2($ret2[[)]] { return s; }

    // Do not hint `auto` for trailing return type.
    auto f3() -> int;

    // Do not hint when a trailing return type is specified.
    auto f4() -> auto* { return "foo"; }

    auto f5($noreturn[[)]] {}

    // `auto` conversion operator
    struct A {
      operator auto($retConv[[)]] { return 42; }
    };

    // FIXME: Dependent types do not work yet.
    template <typename T>
    struct S {
      auto method() { return T(); }
    };
  )cpp",
      ExpectedHint{"-> int", "ret1a"}, ExpectedHint{"-> int", "ret1b"},
      ExpectedHint{"-> int &", "ret2"}, ExpectedHint{"-> void", "noreturn"},
      ExpectedHint{"-> int", "retConv"});
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

    template <typename T>
    void bar(T arg) {
      auto [a, b] = arg;
    }
  )cpp");
}

TEST(TypeHints, LongTypeName) {
  assertTypeHints(R"cpp(
    template <typename, typename, typename>
    struct A {};
    struct MultipleWords {};
    A<MultipleWords, MultipleWords, MultipleWords> foo();
    // Omit type hint past a certain length (currently 32)
    auto var = foo();
  )cpp");

  Config Cfg;
  Cfg.InlayHints.TypeNameLimit = 0;
  WithContextValue WithCfg(Config::Key, std::move(Cfg));

  assertTypeHints(
      R"cpp(
    template <typename, typename, typename>
    struct A {};
    struct MultipleWords {};
    A<MultipleWords, MultipleWords, MultipleWords> foo();
    // Should have type hint with TypeNameLimit = 0
    auto $var[[var]] = foo();
  )cpp",
      ExpectedHint{": A<MultipleWords, MultipleWords, MultipleWords>", "var"});
}

TEST(TypeHints, DefaultTemplateArgs) {
  assertTypeHints(R"cpp(
    template <typename, typename = int>
    struct A {};
    A<float> foo();
    auto $var[[var]] = foo();
    A<float> bar[1];
    auto [$binding[[value]]] = bar;
  )cpp",
                  ExpectedHint{": A<float>", "var"},
                  ExpectedHint{": A<float>", "binding"});
}

TEST(DefaultArguments, Smoke) {
  Config Cfg;
  Cfg.InlayHints.Parameters =
      true; // To test interplay of parameters and default parameters
  Cfg.InlayHints.DeducedTypes = false;
  Cfg.InlayHints.Designators = false;
  Cfg.InlayHints.BlockEnd = false;

  Cfg.InlayHints.DefaultArguments = true;
  WithContextValue WithCfg(Config::Key, std::move(Cfg));

  const auto *Code = R"cpp(
    int foo(int A = 4) { return A; }
    int bar(int A, int B = 1, bool C = foo($default1[[)]]) { return A; }
    int A = bar($explicit[[2]]$default2[[)]];

    void baz(int = 5) { if (false) baz($unnamed[[)]]; };
  )cpp";

  assertHints(InlayHintKind::DefaultArgument, Code,
              ExpectedHint{"A: 4", "default1", Left},
              ExpectedHint{", B: 1, C: foo()", "default2", Left},
              ExpectedHint{"5", "unnamed", Left});

  assertHints(InlayHintKind::Parameter, Code,
              ExpectedHint{"A: ", "explicit", Left});
}

TEST(DefaultArguments, WithoutParameterNames) {
  Config Cfg;
  Cfg.InlayHints.Parameters = false; // To test just default args this time
  Cfg.InlayHints.DeducedTypes = false;
  Cfg.InlayHints.Designators = false;
  Cfg.InlayHints.BlockEnd = false;

  Cfg.InlayHints.DefaultArguments = true;
  WithContextValue WithCfg(Config::Key, std::move(Cfg));

  const auto *Code = R"cpp(
    struct Baz {
      Baz(float a = 3 //
                    + 2);
    };
    struct Foo {
      Foo(int, Baz baz = //
              Baz{$abbreviated[[}]]

          //
      ) {}
    };

    int main() {
      Foo foo1(1$paren[[)]];
      Foo foo2{2$brace1[[}]];
      Foo foo3 = {3$brace2[[}]];
      auto foo4 = Foo{4$brace3[[}]];
    }
  )cpp";

  assertHints(InlayHintKind::DefaultArgument, Code,
              ExpectedHint{"...", "abbreviated", Left},
              ExpectedHint{", Baz{}", "paren", Left},
              ExpectedHint{", Baz{}", "brace1", Left},
              ExpectedHint{", Baz{}", "brace2", Left},
              ExpectedHint{", Baz{}", "brace3", Left});

  assertHints(InlayHintKind::Parameter, Code);
}

TEST(TypeHints, Deduplication) {
  assertTypeHints(R"cpp(
    template <typename T>
    void foo() {
      auto $var[[var]] = 42;
    }
    template void foo<int>();
    template void foo<float>();
  )cpp",
                  ExpectedHint{": int", "var"});
}

TEST(TypeHints, SinglyInstantiatedTemplate) {
  assertTypeHints(R"cpp(
    auto $lambda[[x]] = [](auto *$param[[y]], auto) { return 42; };
    int m = x("foo", 3);
  )cpp",
                  ExpectedHint{": (lambda)", "lambda"},
                  ExpectedHint{": const char *", "param"});

  // No hint for packs, or auto params following packs
  assertTypeHints(R"cpp(
    int x(auto $a[[a]], auto... b, auto c) { return 42; }
    int m = x<void*, char, float>(nullptr, 'c', 2.0, 2);
  )cpp",
                  ExpectedHint{": void *", "a"});
}

TEST(TypeHints, Aliased) {
  // Check that we don't crash for functions without a FunctionTypeLoc.
  // https://github.com/clangd/clangd/issues/1140
  TestTU TU = TestTU::withCode("void foo(void){} extern typeof(foo) foo;");
  TU.ExtraArgs.push_back("-xc");
  auto AST = TU.build();

  EXPECT_THAT(hintsOfKind(AST, InlayHintKind::Type), IsEmpty());
}

TEST(TypeHints, CallingConvention) {
  // Check that we don't crash for lambdas without a FunctionTypeLoc
  // https://github.com/clangd/clangd/issues/2223
  std::string Code = R"cpp(
    void test() {
      []() __cdecl {};
    }
  )cpp";
  TestTU TU = TestTU::withCode(Code);
  TU.ExtraArgs.push_back("--target=x86_64-w64-mingw32");
  TU.PredefineMacros = true; // for the __cdecl
  auto AST = TU.build();

  EXPECT_THAT(hintsOfKind(AST, InlayHintKind::Type), IsEmpty());
}

TEST(TypeHints, Decltype) {
  assertTypeHints(R"cpp(
    $a[[decltype(0)]] a;
    $b[[decltype(a)]] b;
    const $c[[decltype(0)]] &c = b;

    // Don't show for dependent type
    template <class T>
    constexpr decltype(T{}) d;

    $e[[decltype(0)]] e();
    auto f() -> $f[[decltype(0)]];

    template <class, class> struct Foo;
    using G = Foo<$g[[decltype(0)]], float>;

    auto $h[[h]] = $i[[decltype(0)]]{};

    // No crash
    /* error-ok */
    auto $j[[s]];
  )cpp",
                  ExpectedHint{": int", "a"}, ExpectedHint{": int", "b"},
                  ExpectedHint{": int", "c"}, ExpectedHint{": int", "e"},
                  ExpectedHint{": int", "f"}, ExpectedHint{": int", "g"},
                  ExpectedHint{": int", "h"}, ExpectedHint{": int", "i"});
}

TEST(TypeHints, SubstTemplateParameterAliases) {
  llvm::StringRef Header = R"cpp(
  template <class T> struct allocator {};

  template <class T, class A>
  struct vector_base {
    using pointer = T*;
  };

  template <class T, class A>
  struct internal_iterator_type_template_we_dont_expect {};

  struct my_iterator {};

  template <class T, class A = allocator<T>>
  struct vector : vector_base<T, A> {
    using base = vector_base<T, A>;
    typedef T value_type;
    typedef base::pointer pointer;
    using allocator_type = A;
    using size_type = int;
    using iterator = internal_iterator_type_template_we_dont_expect<T, A>;
    using non_template_iterator = my_iterator;

    value_type& operator[](int index) { return elements[index]; }
    const value_type& at(int index) const { return elements[index]; }
    pointer data() { return &elements[0]; }
    allocator_type get_allocator() { return A(); }
    size_type size() const { return 10; }
    iterator begin() { return iterator(); }
    non_template_iterator end() { return non_template_iterator(); }

    T elements[10];
  };
  )cpp";

  llvm::StringRef VectorIntPtr = R"cpp(
    vector<int *> array;
    auto $no_modifier[[x]] = array[3];
    auto* $ptr_modifier[[ptr]] = &array[3];
    auto& $ref_modifier[[ref]] = array[3];
    auto& $at[[immutable]] = array.at(3);

    auto $data[[data]] = array.data();
    auto $allocator[[alloc]] = array.get_allocator();
    auto $size[[size]] = array.size();
    auto $begin[[begin]] = array.begin();
    auto $end[[end]] = array.end();
  )cpp";

  assertHintsWithHeader(
      InlayHintKind::Type, VectorIntPtr, Header,
      ExpectedHint{": int *", "no_modifier"},
      ExpectedHint{": int **", "ptr_modifier"},
      ExpectedHint{": int *&", "ref_modifier"},
      ExpectedHint{": int *const &", "at"}, ExpectedHint{": int **", "data"},
      ExpectedHint{": allocator<int *>", "allocator"},
      ExpectedHint{": size_type", "size"}, ExpectedHint{": iterator", "begin"},
      ExpectedHint{": non_template_iterator", "end"});

  llvm::StringRef VectorInt = R"cpp(
  vector<int> array;
  auto $no_modifier[[by_value]] = array[3];
  auto* $ptr_modifier[[ptr]] = &array[3];
  auto& $ref_modifier[[ref]] = array[3];
  auto& $at[[immutable]] = array.at(3);

  auto $data[[data]] = array.data();
  auto $allocator[[alloc]] = array.get_allocator();
  auto $size[[size]] = array.size();
  auto $begin[[begin]] = array.begin();
  auto $end[[end]] = array.end();
  )cpp";

  assertHintsWithHeader(
      InlayHintKind::Type, VectorInt, Header,
      ExpectedHint{": int", "no_modifier"},
      ExpectedHint{": int *", "ptr_modifier"},
      ExpectedHint{": int &", "ref_modifier"},
      ExpectedHint{": const int &", "at"}, ExpectedHint{": int *", "data"},
      ExpectedHint{": allocator<int>", "allocator"},
      ExpectedHint{": size_type", "size"}, ExpectedHint{": iterator", "begin"},
      ExpectedHint{": non_template_iterator", "end"});

  llvm::StringRef TypeAlias = R"cpp(
  // If the type alias is not of substituted template parameter type,
  // do not show desugared type.
  using VeryLongLongTypeName = my_iterator;
  using Short = VeryLongLongTypeName;

  auto $short_name[[my_value]] = Short();

  // Same applies with templates.
  template <typename T, typename A>
  using basic_static_vector = vector<T, A>;
  template <typename T>
  using static_vector = basic_static_vector<T, allocator<T>>;

  auto $vector_name[[vec]] = static_vector<int>();
  )cpp";

  assertHintsWithHeader(InlayHintKind::Type, TypeAlias, Header,
                        ExpectedHint{": Short", "short_name"},
                        ExpectedHint{": static_vector<int>", "vector_name"});
}

TEST(DesignatorHints, Basic) {
  assertDesignatorHints(R"cpp(
    struct S { int x, y, z; };
    S s {$x[[1]], $y[[2+2]]};

    int x[] = {$0[[0]], $1[[1]]};
  )cpp",
                        ExpectedHint{".x=", "x"}, ExpectedHint{".y=", "y"},
                        ExpectedHint{"[0]=", "0"}, ExpectedHint{"[1]=", "1"});
}

TEST(DesignatorHints, Nested) {
  assertDesignatorHints(R"cpp(
    struct Inner { int x, y; };
    struct Outer { Inner a, b; };
    Outer o{ $a[[{ $x[[1]], $y[[2]] }]], $bx[[3]] };
  )cpp",
                        ExpectedHint{".a=", "a"}, ExpectedHint{".x=", "x"},
                        ExpectedHint{".y=", "y"}, ExpectedHint{".b.x=", "bx"});
}

TEST(DesignatorHints, AnonymousRecord) {
  assertDesignatorHints(R"cpp(
    struct S {
      union {
        struct {
          struct {
            int y;
          };
        } x;
      };
    };
    S s{$xy[[42]]};
  )cpp",
                        ExpectedHint{".x.y=", "xy"});
}

TEST(DesignatorHints, Suppression) {
  assertDesignatorHints(R"cpp(
    struct Point { int a, b, c, d, e, f, g, h; };
    Point p{/*a=*/1, .c=2, /* .d = */3, $e[[4]]};
  )cpp",
                        ExpectedHint{".e=", "e"});
}

TEST(DesignatorHints, StdArray) {
  // Designators for std::array should be [0] rather than .__elements[0].
  // While technically correct, the designator is useless and horrible to read.
  assertDesignatorHints(R"cpp(
    template <typename T, int N> struct Array { T __elements[N]; };
    Array<int, 2> x = {$0[[0]], $1[[1]]};
  )cpp",
                        ExpectedHint{"[0]=", "0"}, ExpectedHint{"[1]=", "1"});
}

TEST(DesignatorHints, OnlyAggregateInit) {
  assertDesignatorHints(R"cpp(
    struct Copyable { int x; } c;
    Copyable d{c};

    struct Constructible { Constructible(int x); };
    Constructible x{42};
  )cpp" /*no designator hints expected (but param hints!)*/);
}

TEST(DesignatorHints, NoCrash) {
  assertDesignatorHints(R"cpp(
    /*error-ok*/
    struct A {};
    struct Foo {int a; int b;};
    void test() {
      Foo f{A(), $b[[1]]};
    }
  )cpp",
                        ExpectedHint{".b=", "b"});
}

TEST(InlayHints, RestrictRange) {
  Annotations Code(R"cpp(
    auto a = false;
    [[auto b = 1;
    auto c = '2';]]
    auto d = 3.f;
  )cpp");
  auto AST = TestTU::withCode(Code.code()).build();
  EXPECT_THAT(inlayHints(AST, Code.range()),
              ElementsAre(labelIs(": int"), labelIs(": char")));
}

TEST(ParameterHints, PseudoObjectExpr) {
  Annotations Code(R"cpp(
    struct S {
      __declspec(property(get=GetX, put=PutX)) int x[];
      int GetX(int y, int z) { return 42 + y; }
      void PutX(int) { }

      // This is a PseudoObjectExpression whose syntactic form is a binary
      // operator.
      void Work(int y) { x = y; } // Not `x = y: y`.
    };

    int printf(const char *Format, ...);

    int main() {
      S s;
      __builtin_dump_struct(&s, printf); // Not `Format: __builtin_dump_struct()`
      printf($Param[["Hello, %d"]], 42); // Normal calls are not affected.
      // This builds a PseudoObjectExpr, but here it's useful for showing the
      // arguments from the semantic form.
      return s.x[ $one[[1]] ][ $two[[2]] ]; // `x[y: 1][z: 2]`
    }
  )cpp");
  auto TU = TestTU::withCode(Code.code());
  TU.ExtraArgs.push_back("-fms-extensions");
  auto AST = TU.build();
  EXPECT_THAT(inlayHints(AST, std::nullopt),
              ElementsAre(HintMatcher(ExpectedHint{"Format: ", "Param"}, Code),
                          HintMatcher(ExpectedHint{"y: ", "one"}, Code),
                          HintMatcher(ExpectedHint{"z: ", "two"}, Code)));
}

TEST(ParameterHints, ArgPacksAndConstructors) {
  assertParameterHints(
      R"cpp(
    struct Foo{ Foo(); Foo(int x); };
    void foo(Foo a, int b);
    template <typename... Args>
    void bar(Args... args) {
      foo(args...);
    }
    template <typename... Args>
    void baz(Args... args) { foo($param1[[Foo{args...}]], $param2[[1]]); }

    template <typename... Args>
    void bax(Args... args) { foo($param3[[{args...}]], args...); }

    void foo() {
      bar($param4[[Foo{}]], $param5[[42]]);
      bar($param6[[42]], $param7[[42]]);
      baz($param8[[42]]);
      bax($param9[[42]]);
    }
  )cpp",
      ExpectedHint{"a: ", "param1"}, ExpectedHint{"b: ", "param2"},
      ExpectedHint{"a: ", "param3"}, ExpectedHint{"a: ", "param4"},
      ExpectedHint{"b: ", "param5"}, ExpectedHint{"a: ", "param6"},
      ExpectedHint{"b: ", "param7"}, ExpectedHint{"x: ", "param8"},
      ExpectedHint{"b: ", "param9"});
}

TEST(ParameterHints, DoesntExpandAllArgs) {
  assertParameterHints(
      R"cpp(
    void foo(int x, int y);
    int id(int a, int b, int c);
    template <typename... Args>
    void bar(Args... args) {
      foo(id($param1[[args]], $param2[[1]], $param3[[args]])...);
    }
    void foo() {
      bar(1, 2); // FIXME: We could have `bar(a: 1, a: 2)` here.
    }
  )cpp",
      ExpectedHint{"a: ", "param1"}, ExpectedHint{"b: ", "param2"},
      ExpectedHint{"c: ", "param3"});
}

TEST(BlockEndHints, Functions) {
  assertBlockEndHints(R"cpp(
    int foo() {
      return 41;
    $foo[[}]]

    template<int X> 
    int bar() { 
      // No hint for lambda for now
      auto f = []() { 
        return X; 
      };
      return f(); 
    $bar[[}]]

    // No hint because this isn't a definition
    int buz();

    struct S{};
    bool operator==(S, S) {
      return true;
    $opEqual[[}]]
  )cpp",
                      ExpectedHint{" // foo", "foo"},
                      ExpectedHint{" // bar", "bar"},
                      ExpectedHint{" // operator==", "opEqual"});
}

TEST(BlockEndHints, Methods) {
  assertBlockEndHints(R"cpp(
    struct Test {
      // No hint because there's no function body
      Test() = default;
      
      ~Test() {
      $dtor[[}]]

      void method1() {
      $method1[[}]]

      // No hint because this isn't a definition
      void method2();

      template <typename T>
      void method3() {
      $method3[[}]]

      // No hint because this isn't a definition
      template <typename T>
      void method4();

      Test operator+(int) const {
        return *this;
      $opIdentity[[}]]

      operator bool() const {
        return true;
      $opBool[[}]]

      // No hint because there's no function body
      operator int() const = delete;
    } x;

    void Test::method2() {
    $method2[[}]]

    template <typename T>
    void Test::method4() {
    $method4[[}]]
  )cpp",
                      ExpectedHint{" // ~Test", "dtor"},
                      ExpectedHint{" // method1", "method1"},
                      ExpectedHint{" // method3", "method3"},
                      ExpectedHint{" // operator+", "opIdentity"},
                      ExpectedHint{" // operator bool", "opBool"},
                      ExpectedHint{" // Test::method2", "method2"},
                      ExpectedHint{" // Test::method4", "method4"});
}

TEST(BlockEndHints, Namespaces) {
  assertBlockEndHints(
      R"cpp(
    namespace {
      void foo();
    $anon[[}]]

    namespace ns {
      void bar();
    $ns[[}]]
  )cpp",
      ExpectedHint{" // namespace", "anon"},
      ExpectedHint{" // namespace ns", "ns"});
}

TEST(BlockEndHints, Types) {
  assertBlockEndHints(
      R"cpp(
    struct S {
    $S[[};]]

    class C {
    $C[[};]]

    union U {
    $U[[};]]

    enum E1 {
    $E1[[};]]

    enum class E2 {
    $E2[[};]]
  )cpp",
      ExpectedHint{" // struct S", "S"}, ExpectedHint{" // class C", "C"},
      ExpectedHint{" // union U", "U"}, ExpectedHint{" // enum E1", "E1"},
      ExpectedHint{" // enum class E2", "E2"});
}

TEST(BlockEndHints, If) {
  assertBlockEndHints(
      R"cpp(
    void foo(bool cond) {
       if (cond)
          ;

       if (cond) {
       $simple[[}]]

       if (cond) {
       } else {
       $ifelse[[}]]

       if (cond) {
       } else if (!cond) {
       $elseif[[}]]

       if (cond) {
       } else {
         if (!cond) {
         $inner[[}]]
       $outer[[}]]

       if (auto X = cond) {
       $init[[}]]

       if (int i = 0; i > 10) {
       $init_cond[[}]]
    } // suppress
  )cpp",
      ExpectedHint{" // if cond", "simple"},
      ExpectedHint{" // if cond", "ifelse"}, ExpectedHint{" // if", "elseif"},
      ExpectedHint{" // if !cond", "inner"},
      ExpectedHint{" // if cond", "outer"}, ExpectedHint{" // if X", "init"},
      ExpectedHint{" // if i > 10", "init_cond"});
}

TEST(BlockEndHints, Loops) {
  assertBlockEndHints(
      R"cpp(
    void foo() {
       while (true)
          ;

       while (true) {
       $while[[}]]

       do {
       } while (true);

       for (;true;) {
       $forcond[[}]]

       for (int I = 0; I < 10; ++I) {
       $forvar[[}]]

       int Vs[] = {1,2,3};
       for (auto V : Vs) {
       $foreach[[}]]
    } // suppress
  )cpp",
      ExpectedHint{" // while true", "while"},
      ExpectedHint{" // for true", "forcond"},
      ExpectedHint{" // for I", "forvar"},
      ExpectedHint{" // for V", "foreach"});
}

TEST(BlockEndHints, Switch) {
  assertBlockEndHints(
      R"cpp(
    void foo(int I) {
      switch (I) {
        case 0: break;
      $switch[[}]]
    } // suppress
  )cpp",
      ExpectedHint{" // switch I", "switch"});
}

TEST(BlockEndHints, PrintLiterals) {
  assertBlockEndHints(
      R"cpp(
    void foo() {
      while ("foo") {
      $string[[}]]

      while ("foo but this time it is very long") {
      $string_long[[}]]

      while (true) {
      $boolean[[}]]

      while (1) {
      $integer[[}]]

      while (1.5) {
      $float[[}]]
    } // suppress
  )cpp",
      ExpectedHint{" // while \"foo\"", "string"},
      ExpectedHint{" // while \"foo but...\"", "string_long"},
      ExpectedHint{" // while true", "boolean"},
      ExpectedHint{" // while 1", "integer"},
      ExpectedHint{" // while 1.5", "float"});
}

TEST(BlockEndHints, PrintRefs) {
  assertBlockEndHints(
      R"cpp(
    namespace ns {
      int Var;
      int func();
      struct S {
        int Field;
        int method() const;
      }; // suppress
    } // suppress
    void foo() {
      while (ns::Var) {
      $var[[}]]

      while (ns::func()) {
      $func[[}]]

      while (ns::S{}.Field) {
      $field[[}]]

      while (ns::S{}.method()) {
      $method[[}]]
    } // suppress
  )cpp",
      ExpectedHint{" // while Var", "var"},
      ExpectedHint{" // while func", "func"},
      ExpectedHint{" // while Field", "field"},
      ExpectedHint{" // while method", "method"});
}

TEST(BlockEndHints, PrintConversions) {
  assertBlockEndHints(
      R"cpp(
    struct S {
      S(int);
      S(int, int);
      explicit operator bool();
    }; // suppress
    void foo(int I) {
      while (float(I)) {
      $convert_primitive[[}]]

      while (S(I)) {
      $convert_class[[}]]

      while (S(I, I)) {
      $construct_class[[}]]
    } // suppress
  )cpp",
      ExpectedHint{" // while float", "convert_primitive"},
      ExpectedHint{" // while S", "convert_class"},
      ExpectedHint{" // while S", "construct_class"});
}

TEST(BlockEndHints, PrintOperators) {
  std::string AnnotatedCode = R"cpp(
    void foo(Integer I) {
      while(++I){
      $preinc[[}]]

      while(I++){
      $postinc[[}]]

      while(+(I + I)){
      $unary_complex[[}]]

      while(I < 0){
      $compare[[}]]

      while((I + I) < I){
      $lhs_complex[[}]]

      while(I < (I + I)){
      $rhs_complex[[}]]

      while((I + I) < (I + I)){
      $binary_complex[[}]]
    } // suppress
  )cpp";

  // We can't store shared expectations in a vector, assertHints uses varargs.
  auto AssertExpectedHints = [&](llvm::StringRef Code) {
    assertBlockEndHints(Code, ExpectedHint{" // while ++I", "preinc"},
                        ExpectedHint{" // while I++", "postinc"},
                        ExpectedHint{" // while", "unary_complex"},
                        ExpectedHint{" // while I < 0", "compare"},
                        ExpectedHint{" // while ... < I", "lhs_complex"},
                        ExpectedHint{" // while I < ...", "rhs_complex"},
                        ExpectedHint{" // while", "binary_complex"});
  };

  // First with built-in operators.
  AssertExpectedHints("using Integer = int;" + AnnotatedCode);
  // And now with overloading!
  AssertExpectedHints(R"cpp(
    struct Integer {
      explicit operator bool();
      Integer operator++();
      Integer operator++(int);
      Integer operator+(Integer);
      Integer operator+();
      bool operator<(Integer);
      bool operator<(int);
    }; // suppress
  )cpp" + AnnotatedCode);
}

TEST(BlockEndHints, TrailingSemicolon) {
  assertBlockEndHints(R"cpp(
    // The hint is placed after the trailing ';'
    struct S1 {
    $S1[[}  ;]]   

    // The hint is always placed in the same line with the closing '}'.
    // So in this case where ';' is missing, it is attached to '}'.
    struct S2 {
    $S2[[}]]

    ;

    // No hint because only one trailing ';' is allowed
    struct S3 {
    };;

    // No hint because trailing ';' is only allowed for class/struct/union/enum
    void foo() {
    };

    // Rare case, but yes we'll have a hint here.
    struct {
      int x;
    $anon[[}]]
    
    s2;
  )cpp",
                      ExpectedHint{" // struct S1", "S1"},
                      ExpectedHint{" // struct S2", "S2"},
                      ExpectedHint{" // struct", "anon"});
}

TEST(BlockEndHints, TrailingText) {
  assertBlockEndHints(R"cpp(
    struct S1 {
    $S1[[}      ;]]

    // No hint for S2 because of the trailing comment
    struct S2 {
    }; /* Put anything here */

    struct S3 {
      // No hint for S4 because of the trailing source code
      struct S4 {
      };$S3[[};]]

    // No hint for ns because of the trailing comment
    namespace ns {
    } // namespace ns
  )cpp",
                      ExpectedHint{" // struct S1", "S1"},
                      ExpectedHint{" // struct S3", "S3"});
}

TEST(BlockEndHints, Macro) {
  assertBlockEndHints(R"cpp(
    #define DECL_STRUCT(NAME) struct NAME {
    #define RBRACE }

    DECL_STRUCT(S1)
    $S1[[};]]

    // No hint because we require a '}'
    DECL_STRUCT(S2)
    RBRACE;
  )cpp",
                      ExpectedHint{" // struct S1", "S1"});
}

TEST(BlockEndHints, PointerToMemberFunction) {
  // Do not crash trying to summarize `a->*p`.
  assertBlockEndHints(R"cpp(
    class A {};
    using Predicate = bool(A::*)();
    void foo(A* a, Predicate p) {
      if ((a->*p)()) {
      $ptrmem[[}]]
    } // suppress
  )cpp",
                      ExpectedHint{" // if", "ptrmem"});
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
