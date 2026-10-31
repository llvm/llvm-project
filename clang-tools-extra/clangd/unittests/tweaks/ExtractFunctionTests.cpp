//===-- ExtractFunctionTests.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::StartsWith;

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(ExtractFunction);

TEST_F(ExtractFunctionTest, FunctionTest) {
  Context = Function;

  // Root statements should have common parent.
  EXPECT_EQ(apply("for(;;) [[1+2; 1+2;]]"), "unavailable");
  // Single expression-statements can be extracted.
  EXPECT_THAT(apply("int x = 0; [[x++;]]"), HasSubstr("extracted"));
  // We don't support extraction from lambdas.
  EXPECT_EQ(apply("auto lam = [](){ [[int x;]] }; "), "unavailable");
  // Partial statements aren't extracted.
  EXPECT_THAT(apply("int [[x = 0]];"), "unavailable");
  // FIXME: Support hoisting.
  EXPECT_THAT(apply(" [[int a = 5;]] a++; "), "unavailable");

  // Ensure that end of Zone and Beginning of PostZone being adjacent doesn't
  // lead to break being included in the extraction zone.
  EXPECT_THAT(apply("for(;;) { [[int x;]]break; }"), HasSubstr("extracted"));
  // A loop's initializer has its value discarded just like an ordinary
  // statement, so it remains extractable (unlike the condition; see
  // ControlFlowConditions below).
  EXPECT_THAT(apply(" for([[int i = 0;]];);"), HasSubstr("extracted"));
  // ...but if the declared name is used later (in the condition,
  // increment, or body), extraction is unavailable regardless -- not
  // because of any condition/init-specific logic, but because
  // requiresHoisting() (checked in ExtractFunction::prepare(), independent
  // of what kind of statement is being extracted) catches it.
  EXPECT_EQ(apply("void use(int); for([[int i = 0;]] i < 10; ++i) use(i);"),
            "unavailable");
  // Extract certain return
  EXPECT_THAT(apply(" if(true) [[{ return; }]] "), HasSubstr("extracted"));
  // Don't extract uncertain return
  EXPECT_THAT(apply(" if(true) [[if (false) return;]] "),
              StartsWith("unavailable"));
  EXPECT_THAT(
      apply("#define RETURN_IF_ERROR(x) if (x) return\nRETU^RN_IF_ERROR(4);"),
      StartsWith("unavailable"));
}

TEST_F(ExtractFunctionTest, FileTest) {
  // Check all parameters are in order. `a` and `ptr` are mutated in the
  // zone (`+=` and postfix `++` respectively), so stay non-const; `b` is
  // an unmutated scalar, so becomes a by-value parameter; `foo` is an
  // unmutated class type, so becomes a const reference.
  std::string ParameterCheckInput = R"cpp(
struct Foo {
  int x;
};
void f(int a) {
  int b;
  int *ptr = &a;
  Foo foo;
  [[a += foo.x + b;
  *ptr++;]]
})cpp";
  std::string ParameterCheckOutput = R"cpp(
struct Foo {
  int x;
};
void extracted(int &a, int b, int *&ptr, const Foo &foo) {
a += foo.x + b;
  *ptr++;
}
void f(int a) {
  int b;
  int *ptr = &a;
  Foo foo;
  extracted(a, b, ptr, foo);
})cpp";
  EXPECT_EQ(apply(ParameterCheckInput), ParameterCheckOutput);

  // Check const qualifier
  std::string ConstCheckInput = R"cpp(
void f(const int c) {
  [[while(c) {}]]
})cpp";
  std::string ConstCheckOutput = R"cpp(
void extracted(const int c) {
while(c) {}
}
void f(const int c) {
  extracted(c);
})cpp";
  EXPECT_EQ(apply(ConstCheckInput), ConstCheckOutput);

  // Check const qualifier: kept for a by-reference non-scalar.
  std::string ConstNamespaceCheckInput = R"cpp(
namespace X { struct Y { int z; }; }
int f(const X::Y &y) {
  [[return y.z + y.z;]]
})cpp";
  std::string ConstNamespaceCheckOutput = R"cpp(
namespace X { struct Y { int z; }; }
int extracted(const X::Y &y) {
return y.z + y.z;
}
int f(const X::Y &y) {
  return extracted(y);
})cpp";
  EXPECT_EQ(apply(ConstNamespaceCheckInput), ConstNamespaceCheckOutput);

  // Don't extract when we need to make a function as a parameter.
  EXPECT_THAT(apply("void f() { [[int a; f();]] }"), StartsWith("fail"));

  std::string MethodInput = R"cpp(
    class T {
      void f() {
        [[int x;]]
      }
    };
  )cpp";
  std::string MethodCheckOutput = R"cpp(
    class T {
      void extracted() {
int x;
}
void f() {
        extracted();
      }
    };
  )cpp";
  EXPECT_EQ(apply(MethodInput), MethodCheckOutput);

  std::string OutOfLineMethodInput = R"cpp(
    class T {
      void f();
    };

    void T::f() {
      [[int x;]]
    }
  )cpp";
  std::string OutOfLineMethodCheckOutput = R"cpp(
    class T {
      void extracted();
void f();
    };

    void T::extracted() {
int x;
}
void T::f() {
      extracted();
    }
  )cpp";
  EXPECT_EQ(apply(OutOfLineMethodInput), OutOfLineMethodCheckOutput);

  // We don't extract from templated functions for now as templates are hard
  // to deal with.
  std::string TemplateFailInput = R"cpp(
    template<typename T>
    void f() {
      [[int x;]]
    }
  )cpp";
  EXPECT_EQ(apply(TemplateFailInput), "unavailable");

  std::string MacroInput = R"cpp(
    #define F(BODY) void f() { BODY }
    F ([[int x = 0;]])
  )cpp";
  std::string MacroOutput = R"cpp(
    #define F(BODY) void f() { BODY }
    void extracted() {
int x = 0;
}
F (extracted();)
  )cpp";
  EXPECT_EQ(apply(MacroInput), MacroOutput);

  // Shouldn't crash.
  EXPECT_EQ(apply("void f([[int a]]);"), "unavailable");
  EXPECT_EQ(apply("void f(int a = [[1]]);"), "unavailable");
  // Don't extract if we select the entire function body (CompoundStmt).
  std::string CompoundFailInput = R"cpp(
    void f() [[{
      int a;
    }]]
  )cpp";
  EXPECT_EQ(apply(CompoundFailInput), "unavailable");

  ExtraArgs.push_back("-std=c++14");
  // A bare expression-statement can be extracted (the semicolon isn't part
  // of the selection either way, since it isn't owned by any AST node).
  EXPECT_THAT(apply(R"cpp(
                void call() { [[1+1]]; }
            )cpp"),
              HasSubstr("extracted"));
  EXPECT_THAT(apply(R"cpp(
                void call() { [[1+1;]] }
            )cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, DifferentHeaderSourceTest) {
  Header = R"cpp(
    class SomeClass {
      void f();
    };
  )cpp";

  std::string OutOfLineSource = R"cpp(
    void SomeClass::f() {
      [[int x;]]
    }
  )cpp";

  std::string OutOfLineSourceOutputCheck = R"cpp(
    void SomeClass::extracted() {
int x;
}
void SomeClass::f() {
      extracted();
    }
  )cpp";

  std::string HeaderOutputCheck = R"cpp(
    class SomeClass {
      void extracted();
void f();
    };
  )cpp";

  llvm::StringMap<std::string> EditedFiles;

  EXPECT_EQ(apply(OutOfLineSource, &EditedFiles), OutOfLineSourceOutputCheck);
  EXPECT_EQ(EditedFiles.begin()->second, HeaderOutputCheck);
}

TEST_F(ExtractFunctionTest, DifferentFilesNestedTest) {
  Header = R"cpp(
    class T {
    class SomeClass {
      void f();
    };
    };
  )cpp";

  std::string NestedOutOfLineSource = R"cpp(
    void T::SomeClass::f() {
      [[int x;]]
    }
  )cpp";

  std::string NestedOutOfLineSourceOutputCheck = R"cpp(
    void T::SomeClass::extracted() {
int x;
}
void T::SomeClass::f() {
      extracted();
    }
  )cpp";

  std::string NestedHeaderOutputCheck = R"cpp(
    class T {
    class SomeClass {
      void extracted();
void f();
    };
    };
  )cpp";

  llvm::StringMap<std::string> EditedFiles;

  EXPECT_EQ(apply(NestedOutOfLineSource, &EditedFiles),
            NestedOutOfLineSourceOutputCheck);
  EXPECT_EQ(EditedFiles.begin()->second, NestedHeaderOutputCheck);
}

TEST_F(ExtractFunctionTest, ConstexprDifferentHeaderSourceTest) {
  Header = R"cpp(
    class SomeClass {
      constexpr void f() const;
    };
  )cpp";

  std::string OutOfLineSource = R"cpp(
    constexpr void SomeClass::f() const {
      [[int x;]]
    }
  )cpp";

  std::string OutOfLineSourceOutputCheck = R"cpp(
    constexpr void SomeClass::extracted() const {
int x;
}
constexpr void SomeClass::f() const {
      extracted();
    }
  )cpp";

  std::string HeaderOutputCheck = R"cpp(
    class SomeClass {
      constexpr void extracted() const;
constexpr void f() const;
    };
  )cpp";

  llvm::StringMap<std::string> EditedFiles;

  EXPECT_EQ(apply(OutOfLineSource, &EditedFiles), OutOfLineSourceOutputCheck);
  EXPECT_NE(EditedFiles.begin(), EditedFiles.end())
      << "The header should be edited and receives the declaration of the new "
         "function";

  if (EditedFiles.begin() != EditedFiles.end()) {
    EXPECT_EQ(EditedFiles.begin()->second, HeaderOutputCheck);
  }
}

TEST_F(ExtractFunctionTest, ConstevalDifferentHeaderSourceTest) {
  ExtraArgs.push_back("--std=c++20");
  Header = R"cpp(
    class SomeClass {
      consteval void f() const;
    };
  )cpp";

  std::string OutOfLineSource = R"cpp(
    consteval void SomeClass::f() const {
      [[int x;]]
    }
  )cpp";

  std::string OutOfLineSourceOutputCheck = R"cpp(
    consteval void SomeClass::extracted() const {
int x;
}
consteval void SomeClass::f() const {
      extracted();
    }
  )cpp";

  std::string HeaderOutputCheck = R"cpp(
    class SomeClass {
      consteval void extracted() const;
consteval void f() const;
    };
  )cpp";

  llvm::StringMap<std::string> EditedFiles;

  EXPECT_EQ(apply(OutOfLineSource, &EditedFiles), OutOfLineSourceOutputCheck);
  EXPECT_NE(EditedFiles.begin(), EditedFiles.end())
      << "The header should be edited and receives the declaration of the new "
         "function";

  if (EditedFiles.begin() != EditedFiles.end()) {
    EXPECT_EQ(EditedFiles.begin()->second, HeaderOutputCheck);
  }
}

TEST_F(ExtractFunctionTest, ConstDifferentHeaderSourceTest) {
  Header = R"cpp(
    class SomeClass {
      void f() const;
    };
  )cpp";

  std::string OutOfLineSource = R"cpp(
    void SomeClass::f() const {
      [[int x;]]
    }
  )cpp";

  std::string OutOfLineSourceOutputCheck = R"cpp(
    void SomeClass::extracted() const {
int x;
}
void SomeClass::f() const {
      extracted();
    }
  )cpp";

  std::string HeaderOutputCheck = R"cpp(
    class SomeClass {
      void extracted() const;
void f() const;
    };
  )cpp";

  llvm::StringMap<std::string> EditedFiles;

  EXPECT_EQ(apply(OutOfLineSource, &EditedFiles), OutOfLineSourceOutputCheck);
  EXPECT_NE(EditedFiles.begin(), EditedFiles.end())
      << "The header should be edited and receives the declaration of the new "
         "function";

  if (EditedFiles.begin() != EditedFiles.end()) {
    EXPECT_EQ(EditedFiles.begin()->second, HeaderOutputCheck);
  }
}

TEST_F(ExtractFunctionTest, StaticDifferentHeaderSourceTest) {
  Header = R"cpp(
    class SomeClass {
      static void f();
    };
  )cpp";

  std::string OutOfLineSource = R"cpp(
    void SomeClass::f() {
      [[int x;]]
    }
  )cpp";

  std::string OutOfLineSourceOutputCheck = R"cpp(
    void SomeClass::extracted() {
int x;
}
void SomeClass::f() {
      extracted();
    }
  )cpp";

  std::string HeaderOutputCheck = R"cpp(
    class SomeClass {
      static void extracted();
static void f();
    };
  )cpp";

  llvm::StringMap<std::string> EditedFiles;

  EXPECT_EQ(apply(OutOfLineSource, &EditedFiles), OutOfLineSourceOutputCheck);
  EXPECT_NE(EditedFiles.begin(), EditedFiles.end())
      << "The header should be edited and receives the declaration of the new "
         "function";

  if (EditedFiles.begin() != EditedFiles.end()) {
    EXPECT_EQ(EditedFiles.begin()->second, HeaderOutputCheck);
  }
}

TEST_F(ExtractFunctionTest, DifferentContextHeaderSourceTest) {
  Header = R"cpp(
    namespace ns{
    class A {
      class C {
      public:
        class RType {};
      };

      class T {
        class SomeClass {
          static C::RType f();
        };
      };
    };
    } // ns
  )cpp";

  std::string OutOfLineSource = R"cpp(
    ns::A::C::RType ns::A::T::SomeClass::f() {
      [[A::C::RType x;
      return x;]]
    }
  )cpp";

  std::string OutOfLineSourceOutputCheck = R"cpp(
    ns::A::C::RType ns::A::T::SomeClass::extracted() {
A::C::RType x;
      return x;
}
ns::A::C::RType ns::A::T::SomeClass::f() {
      return extracted();
    }
  )cpp";

  std::string HeaderOutputCheck = R"cpp(
    namespace ns{
    class A {
      class C {
      public:
        class RType {};
      };

      class T {
        class SomeClass {
          static ns::A::C::RType extracted();
static C::RType f();
        };
      };
    };
    } // ns
  )cpp";

  llvm::StringMap<std::string> EditedFiles;

  EXPECT_EQ(apply(OutOfLineSource, &EditedFiles), OutOfLineSourceOutputCheck);
  EXPECT_EQ(EditedFiles.begin()->second, HeaderOutputCheck);
}

TEST_F(ExtractFunctionTest, DifferentSyntacticContextNamespace) {
  std::string OutOfLineSource = R"cpp(
    namespace ns {
      void f();
    }

    void ns::f() {
      [[int x;]]
    }
  )cpp";

  std::string OutOfLineSourceOutputCheck = R"cpp(
    namespace ns {
      void extracted();
void f();
    }

    void ns::extracted() {
int x;
}
void ns::f() {
      extracted();
    }
  )cpp";

  EXPECT_EQ(apply(OutOfLineSource), OutOfLineSourceOutputCheck);
}

TEST_F(ExtractFunctionTest, ControlFlow) {
  Context = Function;
  // We should be able to extract break/continue with a parent loop/switch.
  EXPECT_THAT(apply(" [[for(;;) if(1) break;]] "), HasSubstr("extracted"));
  EXPECT_THAT(apply(" for(;;) [[while(1) break;]] "), HasSubstr("extracted"));
  EXPECT_THAT(apply(" [[switch(1) { break; }]]"), HasSubstr("extracted"));
  EXPECT_THAT(apply(" [[while(1) switch(1) { continue; }]]"),
              HasSubstr("extracted"));
  // Don't extract break and continue without a loop/switch parent.
  EXPECT_THAT(apply(" for(;;) [[if(1) continue;]] "), StartsWith("fail"));
  EXPECT_THAT(apply(" while(1) [[if(1) break;]] "), StartsWith("fail"));
  EXPECT_THAT(apply(" switch(1) { [[break;]] }"), StartsWith("fail"));
  EXPECT_THAT(apply(" for(;;) { [[while(1) break; break;]] }"),
              StartsWith("fail"));
}

TEST_F(ExtractFunctionTest, ExistingReturnStatement) {
  Context = File;
  const char *Before = R"cpp(
    bool lucky(int N);
    int getNum(bool Superstitious, int Min, int Max) {
      if (Superstitious) [[{
        for (int I = Min; I <= Max; ++I)
          if (lucky(I))
            return I;
        return -1;
      }]] else {
        return (Min + Max) / 2;
      }
    }
  )cpp";
  // FIXME: avoid emitting redundant braces
  const char *After = R"cpp(
    bool lucky(int N);
    int extracted(int Min, int Max) {
{
        for (int I = Min; I <= Max; ++I)
          if (lucky(I))
            return I;
        return -1;
      }
}
int getNum(bool Superstitious, int Min, int Max) {
      if (Superstitious) return extracted(Min, Max); else {
        return (Min + Max) / 2;
      }
    }
  )cpp";
  EXPECT_EQ(apply(Before), After);
}

TEST_F(ExtractFunctionTest, OverloadedOperators) {
  Context = File;
  std::string Before = R"cpp(struct A {
                int operator+(int x) { return x; }
              };
              A &operator<<(A &, int);
              A &operator|(A &, int);

              A stream{};

              void foo(int, int);

              int main() {
                [[foo(1, 2);
                foo(3, 4);
                stream << 42;
                stream + 42;
                stream | 42;
                foo(1, 2);
                foo(3, 4);]]
              })cpp";
  std::string After =
      R"cpp(struct A {
                int operator+(int x) { return x; }
              };
              A &operator<<(A &, int);
              A &operator|(A &, int);

              A stream{};

              void foo(int, int);

              void extracted() {
foo(1, 2);
                foo(3, 4);
                stream << 42;
                stream + 42;
                stream | 42;
                foo(1, 2);
                foo(3, 4);
}
int main() {
                extracted();
              })cpp";
  EXPECT_EQ(apply(Before), After);
}

TEST_F(ExtractFunctionTest, SingleStatement) {
  Context = File;
  // https://github.com/clangd/clangd/issues/698
  // A single call-expression-statement can be extracted.
  EXPECT_THAT(apply(R"cpp(
    void foo(int, int);
    void bar() {
      [[foo(1, 2);]]
    })cpp"),
              HasSubstr("extracted"));
  // https://github.com/clangd/clangd/issues/1254
  // A single statement consisting of an overloaded binary operator call can
  // be extracted, even though the SelectionTree marks the
  // CXXOperatorCallExpr itself as Unselected (its operands claim all the
  // characters).
  EXPECT_THAT(apply(R"cpp(
    struct Stream {};
    Stream &operator<<(Stream &, const char *);
    Stream stream;
    int main() {
      [[stream << "x";]]
    })cpp"),
              HasSubstr("extracted"));
  // Selecting a subexpression of an operator call (rather than the whole
  // statement) must not be extracted: it is not a standalone statement, and
  // "extracting" it would replace only part of the expression.
  EXPECT_EQ(apply(R"cpp(
    struct Stream {};
    Stream &operator<<(Stream &, int);
    Stream stream;
    void test() {
      stream << [[3]];
    })cpp"),
            "unavailable");
  // Same as above, but the selected subexpression is itself an
  // (Unselected-but-fully-covered) CXXOperatorCallExpr nested as an argument
  // of an outer call, rather than a Complete leaf expression.
  EXPECT_EQ(apply(R"cpp(
    struct Stream {};
    Stream &operator<<(Stream &, int);
    void foo(Stream &, int);
    Stream stream;
    void test() {
      foo([[stream << 3]], 4);
    })cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, ControlFlowConditions) {
  Context = File;
  // The condition of an `if` is not a discardable statement -- its value is
  // consumed by the `if` itself.
  EXPECT_EQ(apply(R"cpp(
    int example(int event1, bool event2, double event3) {
      if ([[event1 == 2 && event2 && event3 == 10.3]])
        return 1;
      return 0;
    })cpp"),
            "unavailable");
  // Same, but for other control-flow constructs' conditions.
  EXPECT_EQ(apply("void f(int x) { while ([[x > 0]]) --x; }"), "unavailable");
  EXPECT_EQ(apply("void f(int x) { do {} while ([[x > 0]]); }"), "unavailable");
  EXPECT_EQ(apply("void f(int x) { for (; [[x > 0]];) ; }"), "unavailable");
  EXPECT_EQ(apply("void f(int x) { switch ([[x + 1]]) {} }"), "unavailable");
  // A condition-variable declaration (`if (T x = ...)`) is likewise not a
  // discardable statement: its truthiness *is* the condition.
  EXPECT_EQ(apply("bool cond(); void f() { if ([[bool b = cond()]]) ; }"),
            "unavailable");
  // Unlike the condition, a loop's initializer and increment clauses have
  // their value discarded just like an ordinary statement, so they remain
  // extractable (any hazard from extracting a declaration used later is
  // already caught by ExtractionZone::requiresHoisting, independently of
  // this).
  EXPECT_THAT(apply("void f(int x) { for ([[x = 0]]; x < 10; ++x) ; }"),
              HasSubstr("extracted"));
  EXPECT_THAT(apply("void f(int x) { for (;; [[--x]]) ; }"),
              HasSubstr("extracted"));
  // Likewise, an `if`/`switch` init-statement (C++17) is extractable.
  ExtraArgs.push_back("-std=c++17");
  EXPECT_THAT(apply("void f(int x) { if ([[x = 0]]; x > 0) ; }"),
              HasSubstr("extracted"));
  EXPECT_THAT(apply("void f(int x) { switch ([[x = 0]]; x) {} }"),
              HasSubstr("extracted"));
  // Sanity check: extraction from the *body* of these constructs (as opposed
  // to their condition) is unaffected.
  EXPECT_THAT(apply("void f(int x) { if (x > 0) [[x = x * 2;]] }"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, RangeBasedFor) {
  Context = File;
  // The range-expression of a range-based for is consumed to build the
  // hidden begin/end iterators, so it's not a discardable statement either
  // (same category as an ordinary condition).
  EXPECT_EQ(apply(R"cpp(
    struct Vec { int *begin(); int *end(); };
    Vec V;
    void f() { for (auto X : [[V]]) {} }
  )cpp"),
            "unavailable");
  // Extraction from the body is unaffected.
  EXPECT_THAT(apply(R"cpp(
    struct Vec { int *begin(); int *end(); };
    Vec V;
    void foo(int);
    void f() { for (auto X : V) { [[foo(X + 1);]] } }
  )cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, VarDeclInitializer) {
  Context = File;
  // The initializer of a variable declaration is not a discardable
  // statement either: its value is required to initialize the variable, so
  // replacing it with a call to a void-returning extracted function would
  // not compile. Ascending from the initializer lands on the VarDecl itself (a
  // Decl, not a Stmt or Expr), which the general "is this a genuine statement"
  // check must also reject.
  EXPECT_EQ(apply(R"cpp(
    int func();
    void f() { auto A = [[func()]]; }
  )cpp"),
            "unavailable");
  // Same without `auto`: the type doesn't matter, only that a value is
  // required.
  EXPECT_EQ(apply(R"cpp(
    int func();
    void f() { int A = [[func()]]; }
  )cpp"),
            "unavailable");
  // Sanity check: extracting the whole declaration statement (as opposed to
  // just its initializer) is unaffected.
  EXPECT_THAT(apply(R"cpp(
    int func();
    void f() { [[int A = func();]] }
  )cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, ConstParameters) {
  Context = File;
  // A captured scalar that's only read becomes a by-value parameter;
  // non-scalars instead become a const reference (see the `S` cases
  // below). The duplicated calls below are there to prevent "pure forwarding"
  // suppression.
  EXPECT_THAT(apply(R"cpp(
    void use(int);
    void f(int x) { [[use(x); use(x);]] }
  )cpp"),
              HasSubstr("void extracted(int x)"));
  // Direct assignment: stays non-const.
  EXPECT_THAT(apply("void f(int x) { [[x = 1;]] }"),
              HasSubstr("void extracted(int &x)"));
  // Compound assignment: stays non-const.
  EXPECT_THAT(apply("void f(int x) { [[x += 1;]] }"),
              HasSubstr("void extracted(int &x)"));
  // Increment/decrement: stays non-const.
  EXPECT_THAT(apply("void f(int x) { [[++x;]] }"),
              HasSubstr("void extracted(int &x)"));
  // A non-const method call may mutate the object: stays non-const.
  EXPECT_THAT(apply(R"cpp(
    struct S { void mutate(); };
    void f(S s) { [[s.mutate(); s.mutate();]] }
  )cpp"),
              HasSubstr("void extracted(S &s)"));
  // A const method call cannot mutate the object: becomes const.
  EXPECT_THAT(apply(R"cpp(
    struct S { void inspect() const; };
    void f(S s) { [[s.inspect(); s.inspect();]] }
  )cpp"),
              HasSubstr("void extracted(const S &s)"));
  // Passed to a parameter taking a non-const reference: stays non-const,
  // since the callee could mutate it through that reference.
  EXPECT_THAT(apply(R"cpp(
    void mayMutate(int &);
    void f(int x) { [[mayMutate(x); mayMutate(x);]] }
  )cpp"),
              HasSubstr("void extracted(int &x)"));
  // Passed to a parameter taking a const reference or by value: becomes
  // an unmutated scalar, so by value.
  EXPECT_THAT(apply(R"cpp(
    void readOnly(const int &);
    void f(int x) { [[readOnly(x); readOnly(x);]] }
  )cpp"),
              HasSubstr("void extracted(int x)"));
  EXPECT_THAT(apply(R"cpp(
    void byValue(int);
    void f(int x) { [[byValue(x); byValue(x);]] }
  )cpp"),
              HasSubstr("void extracted(int x)"));
  // A scalar parameter that's already declared const keeps that
  // qualifier when passed by value: It might be relevant for overload
  // resolution.
  EXPECT_THAT(
      apply("void use(int); void f(const int x) { [[use(x); use(x);]] }"),
      HasSubstr("void extracted(const int x)"));
  // A non-scalar parameter that's already declared const keeps that
  // qualifier, since it's still passed by reference.
  EXPECT_THAT(apply(R"cpp(
    struct S {};
    void use(const S &);
    void f(const S s) { [[use(s); use(s);]] }
  )cpp"),
              HasSubstr("void extracted(const S &s)"));
}

TEST_F(ExtractFunctionTest, ConstParametersReferenceAliasing) {
  Context = File;
  // A non-const reference bound to a captured variable conservatively
  // mutates that variable too, without checking whether the reference
  // itself is ever actually mutated -- even when the binding and a later
  // mutation of the reference are two separate root statements of the same
  // zone. Failing to notice this would incorrectly mark `x` const, which
  // wouldn't compile (`int &r` can't bind to a `const int`).
  EXPECT_THAT(apply(R"cpp(
    void f(int x) {
      [[int &r = x;
      r = 2;]]
    }
  )cpp"),
              HasSubstr("void extracted(int &x)"));
}

TEST_F(ExtractFunctionTest, ConstParametersConservativeAliasing) {
  Context = File;
  // Taking the address of a captured variable conservatively mutates it,
  // regardless of what's later done with the pointer.
  EXPECT_THAT(apply("void f(int x) { [[int *p = &x;]] }"),
              HasSubstr("void extracted(int &x)"));
  // Explicit cast to a non-const reference type: stays non-const.
  EXPECT_THAT(apply("void f(int x) { [[static_cast<int &>(x) = 1;]] }"),
              HasSubstr("void extracted(int &x)"));
  // Captured by reference in a lambda: stays non-const, without checking
  // whether the lambda actually mutates it.
  EXPECT_THAT(apply("void f(int x) { [[auto l = [&x]() { int y = x; };]] }"),
              HasSubstr("int &x"));
  // Captured by value in a lambda: doesn't alias x, so it's unmutated and
  // (being a scalar) passed by value.
  EXPECT_THAT(apply("void f(int x) { [[auto l = [x]() { int y = x; };]] }"),
              HasSubstr("void extracted(int x)"));
  // Returning a captured variable is conservatively treated as a possible
  // mutation, regardless of whether the return is actually by value (safe)
  // or by non-const reference (not safe) -- telling these apart isn't
  // worth the complexity here.
  EXPECT_THAT(apply("int f(int x) { [[return x;]] }"), HasSubstr("&x"));
  // Array-typed captures are never made const.
  EXPECT_THAT(apply("void f() { int arr[5]; [[arr[0] = 1;]] }"),
              Not(HasSubstr("const")));
}

TEST_F(ExtractFunctionTest, ConstParametersMemberCallArguments) {
  Context = File;
  // A non-const-ref argument to a member call (as opposed to the implicit
  // object, already covered by ConstParameters) stays non-const. The
  // method itself is const so it doesn't also mark `s` mutated, isolating
  // the argument check from the object check.
  EXPECT_THAT(apply(R"cpp(
    struct S { void mayMutate(int &) const; };
    void f(S s, int x) { [[s.mayMutate(x); s.mayMutate(x);]] }
  )cpp"),
              HasSubstr("extracted(const S &s, int &x)"));
  // Same, but through an overloaded operator: the implicit object is
  // args[0], so the real parameter (checked against args[1:]) must be
  // found at the right offset.
  EXPECT_THAT(apply(R"cpp(
    struct S { void operator()(int &) const; };
    void f(S s, int x) { [[s(x); s(x);]] }
  )cpp"),
              HasSubstr("extracted(const S &s, int &x)"));
  // A free (non-member) operator overload has no implicit object, so all
  // operands align directly with the callee's parameters.
  EXPECT_THAT(apply(R"cpp(
    struct S {};
    void operator+(const S &, int &);
    void f(S s, int x) { [[s + x; s + x;]] }
  )cpp"),
              HasSubstr("extracted(const S &s, int &x)"));
}

TEST_F(ExtractFunctionTest, ConstParametersPointerIndirection) {
  Context = File;
  // Mutating a member/element through a pointer only mutates the pointee,
  // never the pointer's own binding, so the pointer itself is unmutated --
  // unlike the same access through a value or reference (already covered
  // by ConstParameters' `ptr` case, which is mutated directly instead).
  // Being an unmutated scalar, it's then passed by value.
  EXPECT_THAT(apply(R"cpp(
    struct S { int x; };
    void f(S *ptr) { [[ptr->x = 1;]] }
  )cpp"),
              HasSubstr("extracted(S *ptr)"));
  EXPECT_THAT(apply("void f(int *p) { [[p[0] = 1;]] }"),
              HasSubstr("extracted(int *p)"));
  // Same for a plain dereference.
  EXPECT_THAT(apply("void f(int *p) { [[*p = 1;]] }"),
              HasSubstr("extracted(int *p)"));
}

TEST_F(ExtractFunctionTest, ConstParametersConditionalReferenceBinding) {
  Context = File;
  // A reference bound to a conditional expression could alias either
  // branch at runtime, so both must be marked non-const -- getting only
  // one (or neither) would let the other stay const while still being
  // reachable through the reference, producing code that doesn't compile.
  EXPECT_THAT(apply(R"cpp(
    void f(bool cond, int c, int d) {
      [[int &a = cond ? c : d;
      a = 5;]]
    }
  )cpp"),
              HasSubstr("extracted(bool cond, int &c, int &d)"));
}

TEST_F(ExtractFunctionTest, ConstParametersConditionalMutatingAccess) {
  Context = File;
  // Same as above, but no named LHS.
  EXPECT_THAT(apply(R"cpp(
    struct S { int n; };
    void f(bool cond, S s1, S s2) {
      [[(cond ? s1 : s2).n = 0;]]
    }
  )cpp"),
              HasSubstr("extracted(bool cond, S &s1, S &s2)"));
}

TEST_F(ExtractFunctionTest, ConstParametersStaticOperatorCall) {
  Context = File;
  // A static operator() (or operator[], since C++23) has no implicit
  // object at all, so calling it through `s(...)` syntax doesn't touch
  // `s` regardless of the operator's own constness.
  ExtraArgs.push_back("-std=c++23");
  EXPECT_THAT(apply(R"cpp(
    struct S { static void operator()(int); };
    void f(S s, int x) { [[s(x); s(x);]] }
  )cpp"),
              HasSubstr("extracted(const S &s, int x)"));
}

TEST_F(ExtractFunctionTest, ConstParametersFunctionPointer) {
  Context = File;
  // A function pointer's declarator syntax puts the name inside the
  // parens around the "*", not after the whole type: printType()'s
  // placeholder must be threaded through for this to render as
  // `void (*F)(int)` rather than the uncompilable `void (*)(int) F`.
  EXPECT_THAT(apply("void f(void (*F)(int)) { [[F(1);]] }"),
              HasSubstr("extracted(void (*F)(int))"));
}

TEST_F(ExtractFunctionTest, ConstParametersFunctionReference) {
  Context = File;
  // A (non-pointer) function type can't be cv-qualified at all, unlike
  // every other non-scalar, non-array type this conservative-const
  // logic otherwise applies to: adding const here would produce
  // unparseable output (`void (const &F)(int)`).
  EXPECT_THAT(apply("void f(void (&F)(int)) { [[F(0);]] }"),
              HasSubstr("extracted(void (&F)(int))"));
}

TEST_F(ExtractFunctionTest, ConstParametersScalarsByValue) {
  Context = File;
  // An unmutated pointer is a scalar too: passed by value.
  EXPECT_THAT(apply("void use(int *); void f(int *p) { [[use(p); use(p);]] }"),
              HasSubstr("extracted(int *p)"));
  // An unmutated enum: passed by value.
  EXPECT_THAT(apply(R"cpp(
    enum E { A, B };
    void use(E);
    void f(E e) { [[use(e); use(e);]] }
  )cpp"),
              HasSubstr("extracted(E e)"));
  // A class type, even one that's small and trivially copyable, is never
  // passed by value: that's deliberately out of scope for now.
  EXPECT_THAT(apply(R"cpp(
    struct Point { int x, y; };
    void use(Point);
    void f(Point p) { [[use(p);]] }
  )cpp"),
              HasSubstr("extracted(const Point &p)"));
  // An unmutated array is not a scalar (even though its element type is):
  // stays a non-const reference, per the existing array carve-out.
  EXPECT_THAT(apply("void f() { int arr[5]; [[int x = arr[0];]] }"),
              HasSubstr("extracted(int (&arr)[5])"));
}

// Variables of reference type, const or non-const, must stay references,
// otherwise they'd stop tracking their target.
TEST_F(ExtractFunctionTest, ReferenceToScalar) {
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      void bar(int) {}
      void foo() {
      int A = 0;
      int &B = A;
      [[
        A = 1;
        bar(B);
      ]]
    })cpp"),
              HasSubstr("extracted(int &A, const int &B)"));
  EXPECT_THAT(apply(R"cpp(
      void bar(int) {}
      void foo() {
      int A = 0;
      const int &B = A;
      [[
        A = 1;
        bar(B);
      ]]
    })cpp"),
              HasSubstr("extracted(int &A, const int &B)"));
}

TEST_F(ExtractFunctionTest, VolatileScalar) {
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      void bar(const volatile int &, int) {}
      void foo() {
      volatile int V = 0;
      [[
        bar(V, 0);
      ]]
    })cpp"),
              HasSubstr("extracted(const volatile int &V)"));
}

TEST_F(ExtractFunctionTest, CFileAllowUnmodifiedScalar) {
  FileName = "a.c";
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      int i;
      void foo() {
         int j = 0;
         [[i = j;]]
    })cpp"),
              HasSubstr("extracted(int j)"));
}

TEST_F(ExtractFunctionTest, CFileModifiedScalarBecomesPointer) {
  // C has no references: a mutated capture becomes a real pointer
  // parameter instead, with the call site taking its address and the
  // body dereferencing it.
  FileName = "a.c";
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      void foo() {
         int j;
         [[j = 0;]]
    })cpp"),
              AllOf(HasSubstr("extracted(int *j)"), HasSubstr("(*j) = 0;"),
                    HasSubstr("extracted(&j)")));
}

TEST_F(ExtractFunctionTest, CFileUnmodifiedStructBecomesConstPointer) {
  // Same, but for an unmutated non-scalar capture: the parameter becomes
  // a pointer to const, and every member access on it is rewritten too.
  FileName = "a.c";
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      struct pair { int v1; int v2; };
      int i;
      void foo() {
         struct pair p;
         p.v1 = 0;
         [[i = p.v1;]]
    })cpp"),
              AllOf(HasSubstr("extracted(const struct pair *p)"),
                    HasSubstr("i = p->v1;"), HasSubstr("extracted(&p)")));
}

TEST_F(ExtractFunctionTest, CFileStructMixedUses) {
  // The same capture can appear both as a member-access base (rewritten
  // to "->") and as a plain use (wrapped in "(*...)") within a single
  // extraction; each occurrence is rewritten independently.
  FileName = "a.c";
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      struct pair { int v1; int v2; };
      void use(struct pair);
      int i;
      void foo() {
         struct pair p;
         [[use(p); i = p.v1;]]
    })cpp"),
              AllOf(HasSubstr("use((*p));"), HasSubstr("i = p->v1;")));
}

TEST_F(ExtractFunctionTest, CFileRejectMacroDot) {
  // The identifier itself need not be a macro expansion for the
  // member-access ".", immediately following it, to be one -- that dot
  // is a separate token with its own location, which also needs
  // checking before relying on it to splice in "->".
  FileName = "a.c";
  Context = File;
  EXPECT_EQ(apply(R"cpp(
      #define DOT .
      struct pair { int v1; int v2; };
      void foo() {
         struct pair p;
         [[p DOT v1 = 1;]]
    })cpp"),
            "fail: Too complex to extract.");
}

TEST_F(ExtractFunctionTest, CFileModifiedArrayStaysPlainPointer) {
  // Unlike other non-scalar types, an array decays to a pointer on its
  // own wherever it's used, so it needs neither an address-of at the
  // call site nor a dereference-rewrite of its uses in the body. The
  // parameter's own type must be decayed too, though: leaving it as an
  // array type would print as the uncompilable "int[5] arr" (there's no
  // special-cased array declarator syntax, unlike C++'s reference case).
  FileName = "a.c";
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      void foo() {
         int arr[5];
         [[arr[0] = 1;]]
    })cpp"),
              AllOf(HasSubstr("extracted(int *arr)"), HasSubstr("arr[0] = 1;"),
                    HasSubstr("extracted(arr)"), Not(HasSubstr("&arr"))));
}

TEST_F(ExtractFunctionTest, CFileStaticFunctionStaysStatic) {
  // A free function's own `static` (internal linkage) must carry over to
  // an extracted sibling, or that sibling would default to external
  // linkage instead.
  FileName = "a.c";
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      static void foo() {
         int j = 0;
         [[int k = j;]]
    })cpp"),
              HasSubstr("static void extracted"));
}

TEST_F(ExtractFunctionTest, CFileStaticForwardDeclaredFunctionStaysStatic) {
  // Same as above, but the definition itself omits `static` (legal in C:
  // once a prior declaration gives the function internal linkage, a
  // later one doesn't need to repeat it, and still has it). Checking
  // only the current declaration's storage class would miss this.
  FileName = "a.c";
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      static void foo();
      void foo() {
         int j = 0;
         [[int k = j;]]
    })cpp"),
              HasSubstr("static void extracted"));
}

TEST_F(ExtractFunctionTest, CFileRejectArraySizeof) {
  // Decaying the array to a pointer parameter would silently change the
  // meaning of a `sizeof` on it (pointer size instead of array size), so
  // this is refused rather than risk miscompiling it.
  FileName = "a.c";
  Context = File;
  EXPECT_EQ(apply(R"cpp(
      void foo() {
         int arr[5];
         [[int n = sizeof(arr);]]
    })cpp"),
            "fail: Too complex to extract.");
}

TEST_F(ExtractFunctionTest, CFileRejectArrayAlignof) {
  // Same hazard as sizeof, and the same UnaryExprOrTypeTraitExpr AST
  // node: alignof(int) and alignof(int *) aren't guaranteed to match
  // (and commonly don't, e.g. 4 vs 8 on a typical 64-bit target).
  FileName = "a.c";
  Context = File;
  EXPECT_EQ(apply(R"cpp(
      void foo() {
         int arr[5];
         [[int n = __alignof(arr);]]
    })cpp"),
            "fail: Too complex to extract.");
}

TEST_F(ExtractFunctionTest, CFileRejectArrayTypeof) {
  // Same hazard again, but via a completely different AST node
  // (TypeOfExprType, reached through the VarDecl's TypeLoc, not through
  // any Stmt a plain expression visitor would see): typeof(arr) would
  // resolve to the decayed pointer type instead of the array type.
  FileName = "a.c";
  Context = File;
  EXPECT_EQ(apply(R"cpp(
      void foo() {
         int arr[5];
         [[__typeof__(arr) copy;]]
    })cpp"),
            "fail: Too complex to extract.");
}

TEST_F(ExtractFunctionTest, CFileAllowSingleDimensionVariableLengthArray) {
  // A VLA's own outermost dimension is never a problem: once decayed,
  // it no longer appears in the parameter type at all (`int A[N]`
  // decays to plain `int *A`, not something that needs to mention `N`).
  FileName = "a.c";
  Context = File;
  EXPECT_THAT(apply(R"cpp(
      void foo(int N) {
         int A[N];
         [[A[0] = 1;]]
    })cpp"),
              HasSubstr("extracted(int *A)"));
}

TEST_F(ExtractFunctionTest, CFileRejectMultiDimensionVariableLengthArray) {
  // Unlike the single-dimension case, an inner dimension that's still
  // variable-length after decaying the outer one (`int A[2][N]` decays
  // to `int (*A)[N]`) is a real hazard: `N` is an arbitrary expression,
  // typically another variable's name, that isn't necessarily (and
  // here isn't at all) itself a parameter of the extracted function,
  // and even printing it would reference a name meaningless in the new
  // scope.
  FileName = "a.c";
  Context = File;
  EXPECT_EQ(apply(R"cpp(
      void foo(int N) {
         int A[2][N];
         [[A[0][0] = 1;]]
    })cpp"),
            "fail: Too complex to extract.");
}

TEST_F(ExtractFunctionTest, CXXRejectVariableLengthArray) {
  // Clang also supports VLAs in C++ as a GNU extension. An unmutated
  // array capture is passed by reference there (see
  // ConstParametersScalarsByValue's plain, fixed-size array case,
  // which that's still fine for) rather than decaying it to a pointer
  // the way C does -- but a reference's declared type spells out every
  // dimension's bound, including the outermost one:
  // `void extracted(int (&A)[2][N])` looks plausible, and even
  // compiles, but a call to it doesn't -- `N` in the parameter list
  // isn't the same `N` as at the call site, and a reference-to-array
  // parameter can't be bound by just matching the spelling.
  Context = File;
  EXPECT_EQ(apply(R"cpp(
      void foo(int N) {
         int A[2][N];
         [[A[0][0] = 1;]]
    })cpp"),
            "fail: Too complex to extract.");
}

TEST_F(ExtractFunctionTest, CXXRejectSingleDimensionVariableLengthArray) {
  // Unlike in C (where a single-dimension VLA decays to a pointer that
  // no longer mentions the bound at all, see
  // CFileAllowSingleDimensionVariableLengthArray), a C++ reference's
  // declared type keeps the bound regardless of dimension count, so
  // even this simpler case has the same hazard as the 2D one above.
  Context = File;
  EXPECT_EQ(apply(R"cpp(
      void foo(int N) {
         int A[N];
         [[A[0] = 1;]]
    })cpp"),
            "fail: Too complex to extract.");
}

TEST_F(ExtractFunctionTest, CXXRejectPointerToVariableLengthArray) {
  // A pointer to a VLA (`int (*)[N]`) isn't an array type itself, so
  // the array-specific VLA checks above don't apply to it -- but it
  // has the exact same hazard: its pointee's bound is baked into the
  // type regardless of how the pointer itself ends up passed (by
  // value, by reference, or, in C, by a further pointer indirection),
  // so it can't be printed without mentioning a name that isn't in
  // scope.
  Context = File;
  EXPECT_EQ(apply(R"cpp(
      void foo(int N) {
         int (*A)[N];
         [[(*A)[0] = 1;]]
    })cpp"),
            "fail: Too complex to extract.");
}

TEST_F(ExtractFunctionTest, CFileRejectMutatedPointerToVariableLengthArray) {
  // Same hazard as above, but in C, and with the pointer additionally
  // mutated (so it would otherwise go through the C pointer-adapter
  // path instead of being passed by value): the bound is baked into
  // the pointee type regardless, so `int (**A)[N]` is just as broken
  // as the unmutated `int (*A)[N]` would have been.
  FileName = "a.c";
  Context = File;
  EXPECT_EQ(apply(R"cpp(
      void foo(int N) {
         int (*A)[N];
         A = 0;
         [[(*A)[0] = 1;]]
    })cpp"),
            "fail: Too complex to extract.");
}

TEST_F(ExtractFunctionTest, RejectPureForwardingZeroArgCall) {
  Context = File;
  // No abstraction gained: the extracted function would just rename the
  // call, with no customizable behavior at all.
  EXPECT_EQ(apply("void foo(); void f() { [[foo();]] }"), "unavailable");
}

TEST_F(ExtractFunctionTest, RejectPureForwardingSingleArgCall) {
  Context = File;
  // x is forwarded to foo unchanged: still just a rename.
  EXPECT_EQ(apply("void foo(int); void f(int x) { [[foo(x);]] }"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_LiteralArg) {
  Context = File;
  // Hardcoding 100 is a real specialization, unlike a plain rename.
  EXPECT_THAT(apply("void foo(int); void f() { [[foo(100);]] }"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_GlobalArg) {
  Context = File;
  // Hardcoding which already-visible global to use is likewise a real
  // specialization: unlike a captured local, globalVar doesn't need to
  // become a parameter at all.
  EXPECT_THAT(apply(R"cpp(
    int globalVar;
    void foo(int);
    void f() { [[foo(globalVar);]] }
  )cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_ComputedArg) {
  Context = File;
  // foo(x + y) needs 2 parameters for 1 argument slot: real computation,
  // not just forwarding x and y unchanged.
  EXPECT_THAT(apply("void foo(int); void f(int x, int y) { [[foo(x + y);]] }"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, RejectPureForwardingMethodCall) {
  Context = File;
  // Both the receiver and the argument are forwarded unchanged.
  EXPECT_EQ(apply(R"cpp(
    struct S { void method(int); };
    void f(S s, int x) { [[s.method(x);]] }
  )cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_GlobalReceiver) {
  Context = File;
  // Hardcoding which already-visible object to call the method on is a
  // real specialization, same as a global argument.
  EXPECT_THAT(apply(R"cpp(
    struct S { void method(); };
    S globalS;
    void f() { [[globalS.method();]] }
  )cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, RejectPureForwardingOperatorCall) {
  Context = File;
  // Both operands of a + b are forwarded unchanged; the operator function
  // itself (never customizable) is correctly excluded from the check.
  EXPECT_EQ(apply(R"cpp(
    struct S {};
    void operator+(const S &, int);
    void f(S s, int x) { [[s + x;]] }
  )cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, RejectPureForwardingFunctionPointerCall) {
  Context = File;
  // Unlike an ordinary function name, a function-pointer variable callee
  // does need to become a parameter -- but it's still forwarded
  // unchanged here, same as x.
  EXPECT_EQ(apply("void f(void (*fp)(int), int x) { [[fp(x);]] }"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_ComputedCallee) {
  Context = File;
  // Which function pointer gets called is itself computed here, not a
  // bare forwarded reference: real logic, not just forwarding.
  EXPECT_THAT(
      apply("void f(void (*fp[2])(int), int i, int x) { [[fp[i](x);]] }"),
      HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, RejectPureForwardingReturnStmt) {
  Context = File;
  // A call wrapped in a ReturnStmt is just as much a pure rename as a
  // bare call statement.
  EXPECT_EQ(apply("int foo(int); int f(int x) { [[return foo(x);]] }"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, RejectPureForwardingReturnStmtImplicitConversion) {
  Context = File;
  // The implicit int-to-long conversion doesn't change the verdict: the
  // extracted function would perform the exact same conversion
  // internally, so this is still just a rename.
  EXPECT_EQ(apply("int foo(int); long f(int x) { [[return foo(x);]] }"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_ReturnStmtComputedArg) {
  Context = File;
  // Same as the non-return case: a computed argument is real abstraction.
  EXPECT_THAT(
      apply("int foo(int); int f(int x, int y) { [[return foo(x + y);]] }"),
      HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, RejectBareReturnStmt) {
  Context = File;
  // A value-less return has nothing to abstract over at all: extracting
  // it would just wrap it in a function that does the exact same thing.
  EXPECT_EQ(apply("void f() { [[return;]] }"), "unavailable");
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_PointerToMemberCallee) {
  Context = File;
  // Unlike an ordinary method call, a pointer-to-member call has no
  // statically known method (getMethodDecl() is null): which member is
  // invoked is itself computed here (table[I]), not inherent to what the
  // wrapper does, so it must be checked like any other operand.
  EXPECT_THAT(apply(R"cpp(
    struct St {};
    void f(St S, void (St::*table[2])(int), int I, int A) {
      [[(S.*table[I])(A);]]
    })cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, RejectPureForwardingImplicitThis) {
  Context = File;
  // An implicit `this` receiver always refers to the same object -- the
  // extracted function becomes a sibling member itself, so `this` stays
  // implicit there too -- same as an ordinary function name, it must not
  // disqualify suppression just for not being a "bare" reference.
  EXPECT_EQ(apply(R"cpp(
    struct S {
       void foo(int);
       void bar(int A) {
         [[foo(A);]]
       }
    };)cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, RejectPureForwardingDefaultArg) {
  Context = File;
  // An argument filled in from the callee's own default (as opposed to
  // one actually written at this call site) isn't something to hardcode
  // a judgement about: it must not disqualify suppression.
  EXPECT_EQ(apply(R"cpp(
    void foo(int A, int B = 0);
    void bar(int A) {
      [[foo(A);]]
    })cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, RejectPureForwardingParenthesizedReturn) {
  Context = File;
  // Parentheses around the returned call must be looked through, same as
  // an implicit conversion.
  EXPECT_EQ(apply(R"cpp(
    int foo(int);
    int bar(int A) {
      [[return (foo(A));]]
    })cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_RepeatedArg) {
  Context = File;
  // Using the same captured local for more than one argument constrains a
  // relationship (both slots must hold the same value) that a plain
  // rename wouldn't: the same kind of real specialization as hardcoding
  // a literal, even though each individual operand is bare.
  EXPECT_THAT(apply(R"cpp(
    void foo(int A, int B);
    void bar(int A, int B) {
      [[foo(A, A);]]
    })cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_ReorderedArgs) {
  Context = File;
  // createParameters orders the extracted function's parameters by each
  // Decl's declaration order in bar (A, then B), not by how they're used
  // here -- so despite every argument being an independently bare,
  // distinct local, the extracted function ends up permuting its own
  // parameters internally (extracted(int A, int B) { foo(B, A); }): not
  // a transparent pass-through, since its signature alone no longer
  // predicts its behavior the way a plain rename's would.
  EXPECT_THAT(apply(R"cpp(
    void foo(int A, int B);
    void bar(int A, int B) {
      [[foo(B, A);]]
    })cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, RejectPureForwardingPreservedArgOrder) {
  Context = File;
  // Same distinct locals as the reordered case above, but passed in the
  // same order they're declared in bar: no permutation is hidden in the
  // extracted function, so this is still just a rename.
  EXPECT_EQ(apply(R"cpp(
    void foo(int A, int B);
    void bar(int A, int B) {
      [[foo(A, B);]]
    })cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, RejectPureForwardingFunctionScopeStatic) {
  Context = File;
  // A function-scope static variable's name, unlike a namespace-scope
  // global's, isn't visible wherever the extracted function ends up: it
  // does need forwarding, same as an ordinary local variable, so this is
  // still just a rename.
  EXPECT_EQ(apply(R"cpp(
    void foo(int);
    void bar() {
      static int A;
      [[foo(A);]]
    })cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, RejectPureForwardingFunctionScopeThreadLocal) {
  Context = File;
  // Same as the static case: thread_local doesn't change the variable's
  // name visibility, only its storage.
  EXPECT_EQ(apply(R"cpp(
    void foo(int);
    void bar() {
      thread_local int A;
      [[foo(A);]]
    })cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, OfferDespitePureForwarding_NamespaceScopeStatic) {
  Context = File;
  // Unlike a function-scope static, a namespace-scope one (internal
  // linkage notwithstanding) is visible wherever the extracted function
  // ends up, same as an ordinary global: hardcoding it is a real
  // specialization.
  EXPECT_THAT(apply(R"cpp(
    static int A;
    void foo(int);
    void bar() { [[foo(A);]] }
  )cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, RejectPureForwardingStructuredBinding) {
  Context = File;
  // A structured-binding name is declared by a BindingDecl, not a
  // VarDecl, but is just as function-local and needs forwarding the
  // same way.
  ExtraArgs.push_back("-std=c++17");
  EXPECT_EQ(apply(R"cpp(
    void foo(int);
    struct Pair { int A, B; };
    Pair make();
    void bar() {
      auto [A, B] = make();
      [[foo(A);]]
    })cpp"),
            "unavailable");
}

TEST_F(ExtractFunctionTest, RejectLocalTypeConstructedInline) {
  Context = File;
  // The extracted function always ends up placed before the enclosing
  // function, where a type declared inside it isn't visible -- so this
  // can't be extracted at all, not even as a pure rename.
  EXPECT_EQ(apply(R"cpp(
    void f() {
      struct S { int A; };
      [[S X = S{1};]]
    })cpp"),
            "fail: Cannot extract code that uses a type local to the "
            "enclosing function.");
}

TEST_F(ExtractFunctionTest, RejectLocalTypeThroughCapturedVariable) {
  Context = File;
  // Same hazard, but reached through a captured variable's type rather
  // than a type written out directly in the zone: `S` never appears as
  // its own TypeLoc within the selection at all. Caught by checking the
  // extracted function's own generated parameter type instead (`X`
  // becomes a parameter of type `S`), in getExtractedFunction.
  EXPECT_EQ(apply(R"cpp(
    void foo(int);
    void f() {
      struct S { int A; };
      S X;
      [[foo(X.A);]]
    })cpp"),
            "fail: Cannot extract code that uses a type local to the "
            "enclosing function.");
}

TEST_F(ExtractFunctionTest, RejectLocalTypeNamedOnly) {
  Context = File;
  // sizeof(S)'s operand is a bare type name, not an expression: caught
  // by VisitTypeLoc, same as the declarator in
  // RejectLocalTypeConstructedInline.
  EXPECT_EQ(apply(R"cpp(
    void foo(unsigned long);
    void f() {
      struct S { int A; };
      [[foo(sizeof(S));]]
    })cpp"),
            "fail: Cannot extract code that uses a type local to the "
            "enclosing function.");
}

TEST_F(ExtractFunctionTest, OfferDespiteEnclosingClassType) {
  Context = File;
  // A method's own enclosing class isn't function-local (its
  // DeclContext is the namespace/TU, not a function), so referencing it
  // -- unlike a type declared inside the method's own body -- is fine.
  EXPECT_THAT(apply(R"cpp(
    struct S {
      int A;
      void f() { [[S X = S{1};]] }
    };
  )cpp"),
              HasSubstr("extracted"));
}

TEST_F(ExtractFunctionTest, RejectLocalTypeInFunctionPointerSignature) {
  Context = File;
  // S never appears as its own TypeLoc anywhere in the zone -- it's only
  // reachable through captured fp's own declared type (a pointer to a
  // function returning S). The literal argument keeps this from being
  // suppressed as a pure forwarding wrapper instead (see
  // RejectPureForwarding* above), which would otherwise mask this check.
  EXPECT_EQ(apply(R"cpp(
    void f() {
      struct S { int A; };
      S (*fp)(int);
      [[fp(1);]]
    })cpp"),
            "fail: Cannot extract code that uses a type local to the "
            "enclosing function.");
}

TEST_F(ExtractFunctionTest, RejectLocalTypeThroughMemberPointer) {
  Context = File;
  EXPECT_EQ(apply(R"cpp(
    void use(int);
    void f() {
      struct S { int A; };
      S s;
      int S::*pm = &S::A;
      [[use(s.*pm);]]
    })cpp"),
            "fail: Cannot extract code that uses a type local to the "
            "enclosing function.");
}

} // namespace
} // namespace clangd
} // namespace clang
