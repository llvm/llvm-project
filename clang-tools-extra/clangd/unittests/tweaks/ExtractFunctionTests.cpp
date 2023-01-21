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

using ::testing::HasSubstr;
using ::testing::StartsWith;

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(ExtractFunction);

TEST_F(ExtractFunctionTest, FunctionTest) {
  Context = Function;

  // Root statements should have common parent.
  EXPECT_EQ(apply("for(;;) [[1+2; 1+2;]]"), "unavailable");
  // Expressions aren't extracted.
  EXPECT_EQ(apply("int x = 0; [[x++;]]"), "unavailable");
  // We don't support extraction from lambdas.
  EXPECT_EQ(apply("auto lam = [](){ [[int x;]] }; "), "unavailable");
  // Partial statements aren't extracted.
  EXPECT_THAT(apply("int [[x = 0]];"), "unavailable");
  // Extract regions that require hoisting
  EXPECT_THAT(apply(" [[int a = 5;]] a++; "), HasSubstr("extracted"));

  // Ensure that end of Zone and Beginning of PostZone being adjacent doesn't
  // lead to break being included in the extraction zone.
  EXPECT_THAT(apply("for(;;) { [[int x;]]break; }"), HasSubstr("extracted"));
  // FIXME: ExtractFunction should be unavailable inside loop construct
  // initializer/condition.
  EXPECT_THAT(apply(" for([[int i = 0;]];);"), HasSubstr("extracted"));
  // Extract certain return
  EXPECT_THAT(apply(" if(true) [[{ return; }]] "), HasSubstr("extracted"));
  // Don't extract uncertain return
  EXPECT_THAT(apply(" if(true) [[if (false) return;]] "),
              StartsWith("unavailable"));
  EXPECT_THAT(
      apply("#define RETURN_IF_ERROR(x) if (x) return\nRETU^RN_IF_ERROR(4);"),
      StartsWith("unavailable"));

  FileName = "a.c";
  EXPECT_THAT(apply(" for([[int i = 0;]];);"), HasSubstr("unavailable"));
}

TEST_F(ExtractFunctionTest, FileTest) {
  // Check all parameters are in order
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
void extracted(int &a, int &b, int * &ptr, Foo &foo) {
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
void extracted(const int &c) {
while(c) {}
}
void f(const int c) {
  extracted(c);
})cpp";
  EXPECT_EQ(apply(ConstCheckInput), ConstCheckOutput);

  // Check const qualifier with namespace
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
}

TEST_F(ExtractFunctionTest, Hoisting) {
  ExtraArgs.emplace_back("-std=c++17");
  std::string HoistingInput = R"cpp(
    int foo() {
      int a = 3;
      [[int x = 39 + a;
      ++x;
      int y = x * 2;
      int z = 4;]]
      return x + y + z;
    }
  )cpp";
  std::string HoistingOutput = R"cpp(
    auto extracted(int &a) {
int x = 39 + a;
      ++x;
      int y = x * 2;
      int z = 4;
return std::tuple{x, y, z};
}
int foo() {
      int a = 3;
      auto [x, y, z] = extracted(a);
      return x + y + z;
    }
  )cpp";
  EXPECT_EQ(apply(HoistingInput), HoistingOutput);

  std::string HoistingInput2 = R"cpp(
    int foo() {
      int a{};
      [[int b = a + 1;]]
      return b;
    }
  )cpp";
  std::string HoistingOutput2 = R"cpp(
    int extracted(int &a) {
int b = a + 1;
return b;
}
int foo() {
      int a{};
      auto b = extracted(a);
      return b;
    }
  )cpp";
  EXPECT_EQ(apply(HoistingInput2), HoistingOutput2);

  std::string HoistingInput3 = R"cpp(
    int foo(int b) {
      int a{};
      if (b == 42) {
        [[a = 123;
        return a + b;]]
      }
      a = 456;
      return a;
    }
  )cpp";
  std::string HoistingOutput3 = R"cpp(
    int extracted(int &b, int &a) {
a = 123;
        return a + b;
}
int foo(int b) {
      int a{};
      if (b == 42) {
        return extracted(b, a);
      }
      a = 456;
      return a;
    }
  )cpp";
  EXPECT_EQ(apply(HoistingInput3), HoistingOutput3);

  std::string HoistingInput3B = R"cpp(
    int foo(int b) {
      [[int a{};
      if (b == 42) {
        a = 123;
        return a + b;
      }
      a = 456;
      return a;]]
    }
  )cpp";
  std::string HoistingOutput3B = R"cpp(
    int extracted(int &b) {
int a{};
      if (b == 42) {
        a = 123;
        return a + b;
      }
      a = 456;
      return a;
}
int foo(int b) {
      return extracted(b);
    }
  )cpp";
  EXPECT_EQ(apply(HoistingInput3B), HoistingOutput3B);

  std::string HoistingInput4 = R"cpp(
    struct A {
      bool flag;
      int val;
    };
    A bar();
    int foo(int b) {
      int a = 0;
      [[auto [flag, val] = bar();
      int c = 4;
      val = c + a;]]
      return a + b + c + val;
    }
  )cpp";
  std::string HoistingOutput4 = R"cpp(
    struct A {
      bool flag;
      int val;
    };
    A bar();
    auto extracted(int &a) {
auto [flag, val] = bar();
      int c = 4;
      val = c + a;
return std::pair{val, c};
}
int foo(int b) {
      int a = 0;
      auto [val, c] = extracted(a);
      return a + b + c + val;
    }
  )cpp";
  EXPECT_EQ(apply(HoistingInput4), HoistingOutput4);

  // Cannot extract a selection that contains a type declaration that is used
  // outside of the selected range
  EXPECT_THAT(apply(R"cpp(
      [[using MyType = int;]]
      MyType x = 42;
      MyType y = x;
    )cpp"),
              "unavailable");
  EXPECT_THAT(apply(R"cpp(
      [[using MyType = int;
      MyType x = 42;]]
      MyType y = x;
    )cpp"),
              "unavailable");
  EXPECT_THAT(apply(R"cpp(
      [[struct Bar {
        int X;
      };
      auto Y = Bar{42};]]
      auto Z = Bar{Y};
    )cpp"),
              "unavailable");

  // Check that selections containing type declarations can be extracted if
  // there are no uses of the type after the selection
  std::string FullTypeAliasInput = R"cpp(
    void foo() {
      [[using MyType = int;
      MyType x = 42;
      MyType y = x;]]
    }
    )cpp";
  std::string FullTypeAliasOutput = R"cpp(
    void extracted() {
using MyType = int;
      MyType x = 42;
      MyType y = x;
}
void foo() {
      extracted();
    }
    )cpp";
  EXPECT_EQ(apply(FullTypeAliasInput), FullTypeAliasOutput);

  std::string FullStructInput = R"cpp(
    int foo() {
      [[struct Bar {
        int X;
      };
      auto Y = Bar{42};
      auto Z = Bar{Y};
      return 42;]]
    }
    )cpp";
  std::string FullStructOutput = R"cpp(
    int extracted() {
struct Bar {
        int X;
      };
      auto Y = Bar{42};
      auto Z = Bar{Y};
      return 42;
}
int foo() {
      return extracted();
    }
    )cpp";
  EXPECT_EQ(apply(FullStructInput), FullStructOutput);

  std::string ReturnTypeIsAliasedInput = R"cpp(
    int foo() {
      [[struct Bar {
        int X;
      };
      auto Y = Bar{42};
      auto Z = Bar{Y};
      using MyInt = int;
      MyInt A = 42;
      return A;]]
    }
    )cpp";
  std::string ReturnTypeIsAliasedOutput = R"cpp(
    int extracted() {
struct Bar {
        int X;
      };
      auto Y = Bar{42};
      auto Z = Bar{Y};
      using MyInt = int;
      MyInt A = 42;
      return A;
}
int foo() {
      return extracted();
    }
    )cpp";
  EXPECT_EQ(apply(ReturnTypeIsAliasedInput), ReturnTypeIsAliasedOutput);

  EXPECT_THAT(apply(R"cpp(
      [[struct Bar {
        int X;
      };
      auto Y = Bar{42};]]
      auto Z = Bar{Y};
    )cpp"),
              "unavailable");

  std::string ControlStmtInput1 = R"cpp(
    float foo(float* p, int n) {
      [[float sum = 0.0F;
      for (int i = 0; i < n; ++i) {
        if (p[i] > 0.0F) {
          sum += p[i];
        }
      }
      return sum]];
    }
    )cpp";

  std::string ControlStmtOutput1 = R"cpp(
    float extracted(float * &p, int &n) {
float sum = 0.0F;
      for (int i = 0; i < n; ++i) {
        if (p[i] > 0.0F) {
          sum += p[i];
        }
      }
      return sum;
}
float foo(float* p, int n) {
      return extracted(p, n);
    }
    )cpp";

  EXPECT_EQ(apply(ControlStmtInput1), ControlStmtOutput1);

  std::string ControlStmtInput2 = R"cpp(
    float foo(float* p, int n) {
      float sum = 0.0F;
      [[for (int i = 0; i < n; ++i) {
        if (p[i] > 0.0F) {
          sum += p[i];
        }
      }
      return sum]];
    }
    )cpp";

  std::string ControlStmtOutput2 = R"cpp(
    float extracted(float * &p, int &n, float &sum) {
for (int i = 0; i < n; ++i) {
        if (p[i] > 0.0F) {
          sum += p[i];
        }
      }
      return sum;
}
float foo(float* p, int n) {
      float sum = 0.0F;
      return extracted(p, n, sum);
    }
    )cpp";

  EXPECT_EQ(apply(ControlStmtInput2), ControlStmtOutput2);
}

TEST_F(ExtractFunctionTest, HoistingCXX11) {
  ExtraArgs.emplace_back("-std=c++11");
  std::string HoistingInput = R"cpp(
    int foo() {
      int a = 3;
      [[int x = 39 + a;
      ++x;
      int y = x * 2;
      int z = 4;]]
      return x + y + z;
    }
  )cpp";
  EXPECT_THAT(apply(HoistingInput), HasSubstr("unavailable"));

  std::string HoistingInput2 = R"cpp(
    int foo() {
      int a;
      [[int b = a + 1;]]
      return b;
    }
  )cpp";
  std::string HoistingOutput2 = R"cpp(
    int extracted(int &a) {
int b = a + 1;
return b;
}
int foo() {
      int a;
      auto b = extracted(a);
      return b;
    }
  )cpp";
  EXPECT_EQ(apply(HoistingInput2), HoistingOutput2);
}

TEST_F(ExtractFunctionTest, HoistingCXX14) {
  ExtraArgs.emplace_back("-std=c++14");
  std::string HoistingInput = R"cpp(
    int foo() {
      int a = 3;
      [[int x = 39 + a;
      ++x;
      int y = x * 2;
      int z = 4;]]
      return x + y + z;
    }
  )cpp";
  std::string HoistingOutput = R"cpp(
    auto extracted(int &a) {
int x = 39 + a;
      ++x;
      int y = x * 2;
      int z = 4;
return std::tuple{x, y, z};
}
int foo() {
      int a = 3;
      auto returned = extracted(a);
auto x = std::get<0>(returned);
auto y = std::get<1>(returned);
auto z = std::get<2>(returned);
      return x + y + z;
    }
  )cpp";
  EXPECT_EQ(apply(HoistingInput), HoistingOutput);

  std::string HoistingInput2 = R"cpp(
    int foo() {
      int a;
      [[int b = a + 1;]]
      return b;
    }
  )cpp";
  std::string HoistingOutput2 = R"cpp(
    int extracted(int &a) {
int b = a + 1;
return b;
}
int foo() {
      int a;
      auto b = extracted(a);
      return b;
    }
  )cpp";
  EXPECT_EQ(apply(HoistingInput2), HoistingOutput2);
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
  // FIXME: min/max should be by value.
  // FIXME: avoid emitting redundant braces
  const char *After = R"cpp(
    bool lucky(int N);
    int extracted(int &Min, int &Max) {
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

} // namespace
} // namespace clangd
} // namespace clang
