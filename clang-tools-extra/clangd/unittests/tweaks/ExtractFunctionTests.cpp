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
  // FIXME: Support hoisting.
  EXPECT_THAT(apply(" [[int a = 5;]] a++; "), "unavailable");

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
