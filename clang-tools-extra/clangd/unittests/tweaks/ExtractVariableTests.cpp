//===-- ExtractVariableTests.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(ExtractVariable);

TEST_F(ExtractVariableTest, Test) {
  const char *AvailableCases = R"cpp(
    int xyz(int a = 1) {
      struct T {
        int bar(int a = 1);
        int z;
      } t;
      // return statement
      return [[[[t.b[[a]]r]]([[t.z]])]];
    }
    void f() {
      int a = 5 + [[4 * [[[[xyz]]()]]]];
      // multivariable initialization
      if(1)
        int x = [[1]] + 1, y = a + [[1]], a = [[1]] + 2, z = a + 1;
      // if without else
      if([[1]])
        a = [[1]] + 1;
      // if with else
      if(a < [[3]])
        if(a == [[4]])
          a = [[5]] + 1;
        else
          a = [[5]] + 1;
      else if (a < [[4]])
        a = [[4]] + 1;
      else
        a = [[5]] + 1;
      // for loop
      for(a = [[1]] + 1; a > [[[[3]] + [[4]]]]; a++)
        a = [[2]] + 1;
      // while
      while(a < [[1]])
        a = [[1]] + 1;
      // do while
      do
        a = [[1]] + 1;
      while(a < [[3]]);
    }
  )cpp";
  EXPECT_AVAILABLE(AvailableCases);

  ExtraArgs = {"-xc"};
  const char *AvailableC = R"cpp(
    void foo() {
      int x = [[1]] + 1;
    })cpp";
  EXPECT_AVAILABLE(AvailableC);

  ExtraArgs = {"-xc"};
  const char *NoCrashCasesC = R"cpp(
    // error-ok: broken code, but shouldn't crash
    int x = [[foo()]];
    )cpp";
  EXPECT_UNAVAILABLE(NoCrashCasesC);

  ExtraArgs = {"-xc"};
  const char *NoCrashDesignator = R"cpp(
    struct A {
      struct {
        int x;
      };
    };
    struct B {
      int y;
    };
    void foo(struct B *b) {
      struct A a = {.x=b[[->]]y};
    }
  )cpp";
  EXPECT_AVAILABLE(NoCrashDesignator);

  ExtraArgs = {"-xobjective-c"};
  const char *AvailableObjC = R"cpp(
    __attribute__((objc_root_class))
    @interface Foo
    @end
    @implementation Foo
    - (void)method {
      int x = [[1]] + 2;
    }
    @end)cpp";
  EXPECT_AVAILABLE(AvailableObjC);
  ExtraArgs = {};

  const char *NoCrashCases = R"cpp(
    // error-ok: broken code, but shouldn't crash
    template<typename T, typename ...Args>
    struct Test<T, Args...> {
    Test(const T &v) :val[[(^]]) {}
      T val;
    };
  )cpp";
  EXPECT_UNAVAILABLE(NoCrashCases);

  const char *UnavailableCases = R"cpp(
    int xyz(int a = [[1]]) {
      struct T {
        int bar(int a = [[1]]) {
          int b = [[z]];
        }
        int z = [[1]];
      } t;
      int x = [[1 + 2]];
      int y;
      y = [[1 + 2]];
      return [[t]].bar([[t]].z);
    }
    void v() { return; }

    // function default argument
    void f(int b = [[1]]) {
      // empty selection
      int a = ^1 ^+ ^2;
      // void expressions
      auto i = new int, j = new int;
      [[[[delete i]], delete j]];
      [[v]]();
      // if
      if(1)
        int x = 1, y = a + 1, a = 1, z = [[a + 1]];
      if(int a = 1)
        if([[a + 1]] == 4)
          a = [[[[a]] +]] 1;
      // for loop
      for(int a = 1, b = 2, c = 3; a > [[b + c]]; [[a++]])
        a = [[a + 1]];
      // lambda
      auto lamb = [&[[a]], &[[b]]](int r = [[1]]) {return 1;};
      // assignment
      xyz([[a = 5]]);
      xyz([[a *= 5]]);
      // Variable DeclRefExpr
      a = [[b]];
      a = [[xyz()]];
      // statement expression
      [[xyz()]];
      while (a)
        [[++a]];
      // label statement
      goto label;
      label:
        a = [[1]];

      // lambdas: captures
      int x = 0;
      [ [[=]] ](){};
      [ [[&]] ](){};
      [ [[x]] ](){};
      [ [[&x]] ](){};
      [y = [[x]] ](){};
      [ [[y = x]] ](){};

      // lambdas: default args, cannot extract into function-local scope
      [](int x = [[10]]){};
      [](auto h = [[ [i = [](){}](){} ]]) {};

      // lambdas: default args
      // Extracting from capture initializers is usually fine,
      // but not if the capture itself is nested inside a default argument
      [](auto h = [i = [[ [](){} ]]](){}) {};
      [](auto h = [i = [[ 42 ]]](){}) {};

      // lambdas: scope
      if (int a = 1)
            if ([[ [&](){ return a + 1; } ]]() == 4)
              a = a + 1;

      for (int c = 0; [[ [&]() { return c < b; } ]](); ++c) {
      }
      for (int c = 0; [[ [&]() { return c < b; } () ]]; ++c) {
      }

      // lambdas: scope with structured binding
      struct Coordinates {
        int x{};
        int y{};
      };
      Coordinates c{};
      if (const auto [x, y] = c; x > y)
        auto f = [[ [&]() { return x + y; } ]];

      // lambdas: referencing outside variables that block extraction
      //          in trailing return type or in a decltype used
      //          by a parameter
      if (int a = 1)
        if ([[ []() -> decltype(a) { return 1; } ]] () == 4)
          a = a + 1;
      if (int a = 1)
        if ([[ [](int x = decltype(a){}) { return 1; } ]] () == 4)
          a = a + 1;
      if (int a = 1)
        if ([[ [](decltype(a) x) { return 1; } ]] (42) == 4)
          a = a + 1;
    }
  )cpp";
  EXPECT_UNAVAILABLE(UnavailableCases);

  ExtraArgs = {"-std=c++20"};
  const char *UnavailableCasesCXX20 = R"cpp(
    template <typename T>
    concept Constraint = requires (T t) { true; };
    void foo() {
      // lambdas: referencing outside variables that block extraction
      //          in requires clause or defaulted explicit template parameters
      if (int a = 1)
        if ([[ [](auto b) requires (Constraint<decltype(a)> && Constraint<decltype(b)>) { return true; } ]] (a))
          a = a + 1;

      if (int a = 1)
        if ([[ []<typename T = decltype(a)>(T b) { return true; } ]] (a))
          a = a + 1;
    }
  )cpp";
  EXPECT_UNAVAILABLE(UnavailableCasesCXX20);
  ExtraArgs.clear();

  // vector of pairs of input and output strings
  std::vector<std::pair<std::string, std::string>> InputOutputs = {
      // extraction from variable declaration/assignment
      {R"cpp(void varDecl() {
                   int a = 5 * (4 + (3 [[- 1)]]);
                 })cpp",
       R"cpp(void varDecl() {
                   auto placeholder = (3 - 1); int a = 5 * (4 + placeholder);
                 })cpp"},
      // FIXME: extraction from switch case
      /*{R"cpp(void f(int a) {
               if(1)
                 while(a < 1)
                   switch (1) {
                       case 1:
                         a = [[1 + 2]];
                         break;
                       default:
                         break;
                   }
             })cpp",
       R"cpp(void f(int a) {
               auto placeholder = 1 + 2; if(1)
                 while(a < 1)
                   switch (1) {
                       case 1:
                         a = placeholder;
                         break;
                       default:
                         break;
                   }
             })cpp"},*/
      // Macros
      {R"cpp(#define PLUS(x) x++
                 void f(int a) {
                   int y = PLUS([[1+a]]);
                 })cpp",
       /*FIXME: It should be extracted like this.
        R"cpp(#define PLUS(x) x++
              void f(int a) {
                auto placeholder = 1+a; int y = PLUS(placeholder);
              })cpp"},*/
       R"cpp(#define PLUS(x) x++
                 void f(int a) {
                   auto placeholder = PLUS(1+a); int y = placeholder;
                 })cpp"},
      // ensure InsertionPoint isn't inside a macro
      {R"cpp(#define LOOP(x) while (1) {a = x;}
                 void f(int a) {
                   if(1)
                    LOOP(5 + [[3]])
                 })cpp",
       R"cpp(#define LOOP(x) while (1) {a = x;}
                 void f(int a) {
                   auto placeholder = 3; if(1)
                    LOOP(5 + placeholder)
                 })cpp"},
      {R"cpp(#define LOOP(x) do {x;} while(1);
                 void f(int a) {
                   if(1)
                    LOOP(5 + [[3]])
                 })cpp",
       R"cpp(#define LOOP(x) do {x;} while(1);
                 void f(int a) {
                   auto placeholder = 3; if(1)
                    LOOP(5 + placeholder)
                 })cpp"},
      // attribute testing
      {R"cpp(void f(int a) {
                    [ [gsl::suppress("type")] ] for (;;) a = [[1]] + 1;
                 })cpp",
       R"cpp(void f(int a) {
                    auto placeholder = 1; [ [gsl::suppress("type")] ] for (;;) a = placeholder + 1;
                 })cpp"},
      // MemberExpr
      {R"cpp(class T {
                   T f() {
                     return [[T().f()]].f();
                   }
                 };)cpp",
       R"cpp(class T {
                   T f() {
                     auto placeholder = T().f(); return placeholder.f();
                   }
                 };)cpp"},
      // Function DeclRefExpr
      {R"cpp(int f() {
                   return [[f]]();
                 })cpp",
       R"cpp(int f() {
                   auto placeholder = f(); return placeholder;
                 })cpp"},
      // FIXME: Wrong result for \[\[clang::uninitialized\]\] int b = [[1]];
      // since the attr is inside the DeclStmt and the bounds of
      // DeclStmt don't cover the attribute.

      // Binary subexpressions
      {R"cpp(void f() {
                   int x = 1 + [[2 + 3 + 4]] + 5;
                 })cpp",
       R"cpp(void f() {
                   auto placeholder = 2 + 3 + 4; int x = 1 + placeholder + 5;
                 })cpp"},
      {R"cpp(void f() {
                   int x = [[1 + 2 + 3]] + 4 + 5;
                 })cpp",
       R"cpp(void f() {
                   auto placeholder = 1 + 2 + 3; int x = placeholder + 4 + 5;
                 })cpp"},
      {R"cpp(void f() {
                   int x = 1 + 2 + [[3 + 4 + 5]];
                 })cpp",
       R"cpp(void f() {
                   auto placeholder = 3 + 4 + 5; int x = 1 + 2 + placeholder;
                 })cpp"},
      // Non-associative operations have no special support
      {R"cpp(void f() {
                   int x = 1 - [[2 - 3 - 4]] - 5;
                 })cpp",
       R"cpp(void f() {
                   auto placeholder = 1 - 2 - 3 - 4; int x = placeholder - 5;
                 })cpp"},
      // A mix of associative operators isn't associative.
      {R"cpp(void f() {
                   int x = 0 + 1 * [[2 + 3]] * 4 + 5;
                 })cpp",
       R"cpp(void f() {
                   auto placeholder = 1 * 2 + 3 * 4; int x = 0 + placeholder + 5;
                 })cpp"},
      // Overloaded operators are supported, we assume associativity
      // as if they were built-in.
      {R"cpp(struct S {
                   S(int);
                 };
                 S operator+(S, S);

                 void f() {
                   S x = S(1) + [[S(2) + S(3) + S(4)]] + S(5);
                 })cpp",
       R"cpp(struct S {
                   S(int);
                 };
                 S operator+(S, S);

                 void f() {
                   auto placeholder = S(2) + S(3) + S(4); S x = S(1) + placeholder + S(5);
                 })cpp"},
      // lambda expressions
      {R"cpp(template <typename T> void f(T) {}
                void f2() {
                  f([[ [](){ return 42; }]]);
                }
                )cpp",
       R"cpp(template <typename T> void f(T) {}
                void f2() {
                  auto placeholder = [](){ return 42; }; f( placeholder);
                }
                )cpp"},
      {R"cpp(template <typename T> void f(T) {}
                void f2() {
                  f([x = [[40 + 2]] ](){ return 42; });
                }
                )cpp",
       R"cpp(template <typename T> void f(T) {}
                void f2() {
                  auto placeholder = 40 + 2; f([x = placeholder ](){ return 42; });
                }
                )cpp"},
      {R"cpp(auto foo(int VarA) {
                  return [VarA]() {
                    return [[ [VarA, VarC = 42 + VarA](int VarB) { return VarA + VarB + VarC; }]];
                  };
                }
                )cpp",
       R"cpp(auto foo(int VarA) {
                  return [VarA]() {
                    auto placeholder = [VarA, VarC = 42 + VarA](int VarB) { return VarA + VarB + VarC; }; return  placeholder;
                  };
                }
                )cpp"},
      {R"cpp(template <typename T> void f(T) {}
                void f2(int var) {
                  f([[ [&var](){ auto internal_val = 42; return var + internal_val; }]]);
                }
                )cpp",
       R"cpp(template <typename T> void f(T) {}
                void f2(int var) {
                  auto placeholder = [&var](){ auto internal_val = 42; return var + internal_val; }; f( placeholder);
                }
                )cpp"},
      {R"cpp(template <typename T> void f(T) { }
                struct A {
                    void f2(int& var) {
                        auto local_var = 42;
                        f([[ [&var, &local_var, this]() {
                            auto internal_val = 42;
                            return var + local_var + internal_val + member;
                        }]]);
                    }

                    int member = 42;
};
                )cpp",
       R"cpp(template <typename T> void f(T) { }
                struct A {
                    void f2(int& var) {
                        auto local_var = 42;
                        auto placeholder = [&var, &local_var, this]() {
                            auto internal_val = 42;
                            return var + local_var + internal_val + member;
                        }; f( placeholder);
                    }

                    int member = 42;
};
                )cpp"},
      {R"cpp(void f() { auto x = +[[ [](){ return 42; }]]; })cpp",
       R"cpp(void f() { auto placeholder = [](){ return 42; }; auto x = + placeholder; })cpp"},
      {R"cpp(
        template <typename T>
        auto sink(T f) { return f(); }
        int bar() {
          return sink([[ []() { return 42; }]]);
        }
       )cpp",
       R"cpp(
        template <typename T>
        auto sink(T f) { return f(); }
        int bar() {
          auto placeholder = []() { return 42; }; return sink( placeholder);
        }
       )cpp"},
      {R"cpp(
        int main() {
          if (int a = 1) {
            if ([[ [&](){ return a + 1; } ]]() == 4)
              a = a + 1;
          }
        })cpp",
       R"cpp(
        int main() {
          if (int a = 1) {
            auto placeholder = [&](){ return a + 1; }; if ( placeholder () == 4)
              a = a + 1;
          }
        })cpp"},
      {R"cpp(
        int main() {
          if (int a = 1) {
            if ([[ [&](){ return a + 1; }() ]] == 4)
              a = a + 1;
          }
        })cpp",
       R"cpp(
        int main() {
          if (int a = 1) {
            auto placeholder = [&](){ return a + 1; }(); if ( placeholder  == 4)
              a = a + 1;
          }
        })cpp"},
      {R"cpp(
        template <typename T>
        auto call(T t) { return t(); }

        int main() {
          return [[ call([](){ int a = 1; return a + 1; }) ]] + 5;
        })cpp",
       R"cpp(
        template <typename T>
        auto call(T t) { return t(); }

        int main() {
          auto placeholder = call([](){ int a = 1; return a + 1; }); return  placeholder  + 5;
        })cpp"},
      {R"cpp(
        class Foo {
          int bar() {
            return [f = [[ [this](int g) { return g + x; } ]] ]() { return 42; }();
          }
          int x;
        };
      )cpp",
       R"cpp(
        class Foo {
          int bar() {
            auto placeholder = [this](int g) { return g + x; }; return [f =  placeholder  ]() { return 42; }();
          }
          int x;
        };
      )cpp"},
      {R"cpp(
        int main() {
          return [[ []() { return 42; }() ]];
        })cpp",
       R"cpp(
        int main() {
          auto placeholder = []() { return 42; }(); return  placeholder ;
        })cpp"},
      {R"cpp(
        template <typename ...Ts>
        void foo(Ts ...args) {
          auto x = +[[ [&args...]() {} ]];
        }
      )cpp",
       R"cpp(
        template <typename ...Ts>
        void foo(Ts ...args) {
          auto placeholder = [&args...]() {}; auto x = + placeholder ;
        }
      )cpp"},
      {R"cpp(
        struct Coordinates {
          int x{};
          int y{};
        };

        int main() {
          Coordinates c = {};
          const auto [x, y] = c;
          auto f = [[ [&]() { return x + y; } ]]();
        }
        )cpp",
       R"cpp(
        struct Coordinates {
          int x{};
          int y{};
        };

        int main() {
          Coordinates c = {};
          const auto [x, y] = c;
          auto placeholder = [&]() { return x + y; }; auto f =  placeholder ();
        }
        )cpp"},
      {R"cpp(
        struct Coordinates {
          int x{};
          int y{};
        };

        int main() {
          Coordinates c = {};
          if (const auto [x, y] = c; x > y) {
            auto f = [[ [&]() { return x + y; } ]]();
          }
        }
        )cpp",
       R"cpp(
        struct Coordinates {
          int x{};
          int y{};
        };

        int main() {
          Coordinates c = {};
          if (const auto [x, y] = c; x > y) {
            auto placeholder = [&]() { return x + y; }; auto f =  placeholder ();
          }
        }
        )cpp"},
      // Don't try to analyze across macro boundaries
      // FIXME: it'd be nice to do this someday (in a safe way)
      {R"cpp(#define ECHO(X) X
                 void f() {
                   int x = 1 + [[ECHO(2 + 3) + 4]] + 5;
                 })cpp",
       R"cpp(#define ECHO(X) X
                 void f() {
                   auto placeholder = 1 + ECHO(2 + 3) + 4; int x = placeholder + 5;
                 })cpp"},
      {R"cpp(#define ECHO(X) X
                 void f() {
                   int x = 1 + [[ECHO(2) + ECHO(3) + 4]] + 5;
                 })cpp",
       R"cpp(#define ECHO(X) X
                 void f() {
                   auto placeholder = 1 + ECHO(2) + ECHO(3) + 4; int x = placeholder + 5;
                 })cpp"},
  };
  for (const auto &IO : InputOutputs) {
    EXPECT_EQ(IO.second, apply(IO.first)) << IO.first;
  }

  ExtraArgs = {"-xc"};
  InputOutputs = {
      // Function Pointers
      {R"cpp(struct Handlers {
               void (*handlerFunc)(int);
             };
             void runFunction(void (*func)(int)) {}
             void f(struct Handlers *handler) {
               runFunction([[handler->handlerFunc]]);
             })cpp",
       R"cpp(struct Handlers {
               void (*handlerFunc)(int);
             };
             void runFunction(void (*func)(int)) {}
             void f(struct Handlers *handler) {
               void (*placeholder)(int) = handler->handlerFunc; runFunction(placeholder);
             })cpp"},
      {R"cpp(int (*foo(char))(int);
             void bar() {
               (void)[[foo('c')]];
             })cpp",
       R"cpp(int (*foo(char))(int);
             void bar() {
               int (*placeholder)(int) = foo('c'); (void)placeholder;
             })cpp"},
      // Arithmetic on typedef types preserves typedef types
      {R"cpp(typedef long NSInteger;
             void varDecl() {
                NSInteger a = 2 * 5;
                NSInteger b = [[a * 7]] + 3;
             })cpp",
       R"cpp(typedef long NSInteger;
             void varDecl() {
                NSInteger a = 2 * 5;
                NSInteger placeholder = a * 7; NSInteger b = placeholder + 3;
             })cpp"},
  };
  for (const auto &IO : InputOutputs) {
    EXPECT_EQ(IO.second, apply(IO.first)) << IO.first;
  }

  ExtraArgs = {"-xobjective-c"};
  EXPECT_UNAVAILABLE(R"cpp(
      __attribute__((objc_root_class))
      @interface Foo
      - (void)setMethod1:(int)a;
      - (int)method1;
      @property int prop1;
      @end
      @implementation Foo
      - (void)method {
        [[self.method1]] = 1;
        [[self.method1]] += 1;
        [[self.prop1]] = 1;
        [[self.prop1]] += 1;
      }
      @end)cpp");
  InputOutputs = {
      // Support ObjC property references (explicit property getter).
      {R"cpp(__attribute__((objc_root_class))
             @interface Foo
             @property int prop1;
             @end
             @implementation Foo
             - (void)method {
               int x = [[self.prop1]] + 1;
             }
             @end)cpp",
       R"cpp(__attribute__((objc_root_class))
             @interface Foo
             @property int prop1;
             @end
             @implementation Foo
             - (void)method {
               int placeholder = self.prop1; int x = placeholder + 1;
             }
             @end)cpp"},
      // Support ObjC property references (implicit property getter).
      {R"cpp(__attribute__((objc_root_class))
             @interface Foo
             - (int)method1;
             @end
             @implementation Foo
             - (void)method {
               int x = [[self.method1]] + 1;
             }
             @end)cpp",
       R"cpp(__attribute__((objc_root_class))
             @interface Foo
             - (int)method1;
             @end
             @implementation Foo
             - (void)method {
               int placeholder = self.method1; int x = placeholder + 1;
             }
             @end)cpp"},
  };
  for (const auto &IO : InputOutputs) {
    EXPECT_EQ(IO.second, apply(IO.first)) << IO.first;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
