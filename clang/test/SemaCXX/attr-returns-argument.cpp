// RUN: %clang_cc1 -fsyntax-only -verify %s

[[clang::returns_argument(1)]] int test1(int); // expected-error {{'clang::returns_argument' attribute only applies to a pointer or reference ('int' is invalid)}}
[[clang::returns_argument(1)]] int* test2(long*); // expected-error {{returned parameter has to reference the same type as the return type}} \
                                            expected-note {{return type is 'int *'}}
[[clang::returns_argument(1)]] int* test3(int*);
[[clang::returns_argument(1)]] int& test4(int&);
[[clang::returns_argument(1)]] int* test5(int* const);
[[clang::returns_argument(1)]] int& test6(int*);
[[clang::returns_argument(2)]] int& test7(int*); // expected-error {{'clang::returns_argument' attribute parameter 1 is out of bounds}}

struct S {
  [[clang::returns_argument(1)]] S& func1();
  [[clang::returns_argument(2)]] S& func2(); // expected-error {{'clang::returns_argument' attribute parameter 1 is out of bounds}}
  [[clang::returns_argument(1)]] int* func3(int*); // expected-error {{returned parameter has to reference the same type as the return type}} \
                                             expected-note {{return type is 'int *'}}
  [[clang::returns_argument(1)]] static S& func4(S*);
};
