// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fopenmp -DCXX -verify %s

#ifndef CXX
void f1(void) __attribute__((assume(3))); // expected-error {{expected string literal as argument of 'assume' attribute}}
void f2(void) __attribute__((assume(int))); // expected-error {{expected string literal as argument of 'assume' attribute}}
void f3(void) __attribute__((assume(for))); // expected-error {{expected string literal as argument of 'assume' attribute}}
void f4(void) __attribute__((assume("QQQQ"))); // expected-warning {{unknown assumption string 'QQQQ'; attribute is potentially ignored}}
void f5(void) __attribute__((assume("omp_no_openmp")));
void f6(void) __attribute__((assume("omp_noopenmp"))); // expected-warning {{unknown assumption string 'omp_noopenmp' may be misspelled; attribute is potentially ignored, did you mean 'omp_no_openmp'?}}
void f7(void) __attribute__((assume("omp_no_openmp_routine"))); // expected-warning {{unknown assumption string 'omp_no_openmp_routine' may be misspelled; attribute is potentially ignored, did you mean 'omp_no_openmp_routines'?}}
void f8(void) __attribute__((assume("omp_no_openmp1"))); // expected-warning {{unknown assumption string 'omp_no_openmp1' may be misspelled; attribute is potentially ignored, did you mean 'omp_no_openmp'?}}
void f9(void) __attribute__((assume("omp_no_openmp", "omp_no_openmp"))); // expected-error {{'assume' attribute takes one argument}}

int g1 __attribute__((assume(0))); // expected-error {{expected string literal as argument of 'assume' attribute}}
int g2 __attribute__((assume("omp_no_openmp"))); // expected-warning {{'assume' attribute only applies to functions and Objective-C methods}}

#else
[[omp::assume(3)]] void f1(); // expected-error {{expected string literal as argument of 'assume' attribute}}
[[omp::assume(int)]] void f2(); // expected-error {{expected string literal as argument of 'assume' attribute}}
[[omp::assume(for)]] void f3(); // expected-error {{expected string literal as argument of 'assume' attribute}}
[[omp::assume("QQQQ")]] void f4(); // expected-warning {{unknown assumption string 'QQQQ'; attribute is potentially ignored}}
[[omp::assume("omp_no_openmp")]] void f5();
[[omp::assume("omp_noopenmp")]] void f6(); // expected-warning {{unknown assumption string 'omp_noopenmp' may be misspelled; attribute is potentially ignored, did you mean 'omp_no_openmp'?}}
[[omp::assume("omp_no_openmp_routine")]] void f7(); // expected-warning {{unknown assumption string 'omp_no_openmp_routine' may be misspelled; attribute is potentially ignored, did you mean 'omp_no_openmp_routines'?}}
[[omp::assume("omp_no_openmp1")]] void f8(); // expected-warning {{unknown assumption string 'omp_no_openmp1' may be misspelled; attribute is potentially ignored, did you mean 'omp_no_openmp'?}}
[[omp::assume("omp_no_openmp", "omp_no_openmp")]] void f9(); // expected-error {{'assume' attribute takes one argument}}

[[omp::assume(3)]] int g1; // expected-error {{expected string literal as argument of 'assume' attribute}}
[[omp::assume("omp_no_openmp")]] int g2; // expected-warning {{'assume' attribute only applies to functions and Objective-C methods}}
#endif
