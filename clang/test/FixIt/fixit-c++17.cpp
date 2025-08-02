// RUN: %clang_cc1 -verify -std=c++17 -pedantic-errors %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -std=c++17 -fixit %t
// RUN: %clang_cc1 -Wall -pedantic-errors -x c++ -std=c++17 %t

/* This is a test of the various code modification hints that only
   apply in C++17. */
template<int... args>
int foo() {
    return (args + 1 + ...); // expected-error {{expression not permitted as operand of fold expression}}
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"("
// CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:21-[[@LINE-3]]:21}:")"
