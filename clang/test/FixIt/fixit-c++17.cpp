// RUN: %clang_cc1 -verify -std=c++17 -pedantic-errors %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -std=c++17 -fixit %t
// RUN: %clang_cc1 -Wall -pedantic-errors -x c++ -std=c++17 %t

/* This is a test of the various code modification hints that only
   apply in C++17. */
template<int... args>
int foo() {
    int a = (args + 1 + ...); // expected-error {{expression not permitted as operand of fold expression}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:14-[[@LINE-1]]:14}:"("
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:22-[[@LINE-2]]:22}:")"
    int b = (args + 123 + ...); // expected-error {{expression not permitted as operand of fold expression}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:14-[[@LINE-1]]:14}:"("
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:24-[[@LINE-2]]:24}:")"
    int c = (args + 1 + 2 + ...); // expected-error {{expression not permitted as operand of fold expression}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:14-[[@LINE-1]]:14}:"("
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:26-[[@LINE-2]]:26}:")"
    int e = (... + 1 + args); // expected-error {{expression not permitted as operand of fold expression}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:20-[[@LINE-1]]:20}:"("
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:28-[[@LINE-2]]:28}:")"
    int f = (1 + ... + args + 1); // expected-error {{expression not permitted as operand of fold expression}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:24-[[@LINE-1]]:24}:"("
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:32-[[@LINE-2]]:32}:")"
    int g = (args + 1 + ... + 1); // expected-error {{expression not permitted as operand of fold expression}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:14-[[@LINE-1]]:14}:"("
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:22-[[@LINE-2]]:22}:")"
    return a + b + c + e + f + g;
}
