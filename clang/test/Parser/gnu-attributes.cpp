// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

[[gnu::no_stack_protector]] void test1(int i) {} // ok
