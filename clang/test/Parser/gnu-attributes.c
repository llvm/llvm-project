// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s
// expected-no-diagnostics

[[gnu::no_stack_protector]] void test1(int i) {} // ok
