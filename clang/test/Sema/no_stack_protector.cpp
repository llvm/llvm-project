// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

[[gnu::no_stack_protector]] void test1() {}
[[clang::no_stack_protector]] void test2() {}
