// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify %s

void foo();
#pragma alloc_text("hello", foo) // expected-no-diagnostics
void foo() {}

static void foo1();
#pragma alloc_text("hello", foo1) // expected-no-diagnostics
void foo1() {}
