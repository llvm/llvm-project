// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify %s

void foo();
#pragma alloc_text("hello", foo) // no-error
void foo() {}

static void foo1();
#pragma alloc_text("hello", foo1) // no-error
void foo1() {}

int foo2;
#pragma alloc_text(c, foo2) // expected-error {{'#pragma alloc_text' is applicable only to functions}}
