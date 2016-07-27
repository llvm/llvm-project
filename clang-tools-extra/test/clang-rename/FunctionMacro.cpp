// RUN: clang-rename -offset=138 -new-name=macro_function %s -- | FileCheck %s

#define moo foo // CHECK: #define moo macro_function

int foo() {     // CHECK: int macro_function() {
  return 42;
}

void boo(int value) {}

void qoo() {
  foo();        // CHECK: macro_function();
  boo(foo());   // CHECK: boo(macro_function());
  moo();
  boo(moo());
}

// Use grep -FUbo 'foo' <file> to get the correct offset of foo when changing
// this file.
