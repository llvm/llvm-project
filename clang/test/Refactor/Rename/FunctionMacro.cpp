#define moo foo

int foo() /* Test 1 */ {  // CHECK: rename [[@LINE]]:5 -> [[@LINE]]:8
  return 42;
}

void boo(int value) {}

void qoo() {
  foo();                  // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
  boo(foo());             // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
  moo();                  // CHECK: macro [[@LINE]]:3 -> [[@LINE]]:3
  boo(moo());             // CHECK: macro [[@LINE]]:7 -> [[@LINE]]:7
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:3:5 -new-name=macro_function %s | FileCheck %s
