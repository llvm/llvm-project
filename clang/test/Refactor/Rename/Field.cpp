class Baz {
  int Foo; /* Test 1 */ // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
public:
  Baz();
};

Baz::Baz() : Foo(0) /* Test 2 */ {}  // CHECK: rename [[@LINE]]:14 -> [[@LINE]]:17

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:2:7 -new-name=Bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:7:14 -new-name=Bar %s | FileCheck %s
