class Baz {};

class Qux {
  Baz Foo;         /* Test 1 */     // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
public:
  Qux();
};

Qux::Qux() : Foo() /* Test 2 */ {}  // CHECK: rename [[@LINE]]:14 -> [[@LINE]]:17

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:4:7 -new-name=Bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:9:14 -new-name=Bar %s | FileCheck %s
