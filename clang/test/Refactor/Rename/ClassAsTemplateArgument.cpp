class Foo /* Test 1 */ {};    // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10

template <typename T>
void func() {}

template <typename T>
class Baz {};

int main() {
  func<Foo>(); /* Test 2 */   // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:11
  Baz<Foo> /* Test 3 */ obj;  // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
  return 0;
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:1:7 -new-name=Bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:10:8 -new-name=Bar %s | FileCheck %s
// Test 3.
// RUN: clang-refactor-test rename-initiate -at=%s:11:7 -new-name=Bar %s | FileCheck %s
