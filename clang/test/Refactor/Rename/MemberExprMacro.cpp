class Baz {
public:
  int Foo;  /* Test 1 */    // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
};

int qux(int x) { return 0; }
#define MACRO(a) qux(a)

int main() {
  Baz baz;
  baz.Foo = 1; /* Test 2 */ // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
  MACRO(baz.Foo);           // CHECK: rename [[@LINE]]:13 -> [[@LINE]]:16
  int y = baz.Foo;          // CHECK: rename [[@LINE]]:15 -> [[@LINE]]:18
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:3:7 -new-name=Bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:11:7 -new-name=Bar %s | FileCheck %s
