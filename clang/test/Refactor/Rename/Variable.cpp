namespace A {
int Foo;          /* Test 1 */        // CHECK: rename [[@LINE]]:5 -> [[@LINE]]:8
}
int Foo;
int Qux = Foo;
int Baz = A::Foo; /* Test 2 */        // CHECK-NEXT: rename [[@LINE]]:14 -> [[@LINE]]:17
void fun() {
  struct {
    int Foo;
  } b = {100};
  int Foo = 100;
  Baz = Foo;
  {
    extern int Foo;
    Baz = Foo;
    Foo = A::Foo /* Test 3 */ + Baz;  // CHECK-NEXT: rename [[@LINE]]:14 -> [[@LINE]]:17
    A::Foo /* Test 4 */ = b.Foo;      // CHECK-NEXT: rename [[@LINE]]:8 -> [[@LINE]]:11
  }
  Foo = b.Foo;                        // CHECK-NOT: rename [[@LINE]]
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:2:5 -new-name=Bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:6:14 -new-name=Bar %s | FileCheck %s
// Test 3.
// RUN: clang-refactor-test rename-initiate -at=%s:16:14 -new-name=Bar %s | FileCheck %s
// Test 4.
// RUN: clang-refactor-test rename-initiate -at=%s:17:8 -new-name=Bar %s | FileCheck %s
