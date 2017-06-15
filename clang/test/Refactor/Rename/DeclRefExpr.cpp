class C {
public:
  static int Foo; /* Test 1 */  // CHECK: rename [[@LINE]]:14 -> [[@LINE]]:17
};

int foo(int x) { return 0; }
#define MACRO(a) foo(a)

int main() {
  C::Foo = 1;     /* Test 2 */  // CHECK: rename [[@LINE]]:6 -> [[@LINE]]:9
  MACRO(C::Foo);                // CHECK: rename [[@LINE]]:12 -> [[@LINE]]:15
  int y = C::Foo; /* Test 3 */  // CHECK: rename [[@LINE]]:14 -> [[@LINE]]:17
  return 0;
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:3:14 -new-name=Bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:10:6 -new-name=Bar %s | FileCheck %s
// Test 3.
// RUN: clang-refactor-test rename-initiate -at=%s:12:14 -new-name=Bar %s | FileCheck %s
