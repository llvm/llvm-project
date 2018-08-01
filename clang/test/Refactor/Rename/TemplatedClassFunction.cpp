template <typename T>
class A {
public:
  void foo() /* Test 1 */ {}  // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:11
};

int main(int argc, char **argv) {
  A<int> a;
  a.foo();   /* Test 2 */     // CHECK: rename [[@LINE]]:5 -> [[@LINE]]:8
  return 0;
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:4:8 -new-name=bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:9:5 -new-name=bar %s | FileCheck %s
//
// Currently unsupported test.
// XFAIL: *
