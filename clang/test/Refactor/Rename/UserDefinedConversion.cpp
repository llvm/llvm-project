class Foo {       /* Test 1 */          // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
public:
  Foo() {}                              // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
};

class Baz {
public:
  operator Foo()  /* Test 2 */ const {  // CHECK: rename [[@LINE]]:12 -> [[@LINE]]:15
    Foo foo;                            // CHECK: rename [[@LINE]]:5 -> [[@LINE]]:8
    return foo;
  }
};

int main() {
  Baz boo;
  Foo foo = static_cast<Foo>(boo);      // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
  return 0;                             // CHECK: rename [[@LINE-1]]:25 -> [[@LINE-1]]:28
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:1:7 -new-name=Bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:8:12 -new-name=Bar %s | FileCheck %s
