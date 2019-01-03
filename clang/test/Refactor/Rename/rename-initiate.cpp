// Note: the run lines follow their respective tests, since line/column
// matter in this test

class Test {  // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:11
public:
  Test() { }  // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:7
  ~Test() { } // CHECK: rename [[@LINE]]:4 -> [[@LINE]]:8
  void doSomething() {
    return;
  }
};

void foo() {
  Test test;  // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:7
  test.doSomething();
}

// RUN: clang-refactor-test rename-initiate -at=%s:4:7 -new-name=Foo %s | FileCheck %s
// RUN: clang-refactor-test rename-initiate -at=%s:4:8 -new-name=Foo %s | FileCheck %s
// RUN: clang-refactor-test rename-initiate -at=%s:4:9 -new-name=Foo %s | FileCheck %s
// RUN: clang-refactor-test rename-initiate -at=%s:4:10 -new-name=Foo %s | FileCheck %s

// RUN: not clang-refactor-test rename-initiate -at=%s:1:10 -new-name=Foo %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR1 %s
// CHECK-ERROR1: error: could not rename symbol at the given location

// RUN: not clang-refactor-test rename-initiate -at=%s -new-name=Foo %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR2 %s
// CHECK-ERROR2: error: The -at option must use the <file:line:column> format
