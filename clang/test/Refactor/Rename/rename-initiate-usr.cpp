// Note: the run lines follow their respective tests, since line/column
// matter in this test

class Test {  // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:11
public:
  Test() { }  // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:7
  ~Test() { } // CHECK: rename [[@LINE]]:4 -> [[@LINE]]:8
};

void foo() {
  Test test;  // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:7
}

// RUN: clang-refactor-test rename-initiate-usr -usr="c:@S@Test" -new-name=Foo %s | FileCheck %s

// RUN: not clang-refactor-test rename-initiate-usr -usr="c:@S@Foo" -new-name=Foo %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR1 %s
// CHECK-ERROR1: error: could not rename symbol with the given USR

// RUN: not clang-refactor-test rename-initiate-usr -new-name=Foo %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR2 %s
// CHECK-ERROR2: for the -usr option: must be specified at least once
