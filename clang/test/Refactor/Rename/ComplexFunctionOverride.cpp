struct A {
  virtual void foo() {} /* Test 1 */    // CHECK: rename [[@LINE]]:16 -> [[@LINE]]:19
};

struct B : A {
  void foo() override {} /* Test 2 */   // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:11
};

struct C : B {
  void foo() override {} /* Test 3 */   // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:11
};

struct D : B {
  void foo() override {} /* Test 4 */   // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:11
};

struct E : D {
  void foo() override {} /* Test 5 */   // CHECK: rename [[@LINE]]:8 -> [[@LINE]]:11
};

int main() {
  A a;
  a.foo();                              // CHECK: rename [[@LINE]]:5 -> [[@LINE]]:8
  B b;
  b.foo();                              // CHECK: rename [[@LINE]]:5 -> [[@LINE]]:8
  C c;
  c.foo();                              // CHECK: rename [[@LINE]]:5 -> [[@LINE]]:8
  D d;
  d.foo();                              // CHECK: rename [[@LINE]]:5 -> [[@LINE]]:8
  E e;
  e.foo();                              // CHECK: rename [[@LINE]]:5 -> [[@LINE]]:8
  return 0;
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:2:16 -new-name=bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:6:8 -new-name=bar %s | FileCheck %s
// Test 3.
// RUN: clang-refactor-test rename-initiate -at=%s:10:8 -new-name=bar %s | FileCheck %s
// Test 4.
// RUN: clang-refactor-test rename-initiate -at=%s:14:8 -new-name=bar %s | FileCheck %s
// Test 5.
// RUN: clang-refactor-test rename-initiate -at=%s:18:8 -new-name=bar %s | FileCheck %s

// Check virtual inheritance

struct A2 {
  virtual void foo() {}   // CHECK-VIRT: rename [[@LINE]]:16 -> [[@LINE]]:19
};
struct B2 : virtual A2 {
  void foo() { }          // CHECK-VIRT: rename [[@LINE]]:8 -> [[@LINE]]:11
};
struct C2 : virtual A2 {
  void foo() override { } // CHECK-VIRT: rename [[@LINE]]:8 -> [[@LINE]]:11
};
struct D2 : B2, C2 {
  void foo() override {  // CHECK-VIRT: rename [[@LINE]]:8 -> [[@LINE]]:11
    A2::foo();           // CHECK-VIRT: rename [[@LINE]]:9 -> [[@LINE]]:12
  }
};

int bar() {
  A2 a;
  a.foo(); // CHECK-VIRT: rename [[@LINE]]:5 -> [[@LINE]]:8
  D2 d;
  d.foo(); // CHECK-VIRT: rename [[@LINE]]:5 -> [[@LINE]]:8
}

// RUN: clang-refactor-test rename-initiate -at=%s:49:16 -new-name=bar %s | FileCheck --check-prefix=CHECK-VIRT %s
// RUN: clang-refactor-test rename-initiate -at=%s:52:8 -new-name=bar %s | FileCheck --check-prefix=CHECK-VIRT %s
// RUN: clang-refactor-test rename-initiate -at=%s:55:8 -new-name=bar %s | FileCheck --check-prefix=CHECK-VIRT %s
// RUN: clang-refactor-test rename-initiate -at=%s:58:8 -new-name=bar %s | FileCheck --check-prefix=CHECK-VIRT %s
// RUN: clang-refactor-test rename-initiate -at=%s:59:9 -new-name=bar %s | FileCheck --check-prefix=CHECK-VIRT %s
// RUN: clang-refactor-test rename-initiate -at=%s:65:5 -new-name=bar %s | FileCheck --check-prefix=CHECK-VIRT %s
// RUN: clang-refactor-test rename-initiate -at=%s:67:5 -new-name=bar %s | FileCheck --check-prefix=CHECK-VIRT %s
