struct A {
  virtual void foo() {} /* Test 1 */    // CHECK: virtual void bar() {}
};

struct B : A {
  void foo() override {} /* Test 2 */   // CHECK: void bar() override {}
};

struct C : B {
  void foo() override {} /* Test 3 */   // CHECK: void bar() override {}
};

struct D : B {
  void foo() override {} /* Test 4 */   // CHECK: void bar() override {}
};

struct E : D {
  void foo() override {} /* Test 5 */   // CHECK: void bar() override {}
};

int main() {
  A a;
  a.foo();                              // CHECK: a.bar();
  B b;
  b.foo();                              // CHECK: b.bar();
  C c;
  c.foo();                              // CHECK: c.bar();
  D d;
  d.foo();                              // CHECK: d.bar();
  E e;
  e.foo();                              // CHECK: e.bar();
  return 0;
}

// Test 1.
// RUN: clang-rename -offset=26 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=109 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=201 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 4.
// RUN: clang-rename -offset=293 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 5.
// RUN: clang-rename -offset=385 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'foo.*' <file>
