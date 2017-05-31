// RUN: %clang_cc1 %s -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm -o - | FileCheck %s

struct A {
  virtual void f();
};

struct B {
  virtual void f();
};

struct C : A, B {
  virtual void f();
};

void C::f() { }

// CHECK: !DISubprogram(linkageName: "_ZThn{{[48]}}_N1C1fEv"
// CHECK-SAME:          line: 15
// CHECK-SAME:          isDefinition: true
// CHECK-SAME:          ){{$}}
