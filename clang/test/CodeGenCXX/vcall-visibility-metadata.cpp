// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -emit-llvm -fvirtual-function-elimination -fwhole-program-vtables -o - %s | FileCheck %s


// Anonymous namespace.
namespace {
// CHECK: @_ZTVN12_GLOBAL__N_11AE = {{.*}} !vcall_visibility [[VIS_TU:![0-9]+]]
struct A {
  A() {}
  virtual int f() { return 1; }
};
}
void *construct_A() {
  return new A();
}


// Hidden visibility.
// CHECK: @_ZTV1B = {{.*}} !vcall_visibility [[VIS_DSO:![0-9]+]]
struct __attribute__((visibility("hidden"))) B {
  B() {}
  virtual int f() { return 1; }
};
B *construct_B() {
  return new B();
}


// Default visibility.
// CHECK-NOT: @_ZTV1C = {{.*}} !vcall_visibility
struct __attribute__((visibility("default"))) C {
  C() {}
  virtual int f() { return 1; }
};
C *construct_C() {
  return new C();
}


// Hidden visibility, public LTO visibility.
// CHECK-NOT: @_ZTV1D = {{.*}} !vcall_visibility
struct __attribute__((visibility("hidden"))) [[clang::lto_visibility_public]] D {
  D() {}
  virtual int f() { return 1; }
};
D *construct_D() {
  return new D();
}


// Hidden visibility, but inherits from class with default visibility.
// CHECK-NOT: @_ZTV1E = {{.*}} !vcall_visibility
struct __attribute__((visibility("hidden"))) E : C {
  E() {}
  virtual int f() { return 1; }
};
E *construct_E() {
  return new E();
}


// Anonymous namespace, but inherits from class with default visibility.
// CHECK-NOT: @_ZTVN12_GLOBAL__N_11FE = {{.*}} !vcall_visibility
namespace {
struct __attribute__((visibility("hidden"))) F : C {
  F() {}
  virtual int f() { return 1; }
};
}
void *construct_F() {
  return new F();
}


// Anonymous namespace, but inherits from class with hidden visibility.
// CHECK: @_ZTVN12_GLOBAL__N_11GE = {{.*}} !vcall_visibility [[VIS_DSO:![0-9]+]]
namespace {
struct __attribute__((visibility("hidden"))) G : B {
  G() {}
  virtual int f() { return 1; }
};
}
void *construct_G() {
  return new G();
}


// CHECK-DAG: [[VIS_DSO]] = !{i64 1}
// CHECK-DAG: [[VIS_TU]] = !{i64 2}
