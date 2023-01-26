// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -emit-llvm -fvirtual-function-elimination -fwhole-program-vtables -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-VFE
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -emit-llvm -fwhole-program-vtables -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOVFE
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-pc-windows-msvc -emit-llvm -fwhole-program-vtables -o - %s | FileCheck %s --check-prefix=CHECK-MS --check-prefix=CHECK-NOVFE

// Check that in ThinLTO we also get vcall_visibility summary entries in the bitcode
// RUN: %clang_cc1 -flto=thin -flto-unit -triple x86_64-unknown-linux -emit-llvm-bc -fwhole-program-vtables -o - %s | llvm-dis -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOVFE --check-prefix=CHECK-SUMMARY


// Anonymous namespace.
namespace {
// CHECK: @_ZTVN12_GLOBAL__N_11AE = {{.*}} !vcall_visibility [[VIS_TU:![0-9]+]]
// CHECK-MS: @anon.{{.*}} = private unnamed_addr constant {{.*}}@"??_R4A@?A{{.*}}@@6B@"{{.*}} !vcall_visibility [[VIS_TU:![0-9]+]]
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
// CHECK-MS: @anon.{{.*}} = private unnamed_addr constant {{.*}}@"??_R4B@@6B@"{{.*}} !vcall_visibility [[VIS_DSO:![0-9]+]]
struct __attribute__((visibility("hidden"))) B {
  B() {}
  virtual int f() { return 1; }
};
B *construct_B() {
  return new B();
}


// Default visibility.
// CHECK-NOT: @_ZTV1C = {{.*}} !vcall_visibility
// On MS default is hidden
// CHECK-MS: @anon.{{.*}} = private unnamed_addr constant {{.*}}@"??_R4C@@6B@"{{.*}} !vcall_visibility [[VIS_DSO]]
struct __attribute__((visibility("default"))) C {
  C() {}
  virtual int f() { return 1; }
};
C *construct_C() {
  return new C();
}


// Hidden visibility, public LTO visibility.
// CHECK-NOT: @_ZTV1D = {{.*}} !vcall_visibility
// CHECK-MS-NOT: @anon.{{.*}} = private unnamed_addr constant {{.*}}@"??_R4D@@6B@"{{.*}} !vcall_visibility
struct __attribute__((visibility("hidden"))) [[clang::lto_visibility_public]] D {
  D() {}
  virtual int f() { return 1; }
};
D *construct_D() {
  return new D();
}


// Hidden visibility, but inherits from class with default visibility.
// CHECK-NOT: @_ZTV1E = {{.*}} !vcall_visibility
// On MS default is hidden
// CHECK-MS: @anon.{{.*}} = private unnamed_addr constant {{.*}}@"??_R4E@@6B@"{{.*}} !vcall_visibility [[VIS_DSO]]
struct __attribute__((visibility("hidden"))) E : C {
  E() {}
  virtual int f() { return 1; }
};
E *construct_E() {
  return new E();
}


// Anonymous namespace, but inherits from class with default visibility.
// CHECK-NOT: @_ZTVN12_GLOBAL__N_11FE = {{.*}} !vcall_visibility
// On MS default is hidden
// CHECK-MS: @anon.{{.*}} = private unnamed_addr constant {{.*}}@"??_R4F@?A{{.*}}@@6B@"{{.*}} !vcall_visibility [[VIS_DSO]]
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
// CHECK-MS: @anon.{{.*}} = private unnamed_addr constant {{.*}}@"??_R4G@?A{{.*}}@@6B@"{{.*}} !vcall_visibility [[VIS_DSO]]
namespace {
struct __attribute__((visibility("hidden"))) G : B {
  G() {}
  virtual int f() { return 1; }
};
}
void *construct_G() {
  return new G();
}

// CHECK-MS-DAG: [[VIS_DSO]] = !{i64 1}
// CHECK-MS-DAG: [[VIS_TU]] = !{i64 2}
// CHECK-DAG: [[VIS_DSO]] = !{i64 1}
// CHECK-DAG: [[VIS_TU]] = !{i64 2}
// CHECK-VFE-DAG: !{i32 1, !"Virtual Function Elim", i32 1}
// CHECK-NOVFE-DAG: !{i32 1, !"Virtual Function Elim", i32 0}

// CHECK-SUMMARY-DAG: gv: (name: "_ZTV1B", {{.*}} vcall_visibility: 1
// CHECK-SUMMARY-DAG: gv: (name: "_ZTVN12_GLOBAL__N_11FE", {{.*}} vcall_visibility: 0
// CHECK-SUMMARY-DAG: gv: (name: "_ZTV1D", {{.*}} vcall_visibility: 0
// CHECK-SUMMARY-DAG: gv: (name: "_ZTV1C", {{.*}} vcall_visibility: 0
// CHECK-SUMMARY-DAG: gv: (name: "_ZTV1E", {{.*}} vcall_visibility: 0
// CHECK-SUMMARY-DAG: gv: (name: "_ZTVN12_GLOBAL__N_11AE", {{.*}} vcall_visibility: 2
// CHECK-SUMMARY-DAG: gv: (name: "_ZTVN12_GLOBAL__N_11GE", {{.*}} vcall_visibility: 1
