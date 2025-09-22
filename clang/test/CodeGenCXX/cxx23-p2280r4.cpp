// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++23 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++17 %s -emit-llvm -o - | FileCheck %s

extern int& s;

// CHECK-LABEL: @_Z4testv()
// CHECK-NEXT: entry:
// CHECK-NEXT: [[I:%.*]] = alloca ptr, align {{.*}}
// CHECK-NEXT: [[X:%.*]] = load ptr, ptr @s, align {{.*}}
// CHECK-NEXT: store ptr [[X]], ptr [[I]], align {{.*}}
int& test() {
  auto &i = s;
  return i;
}

// CHECK-LABEL: @_Z1fv(
// CHECK: [[X1:%.*]] = load ptr, ptr @x, align {{.*}}
// CHECK-NEXT: store ptr [[X1]]
// CHECK: [[X2:%.*]] = load ptr, ptr @x, align {{.*}}
// CHECK-NEXT: store ptr [[X2]]
// CHECK: [[X3:%.*]] = load ptr, ptr @x, align {{.*}}
// CHECK-NEXT: store ptr [[X3]]
int &ff();
int &x = ff();
struct A { int& x; };
struct B { A x[20]; };
B f() { return {x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x}; }
