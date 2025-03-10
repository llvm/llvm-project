// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++23 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++17 %s -emit-llvm -o - | FileCheck %s

extern int& s;

// CHECK: @_Z4testv()
// CHECK-NEXT: entry:
// CHECK-NEXT: [[I:%.*]] = alloca ptr, align {{.*}}
// CHECK-NEXT: [[X:%.*]] = load ptr, ptr @s, align {{.*}}
// CHECK-NEXT: store ptr [[X]], ptr [[I]], align {{.*}}
int& test() {
  auto &i = s;
  return i;
}
