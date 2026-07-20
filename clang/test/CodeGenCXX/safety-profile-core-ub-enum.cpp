// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

// std::core_ub / {expr.static.cast.enum.outside.range} (P4317 A.1): producing
// an enumeration value outside the range of its enumerators is undefined; under
// enforcement a load of an enum value is range-checked and traps. (The earlier
// trap here is the null/alignment check on the pointer; the enum check is the
// one with handler id 10.)

[[profiles::enforce(std::core_ub)]];

enum E { A, B, C };

// CHECK-LABEL: define {{.*}}@_Z3useP1E
// CHECK: icmp ule i32 %{{.*}}, 3
// CHECK: call void @llvm.ubsantrap(i8 10)
// OFF-LABEL: define {{.*}}@_Z3useP1E
// OFF-NOT: llvm.ubsantrap
int use(E *p) { return *p; }
