// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

// std::core_ub / {conv.fpint.*}, {conv.double.out.of.range} (P4317 A.1):
// converting a floating-point value whose truncated value does not fit the
// destination integer type is undefined; under enforcement the conversion
// range-checks the source and traps.

[[profiles::enforce(std::core_ub)]];

// CHECK-LABEL: define {{.*}}@_Z3f2id
// CHECK: fcmp
// CHECK: call void @llvm.ubsantrap(i8 5)
// CHECK: fptosi double %{{.*}} to i32
// OFF-LABEL: define {{.*}}@_Z3f2id
// OFF-NOT: llvm.ubsantrap
int f2i(double d) { return (int)d; }
