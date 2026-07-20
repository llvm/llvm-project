// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

// std::core_ub / {expr.mul.div.by.zero} (P4317 A.1): under enforcement an
// integer division verifies its divisor and traps on zero, with no -fsanitize
// flag. Only the divide-by-zero check is emitted here; the INT_MIN/-1 signed
// overflow arm belongs to {expr.mul.representable.type.result}, guarded
// separately. Without -fprofiles the profile is inert.

[[profiles::enforce(std::core_ub)]];

// CHECK-LABEL: define {{.*}}@_Z6divideii
// CHECK: icmp ne i32 %{{.*}}, 0
// CHECK: call void @llvm.ubsantrap(i8 3)
// OFF-LABEL: define {{.*}}@_Z6divideii
// OFF-NOT: llvm.ubsantrap
int divide(int a, int b) { return a / b; }

// A remainder is the same guarded operation.
// CHECK-LABEL: define {{.*}}@_Z9remainderii
// CHECK: call void @llvm.ubsantrap(i8 3)
int remainder(int a, int b) { return a % b; }
