// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

// std::core_ub / {expr.shift.neg.and.width} (P4317 A.1): a shift whose right
// operand is negative or at least the width of the left operand, or a signed
// left shift that shifts bits out, is undefined; under enforcement the shift
// checks its operands and traps.

[[profiles::enforce(std::core_ub)]];

// CHECK-LABEL: define {{.*}}@_Z3shlii
// CHECK: call void @llvm.ubsantrap(i8 20)
// OFF-LABEL: define {{.*}}@_Z3shlii
// OFF-NOT: llvm.ubsantrap
int shl(int a, int b) { return a << b; }

// CHECK-LABEL: define {{.*}}@_Z3shrii
// CHECK: call void @llvm.ubsantrap(i8 20)
int shr(int a, int b) { return a >> b; }
