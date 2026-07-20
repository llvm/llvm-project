// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

// std::core_ub / {expr.unary.dereference} null case (P4317 A.1): dereferencing
// a null pointer is undefined; under enforcement the access checks the pointer
// against null and traps.

[[profiles::enforce(std::core_ub)]];

// CHECK-LABEL: define {{.*}}@_Z5derefPi
// CHECK: icmp ne ptr %{{.*}}, null
// CHECK: call void @llvm.ubsantrap(i8 22)
// OFF-LABEL: define {{.*}}@_Z5derefPi
// OFF-NOT: llvm.ubsantrap
int deref(int *p) { return *p; }
