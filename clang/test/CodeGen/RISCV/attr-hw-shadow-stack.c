// RUN: %clang_cc1 -triple riscv64 -target-feature +zimop -emit-llvm -o - %s -fcf-protection=return | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +zimop -emit-llvm -o - %s | FileCheck -check-prefix=NOSHADOWSTACK %s
// RUN: %clang_cc1 -triple riscv32 -target-feature +zimop -emit-llvm -o - %s -fcf-protection=return | FileCheck %s
// RUN: %clang_cc1 -triple riscv32 -target-feature +zimop -emit-llvm -o - %s | FileCheck -check-prefix=NOSHADOWSTACK %s

int foo(int *a) { return *a; }

// CHECK: attributes {{.*}}"hw-shadow-stack"{{.*}}
// CHECK: !{i32 8, !"hw-shadow-stack", i32 1}
// NOSHADOWSTACK-NOT: "hw-shadow-stack"
