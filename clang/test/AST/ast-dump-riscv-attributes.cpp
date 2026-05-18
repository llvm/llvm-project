// RUN: %clang_cc1 -triple riscv64 -ast-dump -ast-dump-filter c23 -std=c23 -x c %s | FileCheck --strict-whitespace %s

// CHECK:       FunctionDecl{{.*}}pre_c23
// CHECK-NEXT:    |-CompoundStmt
// CHECK-NEXT:    `-RISCVInterruptAttr{{.*}}supervisor
__attribute__((interrupt("supervisor"))) void pre_c23(){}

// CHECK:       FunctionDecl{{.*}}in_c23
// CHECK-NEXT:    |-CompoundStmt
// CHECK-NEXT:    `-RISCVInterruptAttr{{.*}}supervisor
// CHECK-NOT:     `-RISCVInterruptAttr{{.*}}machine
[[gnu::interrupt("supervisor")]] void in_c23(){}
