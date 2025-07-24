// RUN: %clang_cc1 -triple riscv64 -ast-dump -ast-dump-filter c23 -std=c23 -x c %s | FileCheck --strict-whitespace %s

// CHECK: FunctionDecl {{.*}} pre_c23 'void (void)'
// CHECK-NEXT: |-CompoundStmt {{.*}}
// CHECK-NEXT: `-attrDetails: RISCVInterruptAttr {{.*}} supervisor
__attribute__((interrupt("supervisor"))) void pre_c23(){}

// CHECK: FunctionDecl {{.*}} in_c23 'void (void)'
// CHECK-NEXT: |-CompoundStmt {{.*}}
// CHECK-NEXT: `-attrDetails: RISCVInterruptAttr {{.*}} supervisor
[[gnu::interrupt("supervisor")]] void in_c23(){}
