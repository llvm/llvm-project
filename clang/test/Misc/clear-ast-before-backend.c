// RUN: %clang_cc1 -clear-ast-before-backend %s -emit-obj -o /dev/null -O1
// RUN: %clang_cc1 -clear-ast-before-backend %s -emit-obj -o /dev/null -print-stats 2>&1 | FileCheck %s

// CHECK: *** Decl Stats:
// CHECK: {{.*}} decls total
// CHECK: 1 Function decls
// CHECK: Total bytes =

void f() {}
