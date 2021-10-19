// REQUIRES: asserts

// RUN: %clang_cc1 -mllvm -debug-only=codegenaction -clear-ast-before-backend %s -emit-obj -o /dev/null -O1 2>&1 | FileCheck %s --check-prefix=YES
// RUN: %clang_cc1 -mllvm -debug-only=codegenaction -clear-ast-before-backend -no-clear-ast-before-backend %s -emit-obj -o /dev/null -O1 2>&1 | FileCheck %s --allow-empty --check-prefix=NO
// RUN: %clang_cc1 -clear-ast-before-backend %s -emit-obj -o /dev/null -print-stats 2>&1 | FileCheck %s --check-prefix=STATS

// YES: Clearing AST
// NO-NOT: Clearing AST
// STATS: *** Decl Stats:
// STATS: {{.*}} decls total
// STATS: 1 Function decls
// STATS: Total bytes =

void f() {}
