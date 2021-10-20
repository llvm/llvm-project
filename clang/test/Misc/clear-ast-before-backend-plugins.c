// REQUIRES: plugins, examples, asserts

// RUN: %clang_cc1 -mllvm -debug-only=codegenaction -clear-ast-before-backend -emit-obj -o /dev/null -load %llvmshlibdir/PrintFunctionNames%pluginext %s 2>&1 | FileCheck %s --check-prefix=YES
// YES: Clearing AST

// RUN: %clang_cc1 -mllvm -debug-only=codegenaction -clear-ast-before-backend -emit-obj -o /dev/null -load %llvmshlibdir/PrintFunctionNames%pluginext -add-plugin print-fns -plugin-arg-print-fns help %s 2>&1 | FileCheck %s --check-prefix=NO
// NO-NOT: Clearing AST

void f() {}
