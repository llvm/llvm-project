// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext -plugin print-fns %s 2>&1 | FileCheck %s
// RUN: %clang_cl -c -Xclang -load -Xclang %llvmshlibdir/PrintFunctionNames%pluginext -Xclang -plugin -Xclang print-fns -Tc %s 2>&1 | FileCheck %s
// REQUIRES: plugins, examples

// CHECK: top-level-decl: "x"
void x();
