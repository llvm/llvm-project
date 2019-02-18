// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: cp %s rel.c
// RUN: %clang_cc1 -fdebug-compilation-dir /nonsense -emit-llvm -debug-info-kind=limited rel.c -o - | FileCheck -check-prefix=CHECK-NONSENSE %s
// CHECK-NONSENSE: nonsense

// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck -check-prefix=CHECK-DIR %s
// CHECK-DIR: CodeGen

