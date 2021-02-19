// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: cp %s rel.c
// RUN: %clang_cc1 -fprofile-instrument=clang -fprofile-compilation-dir=/nonsense -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false rel.c -o - | FileCheck -check-prefix=CHECK-NONSENSE %s

// CHECK-NONSENSE: nonsense

void f() {}
