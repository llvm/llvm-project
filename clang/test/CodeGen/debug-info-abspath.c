// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %s -emit-llvm -o - | FileCheck %s

// RUN: cp %s %t.c
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %t.c -emit-llvm -o - | FileCheck %s --check-prefix=INTREE
void foo() {}

// Since %s is an absolute path, directory should be a nonempty
// prefix, but the CodeGen part should be part of the filename.

// CHECK: DIFile(filename: "{{.*}}CodeGen{{.*}}debug-info-abspath.c"
// CHECK-SAME:   directory: "{{.+}}")

// INTREE: DIFile({{.*}}directory: "{{.+}}CodeGen{{.*}}")
