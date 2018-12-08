// RUN: mkdir -p %t/UNIQUEISH_SENTINEL
// RUN: cp %s %t/UNIQUEISH_SENTINEL/debug-info-abspath.c

// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %t/UNIQUEISH_SENTINEL/debug-info-abspath.c -emit-llvm -o - \
// RUN:   | FileCheck %s

// RUN: cp %s %t.c
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %t.c -emit-llvm -o - | FileCheck %s --check-prefix=INTREE
void foo() {}

// Since %s is an absolute path, directory should be the common
// prefix, but the directory part should be part of the filename.

// CHECK: DIFile(filename: "{{.*}}UNIQUEISH_SENTINEL{{.*}}debug-info-abspath.c"
// CHECK-NOT:    directory: "{{.*}}UNIQUEISH_SENTINEL

// INTREE: DIFile({{.*}}directory: "{{.+}}CodeGen{{.*}}")
