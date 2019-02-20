// RUN: mkdir -p %t/UNIQUEISH_SENTINEL
// RUN: cp %s %t/UNIQUEISH_SENTINEL/debug-info-abspath.c

// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %t/UNIQUEISH_SENTINEL/debug-info-abspath.c -emit-llvm -o - \
// RUN:   | FileCheck %s

// RUN: cp %s %t.c
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %t.c -emit-llvm -o - | FileCheck %s --check-prefix=INTREE

// RUN: cd %t/UNIQUEISH_SENTINEL
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   debug-info-abspath.c -emit-llvm -o - \
// RUN:   | FileCheck %s --check-prefix=CURDIR
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %s -emit-llvm -o - | FileCheck %s --check-prefix=CURDIR

void foo() {}

// Since %s is an absolute path, directory should be the common
// prefix, but the directory part should be part of the filename.

// CHECK: = distinct !DISubprogram({{.*}}file: ![[SPFILE:[0-9]+]]
// CHECK: ![[SPFILE]] = !DIFile(filename: "{{.*}}UNIQUEISH_SENTINEL
// CHECK-SAME:                  debug-info-abspath.c"
// CHECK-NOT:                   directory: "{{.*}}UNIQUEISH_SENTINEL

// INTREE: = distinct !DISubprogram({{.*}}![[SPFILE:[0-9]+]]
// INTREE: DIFile({{.*}}directory: "{{.+}}CodeGen{{.*}}")

// CURDIR: = distinct !DICompileUnit({{.*}}file: ![[CUFILE:[0-9]+]]
// CURDIR: ![[CUFILE]] = !DIFile({{.*}}directory: "{{.+}}UNIQUEISH_SENTINEL")

