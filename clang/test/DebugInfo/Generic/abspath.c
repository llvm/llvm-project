// RUN: mkdir -p %t/UNIQUEISH_SENTINEL
// RUN: cp %s %t/UNIQUEISH_SENTINEL/abspath.c

// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   -fdebug-compilation-dir=%t/UNIQUEISH_SENTINEL/abspath.c \
// RUN:   %t/UNIQUEISH_SENTINEL/abspath.c -emit-llvm -o - \
// RUN:   | FileCheck %s

// RUN: cp %s %t.c
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   -fdebug-compilation-dir=%t \
// RUN:   %t.c -emit-llvm -o - | FileCheck %s --check-prefix=INTREE

void foo(void) {}

// Since %s is an absolute path, directory should be the common
// prefix, but the directory part should be part of the filename.

// CHECK: = distinct !DISubprogram({{.*}}file: ![[SPFILE:[0-9]+]]
// CHECK: ![[SPFILE]] = !DIFile(filename: "{{.*}}UNIQUEISH_SENTINEL
// CHECK-SAME:                  abspath.c"
// CHECK-NOT:                   directory: "{{.*}}UNIQUEISH_SENTINEL

// INTREE: = distinct !DISubprogram({{.*}}![[SPFILE:[0-9]+]]
// INTREE: DIFile({{.*}}directory: "{{.+}}Generic{{.*}}")
