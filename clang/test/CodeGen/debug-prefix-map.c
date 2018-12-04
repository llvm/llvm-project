// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=/var/empty %s -emit-llvm -o - | FileCheck %s -check-prefix CHECK-NO-MAIN-FILE-NAME
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=/var=empty %s -emit-llvm -o - | FileCheck %s -check-prefix CHECK-EVIL
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=/var/empty %s -emit-llvm -o - -main-file-name debug-prefix-map.c | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=/var/empty %s -emit-llvm -o - -fdebug-compilation-dir %p | FileCheck %s -check-prefix CHECK-COMPILATION-DIR

#include "Inputs/stdio.h"

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  return 0;
}

void test_rewrite_includes() {
  __builtin_va_list argp;
  vprintf("string", argp);
}

// CHECK-NO-MAIN-FILE-NAME: !DIFile(filename: "/var/empty{{/|\\5C}}<stdin>"
// CHECK-NO-MAIN-FILE-NAME: !DIFile(filename: "/var/empty{{[/\\]}}{{.*}}"
// CHECK-NO-MAIN-FILE-NAME: !DIFile(filename: "/var/empty{{[/\\]}}Inputs/stdio.h"
// CHECK-NO-MAIN-FILE-NAME-NOT: !DIFile(filename:

// CHECK-EVIL: !DIFile(filename: "/var=empty{{[/\\]}}{{.*}}"
// CHECK-EVIL: !DIFile(filename: "/var=empty{{[/\\]}}Inputs/stdio.h"
// CHECK-EVIL-NOT: !DIFile(filename:

// CHECK: !DIFile(filename: "/var/empty{{[/\\]}}{{.*}}"
// CHECK: !DIFile(filename: "/var/empty{{[/\\]}}Inputs/stdio.h"
// CHECK-NOT: !DIFile(filename:

// CHECK-COMPILATION-DIR: !DIFile(filename: "/var/empty{{[/\\]}}{{.*}}", directory: "/var/empty")
// CHECK-COMPILATION-DIR: !DIFile(filename: "/var/empty{{[/\\]}}Inputs/stdio.h", directory: "/var/empty")
// CHECK-COMPILATION-DIR-NOT: !DIFile(filename:
