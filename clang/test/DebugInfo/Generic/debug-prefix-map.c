// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=%{fs-src-root}UNLIKELY_PATH%{fs-sep}empty %s -emit-llvm -o - | FileCheck %s -check-prefix CHECK-NO-MAIN-FILE-NAME
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=%{fs-src-root}UNLIKELY_PATH=empty %s -emit-llvm -o - | FileCheck %s -check-prefix CHECK-EVIL
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=%{fs-src-root}UNLIKELY_PATH%{fs-sep}empty %s -emit-llvm -o - -main-file-name debug-prefix-map.c | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=%{fs-src-root}UNLIKELY_PATH%{fs-sep}empty %s -emit-llvm -o - -fdebug-compilation-dir %p | FileCheck %s -check-prefix CHECK-COMPILATION-DIR
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=%{fs-src-root}UNLIKELY_PATH%{fs-sep}empty %s -emit-llvm -o - -isysroot %p -debugger-tuning=lldb | FileCheck %s -check-prefix CHECK-SYSROOT
// RUN: %clang -g -fdebug-prefix-map=%p=%{fs-src-root}UNLIKELY_PATH%{fs-sep}empty -S -c %s -emit-llvm -o - | FileCheck %s
// RUN: %clang -g -ffile-prefix-map=%p=%{fs-src-root}UNLIKELY_PATH%{fs-sep}empty -S -c %s -emit-llvm -o - | FileCheck %s

// RUN: %clang -g -fdebug-prefix-map=%p=./UNLIKELY_PATH/empty -S -c %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-REL
// RUN: %clang -g -ffile-prefix-map=%p=./UNLIKELY_PATH/empty -S -c %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-REL

// RUN: rm -rf %t && mkdir -p %t/a/b && cp %s %t/a/b/c.c
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -I%S -fdebug-prefix-map=%t/a/b=y -fdebug-prefix-map=%t/a=x %t/a/b/c.c -o - | FileCheck %s --check-prefix=CHECK-X
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -I%S -fdebug-prefix-map=%t/a=x -fdebug-prefix-map=%t/a/b=y %t/a/b/c.c -o - | FileCheck %s --check-prefix=CHECK-Y
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -I%S -main-file-name %t/a/b/c.c -fdebug-compilation-dir=%t/a -fdebug-prefix-map=%t/a=x -fdebug-prefix-map=%t/a/b=y %t/a/b/c.c -o - | FileCheck %s --check-prefix=CHECK-REMAP-Y

#include "Inputs/stdio.h"

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  return 0;
}

void test_rewrite_includes(void) {
  __builtin_va_list argp;
  vprintf("string", argp);
}

// CHECK-NO-MAIN-FILE-NAME: !DIFile(filename: "{{/|.:\\\\}}UNLIKELY_PATH{{/|\\\\}}empty{{/|\\\\}}{{.*}}",
// CHECK-NO-MAIN-FILE-NAME-SAME:    directory: "")
// CHECK-NO-MAIN-FILE-NAME: !DIFile(filename: "{{/|.:\\\\}}UNLIKELY_PATH{{/|\\\\}}empty{{/|\\\\}}<stdin>",
// CHECK-NO-MAIN-FILE-NAME: !DIFile(filename: "{{/|.:\\\\}}UNLIKELY_PATH{{/|\\\\}}empty{{/|\\\\}}Inputs{{/|\\\\}}stdio.h",
// CHECK-NO-MAIN-FILE-NAME-SAME:    directory: "")
// CHECK-NO-MAIN-FILE-NAME-NOT: !DIFile(filename:

// CHECK-EVIL: !DIFile(filename: "{{/|.:\\\\}}UNLIKELY_PATH=empty{{/|\\\\}}{{.*}}"
// CHECK-EVIL: !DIFile(filename: "{{/|.:\\\\}}UNLIKELY_PATH=empty{{/|\\\\}}{{.*}}Inputs{{/|\\\\}}stdio.h",
// CHECK-EVIL-SAME:    directory: "")
// CHECK-EVIL-NOT: !DIFile(filename:

// CHECK: !DIFile(filename: "{{/|.:\\\\}}UNLIKELY_PATH{{/|\\\\}}empty{{/|\\\\}}{{.*}}",
// CHECK: !DIFile(filename: "{{/|.:\\\\}}UNLIKELY_PATH{{/|\\\\}}empty{{/|\\\\}}{{.*}}Inputs{{/|\\\\}}stdio.h",
// CHECK-SAME:    directory: ""
// CHECK-NOT: !DIFile(filename:

// CHECK-COMPILATION-DIR: !DIFile(filename: "{{.*}}", directory: "{{/|.:\\\\}}UNLIKELY_PATH{{/|\\\\}}empty")
// CHECK-COMPILATION-DIR: !DIFile(filename: "{{.*}}Inputs{{/|\\\\}}stdio.h", directory: "{{/|.:\\\\}}UNLIKELY_PATH{{/|\\\\}}empty")
// CHECK-COMPILATION-DIR-NOT: !DIFile(filename:
// CHECK-SYSROOT: !DICompileUnit({{.*}}sysroot: "{{/|.:\\\\}}UNLIKELY_PATH{{/|\\\\}}empty"

// CHECK-REL: !DIFile(filename: "./UNLIKELY_PATH/empty{{/|\\\\}}{{.*}}",
// CHECK-REL: !DIFile(filename: "./UNLIKELY_PATH/empty{{/|\\\\}}{{.*}}Inputs/stdio.h",
// CHECK-REL-SAME:    directory: ""

// CHECK-X: !DIFile(filename: "x{{/|\\\\}}b{{/|\\\\}}c.c", directory: "")
// CHECK-Y: !DIFile(filename: "y{{/|\\\\}}c.c", directory: "")

// CHECK-REMAP-Y: !DIFile(filename: "y{{/|\\\\}}c.c", directory: "x")
