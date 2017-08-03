// XFAIL: linux

#include "header.h"

void foo(void) {
  bar();
}

// RUN: rm -rf %t
// RUN: mkdir -p %t/Directory
// RUN: mkdir -p %t/Directory.surprise
// RUN: mkdir -p %t/sdk
// RUN: mkdir -p %t/sdk_other
// RUN: echo "void bar(void);" > %t/sdk_other/header.h
// RUN: cp %s %t/Directory.surprise/main.c
//
// RUN: %clang_cc1 -isystem %t/sdk_other -isysroot %t/sdk -index-store-path %t/idx %t/Directory.surprise/main.c -triple x86_64-apple-macosx10.8 -working-directory %t/Directory
// RUN: c-index-test core -print-unit %t/idx | FileCheck %s

// CHECK: main.c.o
// CHECK: provider: clang-
// CHECK: is-system: 0
// CHECK: has-main: 1
// CHECK: main-path: {{.*}}Directory.surprise{{/|\\}}main.c
// CHECK: out-file: {{.*}}Directory.surprise{{/|\\}}main.c.o
// CHECK: target: x86_64-apple-macosx10.8
// CHECK: is-debug: 1
// CHECK: DEPEND START
// CHECK: Record | user | {{.*}}Directory.surprise{{/|\\}}main.c | main.c-
// CHECK: Record | system | {{.*}}sdk_other{{/|\\}}header.h | header.h-
