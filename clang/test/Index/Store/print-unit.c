// XFAIL: linux

#include "print-unit.h"
#include "syshead.h"

void foo(int i);

// RUN: rm -rf %t
// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -index-store-path %t/idx %s -triple x86_64-apple-macosx10.8
// RUN: c-index-test core -print-unit %t/idx | FileCheck %s
// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -index-store-path %t/idx_opt1 %s -triple x86_64-apple-macosx10.8 -O2
// RUN: c-index-test core -print-unit %t/idx_opt1 | FileCheck %s -check-prefix=OPT
// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -index-store-path %t/idx_opt2 %s -triple x86_64-apple-macosx10.8 -Os
// RUN: c-index-test core -print-unit %t/idx_opt2 | FileCheck %s -check-prefix=OPT

// CHECK: print-unit.c.o
// CHECK: provider: clang-
// CHECK: is-system: 0
// CHECK: has-main: 1
// CHECK: main-path: {{.*}}/print-unit.c
// CHECK: out-file: {{.*}}/print-unit.c.o
// CHECK: target: x86_64-apple-macosx10.8
// CHECK: is-debug: 1
// CHECK: DEPEND START
// CHECK: Record | user | {{.*}}/print-unit.c | print-unit.c-
// CHECK: Record | user | {{.*}}/Inputs/head.h | head.h-
// CHECK: Record | user | {{.*}}/Inputs/using-overlay.h | using-overlay.h-
// CHECK: Record | system | {{.*}}/Inputs/sys/syshead.h | syshead.h-
// CHECK: Record | system | {{.*}}/Inputs/sys/another.h | another.h-
// CHECK: File | user | {{.*}}/Inputs/print-unit.h | | {{[0-9]*$}}
// CHECK: DEPEND END (6)
// CHECK: INCLUDE START
// CHECK: {{.*}}/print-unit.c:3 | {{.*}}/Inputs/print-unit.h
// CHECK: {{.*}}/print-unit.c:4 | {{.*}}/Inputs/sys/syshead.h
// CHECK: {{.*}}/Inputs/print-unit.h:1 | {{.*}}/Inputs/head.h
// CHECK: {{.*}}/Inputs/print-unit.h:2 | {{.*}}/Inputs/using-overlay.h
// CHECK: INCLUDE END (4)

// OPT: is-debug: 0
